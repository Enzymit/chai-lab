# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

"""Batch inference: run predictions on multiple fasta files across GPUs."""

import logging
import multiprocessing as mp
import traceback
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class BatchResult:
    """Result of processing a single fasta file."""

    fasta_file: Path
    output_dir: Path
    success: bool
    error_message: str | None = None


def _discover_fasta_files(input_dir: Path) -> list[Path]:
    """Find all .fasta and .fa files in the input directory (non-recursive)."""
    extensions = {".fasta", ".fa"}
    fasta_files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions
    )
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta or .fa files found in {input_dir}")
    return fasta_files


def _parse_devices(devices: str | None) -> list[str]:
    """Parse device specification into a list of CUDA device strings.

    If devices is None, use all available CUDA devices.
    Otherwise parse comma-separated device indices like "0,1,3".
    """
    if devices is None:
        count = torch.cuda.device_count()
        if count == 0:
            raise RuntimeError("No CUDA devices available")
        return [f"cuda:{i}" for i in range(count)]

    indices = [int(x.strip()) for x in devices.split(",")]
    available = torch.cuda.device_count()
    for idx in indices:
        if idx < 0 or idx >= available:
            raise ValueError(
                f"Device index {idx} out of range. " f"Available devices: 0-{available - 1}"
            )
    return [f"cuda:{i}" for i in indices]


def _worker_loop(
    device: str,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    inference_kwargs: dict,
):
    """Worker process main loop.

    Pulls fasta files from task_queue, runs inference on the assigned device,
    and posts results to result_queue. Each worker process has its own
    _component_cache and _esm_model (fresh via spawn start method).
    """
    from chai_lab.chai1 import run_inference

    while True:
        item = task_queue.get()
        if item is None:
            break

        fasta_file, output_dir = item
        try:
            logger.info(f"[{device}] Processing {fasta_file.name} -> {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)
            run_inference(
                fasta_file=fasta_file,
                output_dir=output_dir,
                device=device,
                **inference_kwargs,
            )
            result_queue.put(
                BatchResult(
                    fasta_file=fasta_file,
                    output_dir=output_dir,
                    success=True,
                )
            )
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"[{device}] Failed on {fasta_file.name}: {e}\n{tb}")
            result_queue.put(
                BatchResult(
                    fasta_file=fasta_file,
                    output_dir=output_dir,
                    success=False,
                    error_message=f"{type(e).__name__}: {e}",
                )
            )


def run_batch_inference(
    input_dir: Path,
    *,
    output_dir: Path,
    devices: str | None = None,
    # Pass-through parameters matching run_inference signature
    use_esm_embeddings: bool = True,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_directory: Path | None = None,
    constraint_path: Path | None = None,
    use_templates_server: bool = False,
    template_hits_path: Path | None = None,
    recycle_msa_subsample: int = 0,
    num_trunk_recycles: int = 3,
    num_diffn_timesteps: int = 200,
    num_diffn_samples: int = 5,
    num_trunk_samples: int = 1,
    seed: int | None = None,
    low_memory: bool = True,
    fasta_names_as_cif_chains: bool = False,
) -> list[BatchResult]:
    """Run inference on all fasta files in input_dir, distributed across GPUs.

    Each fasta file's output goes to output_dir/<fasta_stem>/.

    Args:
        input_dir: Directory containing .fasta/.fa files.
        output_dir: Base output directory. Subdirectories created per fasta.
        devices: Comma-separated GPU indices (e.g. "0,1,3").
                 Defaults to all available CUDA devices.
        (remaining args): Same as run_inference.

    Returns:
        List of BatchResult with success/failure status per file.
    """
    fasta_files = _discover_fasta_files(input_dir)
    device_list = _parse_devices(devices)

    logger.info(
        f"Batch inference: {len(fasta_files)} fasta files "
        f"across {len(device_list)} devices: {device_list}"
    )

    # Build work items
    work_items: list[tuple[Path, Path]] = []
    for fasta_file in fasta_files:
        sub_output = output_dir / fasta_file.stem
        work_items.append((fasta_file, sub_output))

    # Collect inference kwargs (everything except fasta_file, output_dir, device)
    inference_kwargs = dict(
        use_esm_embeddings=use_esm_embeddings,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_directory=msa_directory,
        constraint_path=constraint_path,
        use_templates_server=use_templates_server,
        template_hits_path=template_hits_path,
        recycle_msa_subsample=recycle_msa_subsample,
        num_trunk_recycles=num_trunk_recycles,
        num_diffn_timesteps=num_diffn_timesteps,
        num_diffn_samples=num_diffn_samples,
        num_trunk_samples=num_trunk_samples,
        seed=seed,
        low_memory=low_memory,
        fasta_names_as_cif_chains=fasta_names_as_cif_chains,
    )

    # Use 'spawn' to avoid CUDA fork issues
    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Enqueue all work items
    for item in work_items:
        task_queue.put(item)

    # Add poison pills (one per worker)
    num_workers = min(len(device_list), len(work_items))
    for _ in range(num_workers):
        task_queue.put(None)

    # Spawn one worker per device
    workers = []
    for i in range(num_workers):
        device = device_list[i]
        p = ctx.Process(
            target=_worker_loop,
            args=(device, task_queue, result_queue, inference_kwargs),
            name=f"chai-worker-{device}",
        )
        p.start()
        workers.append(p)

    # Collect results
    results: list[BatchResult] = []
    for _ in range(len(work_items)):
        result = result_queue.get()
        status = "OK" if result.success else f"FAILED: {result.error_message}"
        logger.info(f"  {result.fasta_file.name}: {status}")
        results.append(result)

    # Wait for workers to finish
    for p in workers:
        p.join(timeout=30)
        if p.is_alive():
            logger.warning(f"Worker {p.name} did not exit, terminating")
            p.terminate()
            p.join()

    # Summary
    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)
    logger.info(
        f"Batch complete: {succeeded} succeeded, {failed} failed "
        f"out of {len(results)} total"
    )

    if failed > 0:
        for r in results:
            if not r.success:
                logger.error(f"  FAILED: {r.fasta_file.name} - {r.error_message}")

    return results
