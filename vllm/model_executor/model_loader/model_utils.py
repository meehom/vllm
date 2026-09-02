import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import logger


def _get_attr(obj: Any, name: str, default=None):
    """Safely get an attribute from a vLLM parameter."""
    return getattr(obj, name, default)


def _get_parameter_metadata(param: torch.nn.Parameter) -> dict:
    """
    Collect TP-related metadata attached to a vLLM parameter.

    vLLM 0.27.1 parameters may carry attributes such as:
        tp_rank
        tp_size
        input_dim
        output_dim

    Custom weight loaders may also expose other attributes.
    """

    metadata = {
        "parameter_class": param.__class__.__name__,
        "shape": list(param.shape),
        "dtype": str(param.dtype),
    }

    # vLLM TP metadata
    for attr in (
        "tp_rank",
        "tp_size",
        "input_dim",
        "output_dim",
        "packed_dim",
    ):
        value = _get_attr(param, attr, None)

        if value is not None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = str(value)

            metadata[attr] = value

    # Some custom parameters use these attributes.
    for attr in (
        "is_parallel",
        "partition_dim",
        "shard_id",
        "num_heads",
        "num_kv_heads",
    ):
        value = _get_attr(param, attr, None)

        if value is not None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = str(value)

            metadata[attr] = value

    return metadata


def save_model_weights(
    model: torch.nn.Module,
    output_dir: str = "./Qwen3.8-27B-MLA-random",
) -> None:
    """
    Save the local TP shard of the current model.

    This function intentionally DOES NOT perform all-gather.

    For TP=N, every rank writes:

        model-rank-0.safetensors
        model-rank-1.safetensors
        ...
        model-rank-N.safetensors

    and rank-local TP metadata is stored in:

        tp_metadata-rank-0.json
        tp_metadata-rank-1.json
        ...

    The checkpoint can later be reconstructed by
    merge_tp_checkpoint.py.

    Requirements:
        TP >= 1
        PP = 1
        unquantized model recommended
    """

    tp_rank = get_tensor_model_parallel_rank()
    tp_size = get_tensor_model_parallel_world_size()

    output_path = Path(output_dir)
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "[MLA SAVE] Saving TP rank %d/%d",
        tp_rank,
        tp_size,
    )

    state_dict = {}
    metadata = {}

    for name, param in model.named_parameters():

        # ------------------------------------------------------------
        # Save the local TP tensor exactly as vLLM currently owns it.
        # ------------------------------------------------------------

        tensor = (
            param.detach()
            .cpu()
            .contiguous()
        )

        state_dict[name] = tensor

        # ------------------------------------------------------------
        # Save TP metadata.
        # ------------------------------------------------------------

        metadata[name] = _get_parameter_metadata(param)

    # ------------------------------------------------------------
    # Save local safetensors.
    # ------------------------------------------------------------

    weight_file = (
        output_path
        / f"model-rank-{tp_rank}.safetensors"
    )

    save_file(
        state_dict,
        str(weight_file),
    )

    # ------------------------------------------------------------
    # Save metadata.
    # ------------------------------------------------------------

    metadata_file = (
        output_path
        / f"tp_metadata-rank-{tp_rank}.json"
    )

    metadata_payload = {
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "parameters": metadata,
    }

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata_payload,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------
    # MLA verification.
    # ------------------------------------------------------------

    mla_params = {
        name: tensor
        for name, tensor in state_dict.items()
        if ".mla." in name
    }

    logger.info(
        "[MLA SAVE] rank=%d saved %d tensors",
        tp_rank,
        len(state_dict),
    )

    logger.info(
        "[MLA SAVE] rank=%d found %d MLA tensors",
        tp_rank,
        len(mla_params),
    )

    for name, tensor in list(mla_params.items())[:20]:
        logger.info(
            "[MLA SAVE] rank=%d %s -> shape=%s",
            tp_rank,
            name,
            tuple(tensor.shape),
        )

    logger.info(
        "[MLA SAVE] rank=%d weight file: %s",
        tp_rank,
        weight_file,
    )

    logger.info(
        "[MLA SAVE] rank=%d metadata file: %s",
        tp_rank,
        metadata_file,
    )