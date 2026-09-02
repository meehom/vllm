#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge TP-sharded vLLM checkpoints into "
            "a full checkpoint."
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
    )

    parser.add_argument(
        "--tp_size",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--dtype",
        choices=[
            "float16",
            "bfloat16",
            "float32",
        ],
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


# ================================================================
# Metadata
# ================================================================


def load_metadata(
    input_dir: Path,
    tp_size: int,
):
    metadata = []

    for rank in range(tp_size):

        path = (
            input_dir
            / f"tp_metadata-rank-{rank}.json"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing metadata file:\n{path}"
            )

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            metadata.append(
                json.load(f)
            )

    return metadata


# ================================================================
# Checkpoint loading
# ================================================================


def load_rank_checkpoints(
    input_dir: Path,
    tp_size: int,
):
    rank_states = []

    for rank in range(tp_size):

        path = (
            input_dir
            / f"model-rank-{rank}.safetensors"
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint:\n{path}"
            )

        print(
            f"[LOAD] rank {rank}: {path}"
        )

        state_dict = load_file(
            str(path),
            device="cpu",
        )

        print(
            f"       {len(state_dict)} tensors"
        )

        rank_states.append(
            state_dict
        )

    return rank_states


# ================================================================
# Metadata helpers
# ================================================================


def get_parameter_metadata(
    metadata,
    name,
):
    result = []

    for rank_meta in metadata:
        parameters = rank_meta.get(
            "parameters",
            {},
        )

        result.append(
            parameters.get(
                name,
                {},
            )
        )

    return result


def metadata_value_same(
    rank_metadata,
    key,
):
    values = [
        item.get(key)
        for item in rank_metadata
    ]

    return all(
        value == values[0]
        for value in values
    )


# ================================================================
# Replicated parameters
# ================================================================


def is_replicated(
    rank_metadata,
    tensors,
):
    """
    Determine whether a parameter is replicated.

    If every TP rank contains exactly the same tensor,
    we safely keep rank 0.

    This handles:
        LayerNorm
        RMSNorm
        biases
        replicated parameters
        etc.
    """

    if len(tensors) <= 1:
        return True

    reference = tensors[0]

    for tensor in tensors[1:]:

        if tensor.shape != reference.shape:
            return False

        if tensor.dtype != reference.dtype:
            return False

        if not torch.equal(
            tensor,
            reference,
        ):
            return False

    return True


# ================================================================
# Infer TP dimension from metadata
# ================================================================


def get_concat_dim(
    rank_metadata,
    tensors,
    tp_size,
):
    """
    Determine the dimension along which TP shards should
    be concatenated.

    Priority:

        1. explicit partition_dim
        2. output_dim
        3. input_dim
        4. parameter class
    """

    # ------------------------------------------------------------
    # Explicit partition_dim
    # ------------------------------------------------------------

    values = [
        item.get("partition_dim")
        for item in rank_metadata
    ]

    if (
        values
        and values[0] is not None
        and all(v == values[0] for v in values)
    ):
        return int(values[0])

    # ------------------------------------------------------------
    # Column-parallel parameter
    #
    # output_dim = sharded dimension
    # ------------------------------------------------------------

    values = [
        item.get("output_dim")
        for item in rank_metadata
    ]

    if (
        values
        and values[0] is not None
        and all(v == values[0] for v in values)
    ):
        return int(values[0])

    # ------------------------------------------------------------
    # Row-parallel parameter
    #
    # input_dim = sharded dimension
    # ------------------------------------------------------------

    values = [
        item.get("input_dim")
        for item in rank_metadata
    ]

    if (
        values
        and values[0] is not None
        and all(v == values[0] for v in values)
    ):
        return int(values[0])

    # ------------------------------------------------------------
    # Parameter class fallback
    # ------------------------------------------------------------

    param_class = rank_metadata[0].get(
        "parameter_class",
        "",
    )

    # Common vLLM column-parallel parameters.
    if param_class in {
        "_ColumnvLLMParameter",
        "ModelWeightParameter",
    }:
        if tensors[0].ndim >= 1:

            # Most vLLM linear weights are [out, in].
            return 0

    raise RuntimeError(
        "\n"
        "Cannot determine TP merge dimension.\n"
        f"Parameter class: {param_class}\n"
        f"Rank shapes:\n"
        + "\n".join(
            f"  rank{i}: {tuple(t.shape)}"
            for i, t in enumerate(tensors)
        )
        + "\n"
        "Please add explicit partition_dim metadata "
        "to the saving function."
    )


# ================================================================
# Merge one parameter
# ================================================================


def merge_parameter(
    name,
    tensors,
    rank_metadata,
    tp_size,
):
    # ------------------------------------------------------------
    # Sanity
    # ------------------------------------------------------------

    if len(tensors) != tp_size:
        raise RuntimeError(
            f"{name}: expected {tp_size} tensors, "
            f"got {len(tensors)}"
        )

    # ------------------------------------------------------------
    # Replicated parameter
    # ------------------------------------------------------------

    if is_replicated(
        rank_metadata,
        tensors,
    ):
        return tensors[0]

    # ------------------------------------------------------------
    # 0-D tensor
    # ------------------------------------------------------------

    if tensors[0].ndim == 0:
        return tensors[0]

    # ------------------------------------------------------------
    # Determine TP dimension
    # ------------------------------------------------------------

    concat_dim = get_concat_dim(
        rank_metadata,
        tensors,
        tp_size,
    )

    # ------------------------------------------------------------
    # Shape validation
    # ------------------------------------------------------------

    reference_shape = list(
        tensors[0].shape
    )

    for rank, tensor in enumerate(
        tensors
    ):
        shape = list(
            tensor.shape
        )

        for dim in range(
            len(reference_shape)
        ):
            if dim == concat_dim:
                continue

            if (
                shape[dim]
                != reference_shape[dim]
            ):
                raise RuntimeError(
                    f"{name}: incompatible "
                    f"shape on rank {rank}.\n"
                    f"rank0={reference_shape}\n"
                    f"rank{rank}={shape}\n"
                    f"concat_dim={concat_dim}"
                )

    # ------------------------------------------------------------
    # Concatenate
    # ------------------------------------------------------------

    merged = torch.cat(
        tensors,
        dim=concat_dim,
    )

    print(
        f"[MERGE] {name}"
        f"  dim={concat_dim}"
        f"  local={tuple(tensors[0].shape)}"
        f"  full={tuple(merged.shape)}"
    )

    return merged


# ================================================================
# Main
# ================================================================


def main():

    args = parse_args()

    input_dir = Path(
        args.input_dir
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        output_dir
        / "model.safetensors"
    )

    if (
        output_file.exists()
        and not args.overwrite
    ):
        raise FileExistsError(
            f"{output_file} already exists. "
            "Use --overwrite."
        )

    print("=" * 80)
    print("vLLM TP Checkpoint Merger")
    print("=" * 80)

    print(
        f"Input : {input_dir}"
    )

    print(
        f"Output: {output_dir}"
    )

    print(
        f"TP    : {args.tp_size}"
    )

    print()

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    metadata = load_metadata(
        input_dir,
        args.tp_size,
    )

    rank_states = load_rank_checkpoints(
        input_dir,
        args.tp_size,
    )

    # ------------------------------------------------------------
    # Parameter names
    # ------------------------------------------------------------

    reference_names = set(
        rank_states[0].keys()
    )

    for rank in range(
        1,
        args.tp_size,
    ):

        current_names = set(
            rank_states[rank].keys()
        )

        if (
            current_names
            != reference_names
        ):

            missing = (
                reference_names
                - current_names
            )

            extra = (
                current_names
                - reference_names
            )

            raise RuntimeError(
                f"Rank {rank} parameter names differ.\n"
                f"Missing: {sorted(missing)[:20]}\n"
                f"Extra: {sorted(extra)[:20]}"
            )

    # ------------------------------------------------------------
    # Merge
    # ------------------------------------------------------------

    merged_state_dict = {}

    names = sorted(
        reference_names
    )

    print()
    print(
        f"Merging {len(names)} tensors..."
    )
    print()

    for index, name in enumerate(
        names,
        start=1,
    ):

        tensors = [
            rank_states[rank][name]
            for rank in range(
                args.tp_size
            )
        ]

        rank_metadata = (
            get_parameter_metadata(
                metadata,
                name,
            )
        )

        merged = merge_parameter(
            name=name,
            tensors=tensors,
            rank_metadata=rank_metadata,
            tp_size=args.tp_size,
        )

        # --------------------------------------------------------
        # Optional dtype conversion
        # --------------------------------------------------------

        if args.dtype == "float16":
            merged = merged.to(
                torch.float16
            )

        elif args.dtype == "bfloat16":
            merged = merged.to(
                torch.bfloat16
            )

        elif args.dtype == "float32":
            merged = merged.to(
                torch.float32
            )

        merged_state_dict[name] = (
            merged.contiguous()
        )

        if index % 100 == 0:
            print(
                f"[PROGRESS] "
                f"{index}/{len(names)}"
            )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------

    print()
    print("=" * 80)
    print("Saving full checkpoint")
    print("=" * 80)

    save_file(
        merged_state_dict,
        str(output_file),
    )

    print(
        f"[SAVE] {output_file}"
    )

    # ------------------------------------------------------------
    # MLA inspection
    # ------------------------------------------------------------

    mla_params = {
        name: tensor
        for name, tensor
        in merged_state_dict.items()
        if ".mla." in name
    }

    print()
    print("=" * 80)
    print("MLA verification")
    print("=" * 80)

    print(
        f"MLA tensors: {len(mla_params)}"
    )

    for name, tensor in mla_params.items():
        print(
            f"{name:90s}"
            f" shape={str(tuple(tensor.shape)):25s}"
            f" dtype={tensor.dtype}"
        )

    # ------------------------------------------------------------
    # Save merge metadata
    # ------------------------------------------------------------

    merge_info = {
        "tp_size": args.tp_size,
        "source_files": [
            f"model-rank-{rank}.safetensors"
            for rank in range(
                args.tp_size
            )
        ],
        "num_tensors": len(
            merged_state_dict
        ),
        "num_mla_tensors": len(
            mla_params
        ),
    }

    with open(
        output_dir
        / "merge_info.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            merge_info,
            f,
            indent=2,
        )

    print()
    print(
        "Merge completed successfully."
    )


if __name__ == "__main__":
    main()