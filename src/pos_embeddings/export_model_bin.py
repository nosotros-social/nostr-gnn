import argparse
import hashlib
import json
import logging
import struct
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_model_bin(model_path: Path, output_path: Path) -> tuple[str, int]:
    logger.info(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")

    header: dict[str, dict[str, int | list[int] | str]] = {}
    raw_bytes = bytearray()
    offset = 0
    param_count = 0

    for name, tensor in state_dict.items():
        cpu_tensor = tensor.detach().cpu().float().contiguous()
        tensor_bytes = cpu_tensor.numpy().tobytes()
        length = len(tensor_bytes)

        header[name] = {
            "shape": list(cpu_tensor.shape),
            "dtype": str(cpu_tensor.dtype).removeprefix("torch."),
            "offset": offset,
            "length": length,
        }
        raw_bytes.extend(tensor_bytes)
        offset += length
        param_count += int(cpu_tensor.numel())

    header_json = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    header_len = struct.pack("<I", len(header_json))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        f.write(header_len)
        f.write(header_json)
        f.write(raw_bytes)

    sha256 = compute_sha256(output_path)
    logger.info(
        f"Exported model.bin with {len(header)} tensors "
        f"({param_count:,} parameters) to {output_path}"
    )
    logger.info(f"model.bin sha256={sha256}")
    return sha256, param_count


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export model.pt to model.bin")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/modal/model.pt"),
        help="Path to model.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/modal/model.bin"),
        help="Output path for model.bin",
    )
    args = parser.parse_args()
    export_model_bin(args.model, args.output)


if __name__ == "__main__":
    main()
