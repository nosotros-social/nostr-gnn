import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ArchitectureMetadata(BaseModel):
    name: str
    hidden_channels: int
    num_layers: int
    aggregator: str
    activation: str
    output_normalization: str
    root_weight: bool
    bias: bool
    edge_types: list[str]
    edge_direction: str
    self_loops: bool


class ArtifactMetadata(BaseModel):
    name: str
    role: str
    format: str
    url: str
    sha256: str


class TensorMetadata(BaseModel):
    name: str
    shape: list[int]
    dtype: str
    offset_bytes: int
    length_bytes: int
    count: int


class ModelManifest(BaseModel):
    model_id: str
    version: str
    architecture: ArchitectureMetadata
    embedding_dim: int
    feature_dim: int
    feature_columns: list[str]
    feature_transform: str
    feature_mean: list[float]
    feature_std: list[float]
    scorer: str
    num_nodes: int
    artifacts: list[ArtifactMetadata]
    param_count: int
    tensors: list[TensorMetadata]


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dtype_size(dtype: str) -> int:
    sizes = {
        "float32": 4,
        "float64": 8,
        "float16": 2,
        "int64": 8,
        "int32": 4,
        "int16": 2,
        "int8": 1,
        "uint8": 1,
        "bool": 1,
    }
    if dtype not in sizes:
        raise ValueError(f"Unsupported dtype for manifest validation: {dtype}")
    return sizes[dtype]


def shape_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def validate_manifest(manifest: ModelManifest) -> None:
    if manifest.feature_dim != len(manifest.feature_columns):
        raise ValueError("feature_dim does not match len(feature_columns)")
    if len(manifest.feature_mean) != manifest.feature_dim:
        raise ValueError("len(feature_mean) does not match feature_dim")
    if len(manifest.feature_std) != manifest.feature_dim:
        raise ValueError("len(feature_std) does not match feature_dim")

    for tensor in manifest.tensors:
        expected_count = shape_count(tensor.shape)
        if expected_count != tensor.count:
            raise ValueError(
                f"Tensor {tensor.name} has count={tensor.count} but shape implies "
                f"{expected_count}"
            )
        expected_length = tensor.count * dtype_size(tensor.dtype)
        if expected_length != tensor.length_bytes:
            raise ValueError(
                f"Tensor {tensor.name} has length_bytes={tensor.length_bytes} but "
                f"count*dtype_size={expected_length}"
            )


def parse_artifact_urls(items: list[str]) -> dict[str, str]:
    artifact_urls: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid --artifact-url value: {item!r}. Expected NAME=URL."
            )
        name, url = item.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name or not url:
            raise ValueError(
                f"Invalid --artifact-url value: {item!r}. Expected NAME=URL."
            )
        artifact_urls[name] = url
    return artifact_urls


def export_parquet(
    embeddings_path: Path,
    index_node_id_path: Path,
    output_path: Path,
) -> tuple[int, int, float]:
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = torch.load(embeddings_path, weights_only=True, map_location="cpu")
    embedding_np = embeddings.numpy()
    num_nodes, embedding_dim = embedding_np.shape
    logger.info(f"Embeddings: {num_nodes:,} nodes x {embedding_dim} dims")
    finite_rows = np.isfinite(embedding_np).all(axis=1)
    if not finite_rows.all():
        bad_rows = int((~finite_rows).sum())
        raise ValueError(f"Embeddings contain {bad_rows:,} rows with NaN or inf values")

    logger.info(f"Loading index_node_id mapping from {index_node_id_path}")
    all_ids = np.load(index_node_id_path)

    if len(all_ids) != num_nodes:
        raise ValueError(
            f"Node count mismatch: {len(all_ids):,} IDs in mapping vs "
            f"{num_nodes:,} embedding rows. The mapping and embeddings "
            f"are from different training runs."
        )

    node_id_array = pa.array(all_ids)
    embedding_bytes_array = pa.array(
        [embedding_np[i].tobytes() for i in range(num_nodes)],
        type=pa.binary(),
    )

    table = pa.table(
        {
            "node_id": node_id_array,
            "embedding": embedding_bytes_array,
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing {output_path}")
    pq.write_table(table, output_path)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(
        f"Exported {num_nodes:,} embeddings "
        f"({embedding_dim} dims, {embedding_dim * 4} bytes each) "
        f"to {output_path} ({file_size_mb:.1f} MB)"
    )
    return num_nodes, embedding_dim, file_size_mb


def load_or_compute_feature_stats(
    feature_stats_path: Path,
    dataset_dir: Path,
    feature_columns: list[str],
) -> tuple[list[float], list[float]]:
    candidate_paths = [feature_stats_path]
    fallback_path = Path("data/processed/feature_stats.npz")
    if fallback_path not in candidate_paths:
        candidate_paths.append(fallback_path)

    for candidate in candidate_paths:
        if candidate.exists():
            logger.info(f"Loading feature stats from {candidate}")
            feature_stats = np.load(candidate)
            return (
                feature_stats["mean"].astype(np.float32).tolist(),
                feature_stats["std"].astype(np.float32).tolist(),
            )

    logger.info(
        f"feature_stats.npz not found at {feature_stats_path}; "
        f"recomputing normalization stats from {dataset_dir}"
    )
    
    features_df = pd.concat(
        pd.read_parquet(f) for f in sorted(dataset_dir.glob("features*.parquet"))
    )
    edges_df = pd.concat(
        pd.read_parquet(f, columns=["src", "dst"])
        for f in sorted(dataset_dir.glob("edges*.parquet"))
    )
    mutes_df = pd.concat(
        pd.read_parquet(f, columns=["src", "dst"])
        for f in sorted(dataset_dir.glob("mutes*.parquet"))
    )

    all_ids = np.union1d(
        features_df["node_id"].unique(),
        np.union1d(
            np.union1d(edges_df["src"].unique(), edges_df["dst"].unique()),
            np.union1d(mutes_df["src"].unique(), mutes_df["dst"].unique()),
        ),
    )
    all_ids.sort()
    id_to_idx = {nid: idx for idx, nid in enumerate(all_ids)}

    feature_matrix = np.zeros((len(all_ids), len(feature_columns)), dtype=np.float32)
    node_indices = features_df["node_id"].map(id_to_idx).values
    feature_matrix[node_indices] = (
        features_df[feature_columns].fillna(0).values.astype(np.float32)
    )

    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std[std == 0] = 1.0

    return mean.astype(np.float32).tolist(), std.astype(np.float32).tolist()


def build_manifest(
    config_path: Path,
    model_path: Path,
    embeddings_path: Path,
    index_node_id_path: Path,
    node_id_pubkey_path: Path,
    feature_stats_path: Path,
    model_bin_path: Path,
    artifact_urls: dict[str, str],
    num_nodes: int,
    embedding_dim: int,
) -> ModelManifest:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    dataset_dir = Path(cfg["dataset"]["dir"])
    feature_columns = list(cfg["dataset"]["feature_columns"])
    feature_mean, feature_std = load_or_compute_feature_stats(
        feature_stats_path=feature_stats_path,
        dataset_dir=dataset_dir,
        feature_columns=feature_columns,
    )
    state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
    tensors = []
    offset_bytes = 0
    for name, tensor in state_dict.items():
        shape = list(tensor.shape)
        dtype = str(tensor.dtype).removeprefix("torch.")
        count = int(tensor.numel())
        length_bytes = count * tensor.element_size()
        tensors.append(
            TensorMetadata(
                name=name,
                shape=shape,
                dtype=dtype,
                offset_bytes=offset_bytes,
                length_bytes=length_bytes,
                count=count,
            )
        )
        offset_bytes += length_bytes
    param_count = sum(int(tensor.numel()) for tensor in state_dict.values())
    return ModelManifest(
        model_id=cfg["model_id"],
        version=cfg["version"],
        architecture=ArchitectureMetadata(
            name="graphsage",
            hidden_channels=cfg["module"]["hidden_channels"],
            num_layers=len(cfg["loader"]["num_neighbors"]),
            aggregator="mean",
            activation="relu",
            output_normalization="none",
            root_weight=True,
            bias=True,
            edge_types=["follow"],
            edge_direction="directed",
            self_loops=False,
        ),
        embedding_dim=embedding_dim,
        feature_dim=len(feature_columns),
        feature_columns=feature_columns,
        feature_transform="zero_fill_then_zscore_v1",
        feature_mean=feature_mean,
        feature_std=feature_std,
        scorer="dot_product",
        num_nodes=num_nodes,
        artifacts=[
            ArtifactMetadata(
                name="model",
                role="training",
                format="pt",
                url=artifact_urls["model"],
                sha256=compute_sha256(model_path),
            ),
            ArtifactMetadata(
                name="embeddings",
                role="dataset",
                format="pt",
                url=artifact_urls["embeddings"],
                sha256=compute_sha256(embeddings_path),
            ),
            ArtifactMetadata(
                name="index_node_id",
                role="dataset",
                format="npy",
                url=artifact_urls["index_node_id"],
                sha256=compute_sha256(index_node_id_path),
            ),
            ArtifactMetadata(
                name="node_id_pubkey",
                role="dataset",
                format="parquet",
                url=artifact_urls["node_id_pubkey"],
                sha256=compute_sha256(node_id_pubkey_path),
            ),
            ArtifactMetadata(
                name="weights",
                role="inference",
                format="bin",
                url=artifact_urls["weights"],
                sha256=compute_sha256(model_bin_path),
            ),
        ],
        param_count=param_count,
        tensors=tensors,
    )


def export_manifest(manifest: ModelManifest, manifest_path: Path):
    validate_manifest(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    logger.info(f"Wrote manifest to {manifest_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export embeddings to parquet")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("outputs/modal/embeddings.pt"),
        help="Path to embeddings .pt file",
    )
    parser.add_argument(
        "--index-node-id",
        type=Path,
        default=Path("outputs/modal/index_node_id.npy"),
        help="Path to index_node_id.npy (embedding row index -> node_id)",
    )
    parser.add_argument(
        "--feature-stats",
        type=Path,
        default=Path("data/processed/feature_stats.npz"),
        help="Path to feature_stats.npz with feature mean/std",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/embeddings.parquet"),
        help="Output parquet path",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output manifest path (defaults to manifest.json next to parquet output)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.modal.yaml"),
        help="Config YAML used for the model run",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("outputs/modal/model.pt"),
        help="Path to model.pt",
    )
    parser.add_argument(
        "--model-bin",
        type=Path,
        default=Path("outputs/modal/model.bin"),
        help="Path to model.bin. Used for weights_sha256 in manifest.json.",
    )
    parser.add_argument(
        "--node-id-pubkey",
        type=Path,
        default=Path("outputs/modal/node_id_pubkey.parquet"),
        help="Path to node_id_pubkey.parquet",
    )
    parser.add_argument(
        "--artifact-url",
        action="append",
        default=[],
        help=(
            "Artifact URL mapping in the form NAME=URL. "
            "Expected names: model, embeddings, index_node_id, "
            "node_id_pubkey, weights."
        ),
    )
    args = parser.parse_args()
    manifest_path = args.manifest or args.output.with_name("manifest.json")

    num_nodes, embedding_dim, _ = export_parquet(
        args.embeddings,
        args.index_node_id,
        args.output,
    )
    artifact_urls = parse_artifact_urls(args.artifact_url)
    required_artifact_names = {
        "model",
        "embeddings",
        "index_node_id",
        "node_id_pubkey",
        "weights",
    }

    if required_artifact_names.issubset(artifact_urls):
        if not args.model_bin.exists():
            raise FileNotFoundError(f"model.bin not found: {args.model_bin}")
        manifest = build_manifest(
            config_path=args.config,
            model_path=args.model,
            embeddings_path=args.embeddings,
            index_node_id_path=args.index_node_id,
            node_id_pubkey_path=args.node_id_pubkey,
            feature_stats_path=args.feature_stats,
            model_bin_path=args.model_bin,
            artifact_urls=artifact_urls,
            num_nodes=num_nodes,
            embedding_dim=embedding_dim,
        )
        export_manifest(manifest, manifest_path)
    else:
        logger.info(
            "Skipped manifest export because required artifact URLs were not all provided. "
            "Provide --artifact-url NAME=URL for model, embeddings, index_node_id, "
            "node_id_pubkey, and weights to generate manifest.json."
        )


if __name__ == "__main__":
    main()
