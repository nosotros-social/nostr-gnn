import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ModelManifest(BaseModel):
    model_id: str
    architecture: str
    embedding_dim: int
    hidden_channels: int
    num_layers: int
    feature_columns: list[str]
    scorer: str
    num_nodes: int
    url: str
    sha256: str


def compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def build_manifest(
    config_path: Path,
    model_path: Path,
    model_url: str,
    num_nodes: int,
    embedding_dim: int,
) -> ModelManifest:
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return ModelManifest(
        model_id=cfg["model_id"],
        architecture="graphsage",
        embedding_dim=embedding_dim,
        hidden_channels=cfg["module"]["hidden_channels"],
        num_layers=len(cfg["loader"]["num_neighbors"]),
        feature_columns=cfg["dataset"]["feature_columns"],
        scorer="dot_product",
        num_nodes=num_nodes,
        url=model_url,
        sha256=compute_sha256(model_path),
    )


def export_manifest(manifest: ModelManifest, manifest_path: Path):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    logger.info(f"Wrote manifest to {manifest_path}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Export embeddings to parquet")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=Path("outputs/embeddings.pt"),
        help="Path to embeddings .pt file",
    )
    parser.add_argument(
        "--index-node-id",
        type=Path,
        default=Path("data/processed/index_node_id.npy"),
        help="Path to index_node_id.npy (embedding row index -> node_id)",
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
        default=Path("outputs/model.pt"),
        help="Path to model.pt",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Published URL for model.pt",
    )
    args = parser.parse_args()
    manifest_path = args.manifest or args.output.with_name("manifest.json")
    num_nodes, embedding_dim, _ = export_parquet(
        args.embeddings,
        args.index_node_id,
        args.output,
    )
    manifest = build_manifest(
        config_path=args.config,
        model_path=args.model,
        model_url=args.url,
        num_nodes=num_nodes,
        embedding_dim=embedding_dim,
    )
    export_manifest(manifest, manifest_path)


if __name__ == "__main__":
    main()
