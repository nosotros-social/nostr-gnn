import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RandomLinkSplit

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 6

# NIP-85 (but not limited to) features
FEATURE_COLUMNS = (
    "rank",
    "follower_cnt",
    "post_cnt",
    "reply_cnt",
    "reactions_cnt",
    "reports_cnt_sent",
    "reports_cnt_recd",
    "zap_amt_sent",
    "zap_amt_recd",
    "zap_cnt_sent",
    "zap_cnt_recd",
    "zap_avg_amt_day_sent",
    "zap_avg_amt_day_recd",
    "first_created_at",
    "active_hours_start",
    "active_hours_end",
)


class GraphDataset(Dataset):
    def __init__(
        self,
        dir: str,
        feature_columns: tuple[str, ...] = FEATURE_COLUMNS,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        disjoint_train_ratio: float = 0.3,
    ):
        self.data_dir = Path(dir)
        self.feature_columns = tuple(feature_columns)
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.disjoint_train_ratio = disjoint_train_ratio
        super().__init__(root=dir)

    def len(self):
        return 1

    def get(self, idx):
        return torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [f"graph-{self._cache_key()}.pt"]

    def _cache_key(self) -> str:
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "feature_columns": self.feature_columns,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "disjoint_train_ratio": self.disjoint_train_ratio,
            "features": self._file_manifest("features*.parquet"),
            "edges": self._file_manifest("edges*.parquet"),
            "mutes": self._file_manifest("mutes*.parquet"),
            "nodes": self._file_manifest("nodes*.parquet"),
        }
        payload = json.dumps(manifest, sort_keys=True).encode()
        return hashlib.sha1(payload).hexdigest()[:12]

    def _file_manifest(self, pattern: str) -> list[dict[str, int | str]]:
        manifest = []
        for path in sorted(self.data_dir.glob(pattern)):
            stat = path.stat()
            manifest.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return manifest

    def process(self):
        nodes_df = pandas.concat(
            pandas.read_parquet(f)
            for f in sorted(self.data_dir.glob("features*.parquet"))
        )
        node_id_pubkey_df = pandas.concat(
            pandas.read_parquet(f) for f in sorted(self.data_dir.glob("nodes*.parquet"))
        )
        edges_df = pandas.concat(
            pandas.read_parquet(f) for f in sorted(self.data_dir.glob("edges*.parquet"))
        )
        mutes_df = pandas.concat(
            pandas.read_parquet(f) for f in sorted(self.data_dir.glob("mutes*.parquet"))
        )

        all_ids = np.union1d(
            nodes_df["node_id"].unique(),
            np.union1d(
                np.union1d(edges_df["src"].unique(), edges_df["dst"].unique()),
                np.union1d(mutes_df["src"].unique(), mutes_df["dst"].unique()),
            ),
        )

        all_ids.sort()
        id_to_idx = {nid: idx for idx, nid in enumerate(all_ids)}
        num_nodes = len(all_ids)

        feature_matrix = np.zeros(
            (num_nodes, len(self.feature_columns)), dtype=np.float32
        )
        node_indices = nodes_df["node_id"].map(id_to_idx).values
        feature_matrix[node_indices] = (
            nodes_df[list(self.feature_columns)].fillna(0).values.astype(np.float32)
        )

        # normalize features
        mean = feature_matrix.mean(axis=0)
        std = feature_matrix.std(axis=0)
        std[std == 0] = 1.0
        feature_matrix = (feature_matrix - mean) / std

        x = torch.from_numpy(feature_matrix)

        valid_edges = edges_df[
            edges_df["src"].isin(id_to_idx) & edges_df["dst"].isin(id_to_idx)
        ]

        src = torch.tensor(valid_edges["src"].map(id_to_idx).values, dtype=torch.long)
        dst = torch.tensor(valid_edges["dst"].map(id_to_idx).values, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)

        valid_mutes = mutes_df[
            mutes_df["src"].isin(id_to_idx) & mutes_df["dst"].isin(id_to_idx)
        ]
        mute_src = torch.tensor(
            valid_mutes["src"].map(id_to_idx).values, dtype=torch.long
        )
        mute_dst = torch.tensor(
            valid_mutes["dst"].map(id_to_idx).values, dtype=torch.long
        )
        mute_edge_index = torch.stack([mute_src, mute_dst], dim=0)

        data = Data(
            x=x,
            edge_index=edge_index,
            mute_edge_index=mute_edge_index,
            num_nodes=num_nodes,
        )
        split = RandomLinkSplit(
            num_val=self.val_ratio,
            num_test=self.test_ratio,
            disjoint_train_ratio=self.disjoint_train_ratio,
            add_negative_train_samples=False,
        )
        train_data, val_data, test_data = split(data)

        num_mutes = mute_edge_index.size(1)
        mute_labels = torch.full(
            (num_mutes,),
            -1,
            dtype=train_data.edge_label.dtype,
        )

        train_data.edge_label_index = torch.cat(
            [train_data.edge_label_index, mute_edge_index],
            dim=1,
        )
        train_data.edge_label = torch.cat(
            [train_data.edge_label, mute_labels],
            dim=0,
        )

        train_data.mute_edge_index = mute_edge_index
        val_data.mute_edge_index = mute_edge_index
        test_data.mute_edge_index = mute_edge_index

        train_data.full_edge_index = edge_index
        val_data.full_edge_index = edge_index
        test_data.full_edge_index = edge_index

        logger.info(
            f"Split edges (disjoint_train_ratio={self.disjoint_train_ratio}): "
            f"full={edge_index.size(1):,}  "
            f"train_mp={train_data.edge_index.size(1):,}  "
            f"train_supervision={train_data.edge_label_index.size(1):,} "
            f"(follow={train_data.edge_label_index.size(1) - num_mutes:,} + "
            f"mute_negatives={num_mutes:,})  "
            f"val_supervision={val_data.edge_label_index.size(1):,}  "
            f"test_supervision={test_data.edge_label_index.size(1):,}"
        )

        # Persist embedding-row-index -> node_id mapping for export.
        index_node_id_path = Path(self.processed_dir) / "index_node_id.npy"
        np.save(index_node_id_path, all_ids)
        logger.info(
            f"Saved index_node_id mapping ({len(all_ids):,} nodes) to "
            f"{index_node_id_path}"
        )

        node_id_pubkey_path = Path(self.processed_dir) / "node_id_pubkey.parquet"
        node_id_pubkey_df[["node_id", "node_pubkey"]].to_parquet(
            node_id_pubkey_path,
            index=False,
        )
        logger.info(
            f"Saved node_id_pubkey mapping ({len(node_id_pubkey_df):,} nodes) to "
            f"{node_id_pubkey_path}"
        )
        torch.save((train_data, val_data, test_data), self.processed_paths[0])


if __name__ == "__main__":
    import logging

    dataset = GraphDataset("data")
    train_data, val_data, test_data = dataset[0]

    logger.info(f"Nodes:    {train_data.num_nodes:,}")
    logger.info(f"Features: {train_data.num_node_features}")
    logger.info(f"MP edges: {train_data.edge_index.size(1):,}")
    logger.info(f"Train supervision: {train_data.edge_label_index.size(1):,}")
    logger.info(f"  Positives: {(train_data.edge_label == 1).sum().item():,}")
    logger.info(f"  Mute negatives: {(train_data.edge_label == -1).sum().item():,}")
    logger.info(f"Val supervision:   {val_data.edge_label_index.size(1):,}")
    logger.info(f"Test supervision:  {test_data.edge_label_index.size(1):,}")
    logger.info(f"Mutes:    {train_data.mute_edge_index.size(1):,}")
