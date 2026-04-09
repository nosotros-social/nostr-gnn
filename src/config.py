from dataclasses import dataclass
from typing import Any

from hydra.core.config_store import ConfigStore
from hydra_zen import MISSING, ZenField, builds, make_config
from torch_geometric.loader import LinkNeighborLoader

from data.dataset import FEATURE_COLUMNS, GraphDataset
from models.graph_sage import NostrSAGE

dataset_config = builds(
    GraphDataset,
    dir="data",
    feature_columns=FEATURE_COLUMNS,
    val_ratio=0.1,
    test_ratio=0.1,
    disjoint_train_ratio=0.3,
    populate_full_signature=True,
)
loader_config = builds(
    LinkNeighborLoader,
    batch_size=8192,
    num_neighbors=[5, 3],
    shuffle=True,
    populate_full_signature=True,
    zen_partial=True,
)
module_config = builds(
    NostrSAGE,
    hidden_channels=128,
    out_channels=64,
    dropout=0.3,
    zen_partial=True,
)


@dataclass
class BaseConfig:
    module: Any = MISSING
    loader: Any = MISSING
    dataset: Any = MISSING

    batch_size: int = 65536
    train_epochs: int = 10
    lr: float = 1e-3
    output_dir: str = "outputs"
    num_workers: int = 4
    model_id: str = "nostr-sage"
    version: str = "v1"
    trackio_project: str = "NostrGNN"
    trackio_space_id: str = "cesardias/nosotros-ml"
    train: bool = True
    test: bool = False


def build_config_store():
    config_store = ConfigStore.instance()

    dataset = dataset_config()
    loader = loader_config()
    module = module_config()

    config_store.store(group="dataset", name="dataset", node=dataset)
    config_store.store(group="loader", name="neighbor", node=loader)
    config_store.store(group="module", name="graphsage", node=module)

    zen_config = []

    for value in BaseConfig.__dataclass_fields__.values():
        item = (
            ZenField(name=value.name, hint=value.type, default=value.default)
            if value.default is not MISSING
            else ZenField(name=value.name, hint=value.type)
        )
        zen_config.append(item)

    config = make_config(
        *zen_config,
        hydra_defaults=[
            "_self_",
            dict(loader="neighbor"),
            dict(dataset="dataset"),
            dict(module="graphsage"),
        ],
    )
    config_store.store(name="config", node=config)

    return config_store
