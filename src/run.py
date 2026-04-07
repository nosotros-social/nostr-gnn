import logging
import os

import colorlog
import hydra
import torch
import trackio
from hydra_zen import instantiate
from omegaconf import OmegaConf

from config import BaseConfig, build_config_store
from eval import evaluate
from train import train

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s %(levelname)s:%(message)s",
        datefmt="%H:%M:%S",
    )
)

logging.basicConfig(level=logging.DEBUG, handlers=[handler])

config_store = build_config_store()

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@hydra.main(config_path=None, config_name="config", version_base=None)
def run(cfg: BaseConfig) -> None:
    device = get_device()
    logger.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg, resolve=True)}")
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    trackio_kwargs = {
        "project": cfg.trackio_project,
        "config": config_dict,
    }
    if cfg.trackio_space_id:
        trackio_kwargs["space_id"] = cfg.trackio_space_id
    trackio.init(**trackio_kwargs)
    trackio.log({"run/started": 1, "run_step": 0})

    dataset = instantiate(cfg.dataset)
    train_data, val_data, test_data = dataset[0]
    num_pos = (train_data.edge_label == 1).sum().item()
    num_mute_neg = (train_data.edge_label == -1).sum().item()

    logger.info(
        f"Loaded graph: "
        f"nodes={train_data.num_nodes:,}  "
        f"full_edges={train_data.full_edge_index.size(1):,}  "
        f"train_mp={train_data.edge_index.size(1):,}  "
        f"train_supervision={train_data.edge_label_index.size(1):,} "
        f"(positives={num_pos:,} mute_negatives={num_mute_neg:,})  "
        f"mutes={train_data.mute_edge_index.size(1):,} "
        f"num_workers={cfg.num_workers}"
    )
    trackio.log(
        {
            "graph/nodes": train_data.num_nodes,
            "graph/full_edges": train_data.full_edge_index.size(1),
            "graph/train_mp_edges": train_data.edge_index.size(1),
            "graph/train_supervision_edges": train_data.edge_label_index.size(1),
            "graph/mutes": train_data.mute_edge_index.size(1),
            "run_step": 0,
        }
    )

    module_factory = instantiate(cfg.module)
    module = module_factory(in_channels=train_data.num_node_features)

    loader_factory = instantiate(cfg.loader)
    train_loader = loader_factory(
        data=train_data,
        edge_label_index=train_data.edge_label_index,
        edge_label=train_data.edge_label,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        neg_sampling_ratio=1.0,
    )

    if cfg.train:
        train(
            module=module,
            loader=train_loader,
            train_epochs=cfg.train_epochs,
            lr=cfg.lr,
            output_dir=cfg.output_dir,
            device=device,
        )
        val_loader = loader_factory(
            data=val_data,
            edge_label_index=val_data.edge_label_index,
            edge_label=val_data.edge_label,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )
        val_metrics = evaluate(module, val_loader, device=device)
        logger.info(
            f"Validation  loss={val_metrics['loss']:.4f}  "
            f"accuracy={val_metrics['accuracy']:.4f}"
        )
    if cfg.test:
        test_loader = loader_factory(
            data=test_data,
            edge_label_index=test_data.edge_label_index,
            edge_label=test_data.edge_label,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            shuffle=False,
        )
        test_metrics = evaluate(module, test_loader, device=device)
        logger.info(
            f"Test  loss={test_metrics['loss']:.4f}  "
            f"accuracy={test_metrics['accuracy']:.4f}"
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    module.eval()
    with torch.no_grad():
        eval_embeddings = module(
            train_data.x.to(device),
            train_data.edge_index.to(device),
        )

    eval_embeddings_cpu = eval_embeddings.cpu()
    torch.save(
        eval_embeddings_cpu,
        os.path.join(cfg.output_dir, "embeddings_eval.pt"),
    )
    logger.info(
        f"Saved eval embeddings {eval_embeddings_cpu.shape} "
        f"(mp graph: {train_data.edge_index.size(1):,} edges)"
    )

    del eval_embeddings
    del eval_embeddings_cpu
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if device.type == "cuda":
        prod_module = module.float().cpu()
        prod_x = train_data.x
        prod_edge_index = train_data.full_edge_index
    else:
        prod_module = module
        prod_x = train_data.x.to(device)
        prod_edge_index = train_data.full_edge_index.to(device)

    with torch.no_grad():
        prod_embeddings = prod_module(
            prod_x,
            prod_edge_index,
        )
    prod_embeddings_cpu = prod_embeddings.cpu()
    if not torch.isfinite(prod_embeddings_cpu).all():
        raise ValueError("Production embeddings contain NaN or inf values")
    torch.save(
        prod_embeddings_cpu,
        os.path.join(cfg.output_dir, "embeddings.pt"),
    )
    logger.info(
        f"Saved production embeddings {prod_embeddings_cpu.shape} "
        f"(full graph: {train_data.full_edge_index.size(1):,} edges)"
    )

    trackio.finish()


if __name__ == "__main__":
    run()
