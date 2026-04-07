import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader


def evaluate(
    module: nn.Module,
    loader: LinkNeighborLoader,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    module.eval()
    total_loss = 0.0
    num_batches = 0
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings = module(batch.x, batch.edge_index)

            src = embeddings[batch.edge_label_index[0]]
            dst = embeddings[batch.edge_label_index[1]]
            scores = (src * dst).sum(dim=1)
            labels = batch.edge_label.float()

            loss = F.binary_cross_entropy_with_logits(scores, labels)
            total_loss += loss.item()
            num_batches += 1

            all_scores.append(scores.cpu())
            all_labels.append(labels.cpu())

    all_scores = torch.cat(all_scores)
    all_labels = torch.cat(all_labels)
    predictions = (torch.sigmoid(all_scores) > 0.5).float()
    accuracy = (predictions == all_labels).float().mean().item()

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy,
    }
