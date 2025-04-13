import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

import wandb


def train_gat(device, train_graph, val_graph, num_epochs, model, optimizer, loss_fn):
    top_accuracy = 0.0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        z = model(train_graph.x.to(device), train_graph.edge_index.to(device))

        pos_edge_index = train_graph.edge_index
        neg_edge_index = train_graph.fake_edge_index

        pos_pred = model.decode(z, pos_edge_index.to(device))
        neg_pred = model.decode(z, neg_edge_index.to(device))

        pos_target = torch.ones(pos_pred.size(0), device=device)
        neg_target = torch.zeros(neg_pred.size(0), device=device)

        # Combine predictions and targets
        pred = torch.cat([pos_pred, neg_pred])
        targets_numpy = torch.cat([pos_target, neg_target])

        # Calculate loss
        train_loss = loss_fn(pred, targets_numpy)

        # Backward pass
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            z = model(val_graph.x.to(device), val_graph.edge_index.to(device))

            pos_edge_index = val_graph.edge_index
            neg_edge_index = val_graph.fake_edge_index

            pos_pred = model.decode(z, pos_edge_index.to(device))
            neg_pred = model.decode(z, neg_edge_index.to(device))

            logits_tensor = torch.cat([pos_pred, neg_pred])
            targets_tensor = torch.cat(
                [
                    torch.ones(pos_pred.size(0), device=device),
                    torch.zeros(neg_pred.size(0), device=device),
                ],
            )

            logits_numpy = logits_tensor.cpu().numpy()
            labels_numpy = (logits_numpy > 0.0).astype(float)
            targets_numpy = targets_tensor.cpu().numpy()

            val_loss = loss_fn(logits_tensor, targets_tensor)

            # Calculate metrics
            val_auc = roc_auc_score(targets_numpy, logits_numpy)
            val_accuracy = accuracy_score(targets_numpy, labels_numpy)
            top_accuracy = max(top_accuracy, val_accuracy)

            # Log metrics
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "val_loss": val_loss.item(),
                    "val_auc": val_auc,
                    "val_accuracy": val_accuracy,
                }
            )

            if epoch % 10 == 0:
                print(
                    f"Epoch {epoch}: Train loss = {train_loss.item():.4f}, Validation Loss: {val_loss.item():.4f} Val AUC = {val_auc:.4f}, Val Accuracy = {val_accuracy:.4f}"
                )

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss.item(),
                    "validation_loss": val_loss.item(),
                    "val_auc": val_auc,
                }
            )

    return top_accuracy  # For Optuna
