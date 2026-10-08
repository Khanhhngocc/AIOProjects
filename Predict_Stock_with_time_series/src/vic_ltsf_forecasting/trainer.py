import torch
import torch.nn as nn

from vic_ltsf_forecasting.config import MODEL_DIR


def load_model_from_checkpoint(model, checkpoint_path, device):
    """Load a trained model from an existing checkpoint."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    return model, checkpoint


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Train a model for one epoch."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        loss.backward()
        optimizer.step()

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device):
    """Calculate average validation loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        batch_size = batch_x.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def train_model(
    model,
    train_loader,
    valid_loader,
    model_name,
    look_back,
    pred_len,
    device,
    num_epochs,
    learning_rate,
):
    """Train one model and save its best validation checkpoint."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5,
    )

    model = model.to(device)

    history = {
        "train_loss": [],
        "valid_loss": [],
    }

    best_valid_loss = float("inf")
    checkpoint_path = MODEL_DIR / (
        f"vic_{model_name.lower()}_{look_back}d_best.pt"
    )

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        valid_loss = evaluate_loss(
            model=model,
            dataloader=valid_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "look_back": look_back,
                    "pred_len": pred_len,
                    "best_valid_loss": best_valid_loss,
                    "epoch": epoch,
                },
                checkpoint_path,
            )

        print(
            f"{model_name:7s} | Look-back: {look_back:3d} | "
            f"Epoch: {epoch:02d}/{num_epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Valid MSE: {valid_loss:.6f}"
        )

    model, checkpoint = load_model_from_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    return model, history, checkpoint_path