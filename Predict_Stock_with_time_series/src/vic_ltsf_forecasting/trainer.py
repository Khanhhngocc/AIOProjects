import torch
import torch.nn as nn


def train_one_epoch(model, dataloader, criterion, optimizer, device):
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

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate_loss(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)

        total_loss += loss.item() * batch_x.size(0)
        total_samples += batch_x.size(0)

    return total_loss / total_samples


def train_model(
    model,
    train_loader,
    valid_loader,
    checkpoint_path,
    device,
    epochs,
    learning_rate
):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.5
    )

    model = model.to(device)

    history = {
        "train_loss": [],
        "valid_loss": []
    }

    best_valid_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device
        )

        valid_loss = evaluate_loss(
            model,
            valid_loader,
            criterion,
            device
        )

        scheduler.step()

        history["train_loss"].append(train_loss)
        history["valid_loss"].append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_epoch = epoch

            torch.save({
                "model_state_dict": model.state_dict(),
                "best_valid_loss": best_valid_loss,
                "epoch": best_epoch
            }, checkpoint_path)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train MSE: {train_loss:.6f} | "
            f"Valid MSE: {valid_loss:.6f}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, history, best_valid_loss, best_epoch