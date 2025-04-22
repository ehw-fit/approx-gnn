from tqdm import tqdm
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from typing import Any
import json
from datetime import datetime
from copy import deepcopy
from torch.utils.data import Dataset
from collections import defaultdict
import random


class ClassPairDataset(Dataset):
    def __init__(self, data, wirings):
        self.data = data
        self.labels = wirings

        self.indices = defaultdict(list)
        for idx, label in enumerate(self.labels):
            self.indices[label].append(idx)

        self.indices = {k: v for k, v in self.indices.items() if len(v) > 2}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        chosen = random.sample(self.indices[self.labels[index]], 2)
        ret = (self.data[chosen[0]], self.data[chosen[1]])
        return ret


class PairDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return random.sample(self.data, 2)


def create_adder_embeddings(
    model: torch.nn.Module, loader: DataLoader, n_features: int
):
    adders = next(iter(loader))
    embeddings = torch.zeros((len(adders.y) + 1, n_features))
    embeddings[:-1] = model(adders)
    return embeddings


PATIENCE = 40


def train_absolute(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    criterion: torch.nn.Module,
    regress_optimizer: torch.optim.Optimizer,
    embed_optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    adder_loader: DataLoader,
    validation_loader: DataLoader,
    n_features: int,
    n_epochs: int,
    train_regress: bool = True,
    train_embed: bool = True,
    regress_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    embed_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_embed: bool = True,
):
    progress_bar = tqdm(range(n_epochs), total=n_epochs, ncols=100, desc="Training")
    losses = []
    best_loss = 1e6
    best_epoch = 0
    best_regress = regress_model.state_dict()
    best_embed = embed_model.state_dict()

    start_time = datetime.now()

    try:
        for epoch in progress_bar:
            for d in loader:
                if use_embed:
                    embeddings = create_adder_embeddings(
                        embed_model, adder_loader, n_features
                    )
                    d.x = embeddings[d.adder_id]

                outs: torch.Tensor = regress_model(d)
                loss = criterion(outs, d.y.unsqueeze(1))
                regress_optimizer.zero_grad()
                embed_optimizer.zero_grad()

                loss.backward()

                if train_regress:
                    regress_optimizer.step()
                if train_embed:
                    embed_optimizer.step()

            if train_regress and regress_scheduler is not None:
                regress_scheduler.step()
            if train_embed and embed_scheduler is not None:
                embed_scheduler.step()

            with torch.no_grad():
                val_loss = 0.0
                cnt = 0

                for d in validation_loader:
                    if use_embed:
                        embeddings = create_adder_embeddings(
                            embed_model, adder_loader, n_features
                        )
                        d.x = embeddings[d.adder_id]

                    outs = regress_model(d)
                    val_loss += criterion(outs, d.y.unsqueeze(1)).item()
                    cnt += 1

                val_loss /= cnt

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    if train_regress:
                        best_regress = deepcopy(regress_model.state_dict())
                    if train_embed:
                        best_embed = deepcopy(embed_model.state_dict())

                progress_bar.set_postfix(best_loss=best_loss, val_loss=val_loss)
                losses.append(val_loss)

            if epoch > best_epoch + PATIENCE:
                break
    except KeyboardInterrupt:
        print("\nInterrupting training.")

    end_time = datetime.now()

    return {
        "best_regress": best_regress,
        "best_embed": best_embed,
        "last_regress": regress_model.state_dict(),
        "last_embed": embed_model.state_dict(),
        "best_loss": best_loss,
        "losses": losses,
        "duration": (end_time - start_time).total_seconds(),
    }


def train_relative(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    criterion: torch.nn.Module,
    regress_optimizer: torch.optim.Optimizer,
    embed_optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    adder_loader: DataLoader,
    validation_loader: DataLoader,
    n_features: int,
    n_epochs: int,
    train_regress: bool = True,
    train_embed: bool = True,
    regress_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    embed_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_embed: bool = True,
):
    progress_bar = tqdm(range(n_epochs), total=n_epochs, ncols=100, desc="Training")
    losses = []
    best_loss = 1e6
    best_epoch = 0
    best_regress = regress_model.state_dict()
    best_embed = embed_model.state_dict()

    start_time = datetime.now()

    try:
        for epoch in progress_bar:
            for d_a, d_b in loader:
                if use_embed:
                    embeddings = create_adder_embeddings(
                        embed_model, adder_loader, n_features
                    )
                    d_a.x = embeddings[d_a.adder_id]
                    d_b.x = embeddings[d_b.adder_id]

                outs: torch.Tensor = regress_model(d_a, d_b)
                loss = criterion(outs, (d_a.y - d_b.y).unsqueeze(1))
                regress_optimizer.zero_grad()
                embed_optimizer.zero_grad()

                loss.backward()

                if train_regress:
                    regress_optimizer.step()
                if train_embed:
                    embed_optimizer.step()

            if train_regress and regress_scheduler is not None:
                regress_scheduler.step()
            if train_embed and embed_scheduler is not None:
                embed_scheduler.step()

            with torch.no_grad():
                val_loss = 0.0
                cnt = 0

                for d_a, d_b in validation_loader:
                    if use_embed:
                        embeddings = create_adder_embeddings(
                            embed_model, adder_loader, n_features
                        )
                        d_a.x = embeddings[d_a.adder_id]
                        d_b.x = embeddings[d_b.adder_id]

                    outs = regress_model(d_a, d_b)
                    val_loss += criterion(outs, (d_a.y - d_b.y).unsqueeze(1)).item()
                    cnt += 1

                val_loss /= cnt

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    if train_regress:
                        best_regress = deepcopy(regress_model.state_dict())
                    if train_embed:
                        best_embed = deepcopy(embed_model.state_dict())

                progress_bar.set_postfix(best_loss=best_loss, val_loss=val_loss)
                losses.append(val_loss)

            if epoch > best_epoch + PATIENCE:
                break
    except KeyboardInterrupt:
        print("\nInterrupting training.")

    end_time = datetime.now()

    return {
        "best_regress": best_regress,
        "best_embed": best_embed,
        "last_regress": regress_model.state_dict(),
        "last_embed": embed_model.state_dict(),
        "best_loss": best_loss,
        "losses": losses,
        "duration": (end_time - start_time).total_seconds(),
    }


def train_classifier(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    criterion: torch.nn.Module,
    regress_optimizer: torch.optim.Optimizer,
    embed_optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    adder_loader: DataLoader,
    validation_loader: DataLoader,
    n_features: int,
    n_epochs: int,
    train_regress: bool = True,
    train_embed: bool = True,
    regress_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    embed_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    use_embed: bool = True,
):
    progress_bar = tqdm(range(n_epochs), total=n_epochs, ncols=100, desc="Training")
    losses = []
    best_loss = 1e6
    best_epoch = 0
    best_regress = regress_model.state_dict()
    best_embed = embed_model.state_dict()

    start_time = datetime.now()

    try:
        for epoch in progress_bar:
            for d_a, d_b in loader:
                if use_embed:
                    embeddings = create_adder_embeddings(
                        embed_model, adder_loader, n_features
                    )
                    d_a.x = embeddings[d_a.adder_id]
                    d_b.x = embeddings[d_b.adder_id]

                outs: torch.Tensor = regress_model(d_a, d_b)
                loss = criterion(outs, ((d_a.y - d_b.y) > 0).unsqueeze(1).float())

                regress_optimizer.zero_grad()
                embed_optimizer.zero_grad()

                loss.backward()

                if train_regress:
                    regress_optimizer.step()
                if train_embed:
                    embed_optimizer.step()

            if train_regress and regress_scheduler is not None:
                regress_scheduler.step()
            if train_embed and embed_scheduler is not None:
                embed_scheduler.step()

            with torch.no_grad():
                val_loss = 0.0
                cnt = 0

                for d_a, d_b in validation_loader:
                    if use_embed:
                        embeddings = create_adder_embeddings(
                            embed_model, adder_loader, n_features
                        )
                        d_a.x = embeddings[d_a.adder_id]
                        d_b.x = embeddings[d_b.adder_id]

                    outs = regress_model(d_a, d_b)
                    val_loss += criterion(
                        outs, ((d_a.y - d_b.y) > 0).unsqueeze(1).float()
                    ).item()
                    cnt += 1

                val_loss /= cnt

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_epoch = epoch
                    if train_regress:
                        best_regress = deepcopy(regress_model.state_dict())
                    if train_embed:
                        best_embed = deepcopy(embed_model.state_dict())

                progress_bar.set_postfix(best_loss=best_loss, val_loss=val_loss)
                losses.append(val_loss)

            if epoch > best_epoch + PATIENCE:
                break
    except KeyboardInterrupt:
        print("\nInterrupting training.")

    end_time = datetime.now()

    return {
        "best_regress": best_regress,
        "best_embed": best_embed,
        "last_regress": regress_model.state_dict(),
        "last_embed": embed_model.state_dict(),
        "best_loss": best_loss,
        "losses": losses,
        "duration": (end_time - start_time).total_seconds(),
    }


def save_run(
    config: dict[str, str | float | int],
    results: dict[str, Any],
    output_path: str | Path,
):
    print(f"Saving results to: {output_path}")

    output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir()

    if not output_path.is_dir():
        raise ValueError("Output path has to be a directory.")

    with open(output_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    torch.save(results["best_regress"], output_path / "best_regress.pt")
    torch.save(results["best_embed"], output_path / "best_embed.pt")
    torch.save(results["last_regress"], output_path / "last_regress.pt")
    torch.save(results["last_embed"], output_path / "last_embed.pt")

    with open(output_path / "results.json", "w") as f:
        json.dump(
            {
                "best_loss": results["best_loss"],
                "duration": results["duration"],
                "losses": results["losses"],
            },
            f,
            indent=2,
        )
