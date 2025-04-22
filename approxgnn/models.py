import torch
import torch_geometric.nn as tgnn
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import json


class AdderEmbedding(torch.nn.Module):
    def __init__(
        self,
        embed_size: int = 8,
        inner_channels: int = 16,
        hidden_layers: int = 2,
    ):
        super().__init__()
        self.conv1 = tgnn.EdgeCNN(
            2, inner_channels, hidden_layers, inner_channels, act=nn.SiLU()
        )
        self.pool1 = tgnn.AttentionalAggregation(
            nn.Linear(inner_channels, inner_channels)
        )
        self.conv2 = tgnn.EdgeCNN(
            2 + inner_channels,
            inner_channels,
            hidden_layers,
            inner_channels,
            act=nn.SiLU(),
        )
        self.final_pool = tgnn.AttentionalAggregation(
            nn.Linear(inner_channels, inner_channels)
        )

        self.out = nn.Linear(inner_channels, embed_size)

    def forward(self, data):
        t, e, batch = data.node_type, data.edge_index, data.batch
        v = data.node_value
        x = torch.concat((t.unsqueeze(1), v.unsqueeze(1)), dim=1)
        edata = data.edge_id

        x_init = x

        x = self.conv1(x, e, edge_attr=edata)
        pooled = self.pool1(x, batch)[batch]

        x = torch.concat((x_init, pooled), dim=1)

        x = self.conv2(x, e, edge_attr=edata)
        x = self.final_pool(x, batch)

        x = self.out(x)
        return x


class GNNRegressor(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels: int = 32,
        hidden_layers: int = 2,
        head_channels: int = 256,
        out_channels: int = 1,
        critical_path: bool = False,
    ):
        super().__init__()
        self.conv1 = tgnn.EdgeCNN(
            in_channels + 2 + (1 if critical_path else 0),
            inner_channels,
            hidden_layers,
            inner_channels,
            act=nn.SiLU(),
        )

        self.pool1 = tgnn.AttentionalAggregation(nn.Linear(inner_channels, 1))

        self.conv2 = tgnn.EdgeCNN(
            in_channels + 2 + (1 if critical_path else 0) + inner_channels,
            inner_channels,
            hidden_layers,
            inner_channels,
            act=nn.SiLU(),
        )

        self.final_pool = tgnn.AttentionalAggregation(nn.Linear(inner_channels, 1))

        self.out = torch.nn.Sequential(
            nn.Linear(inner_channels, head_channels),
            nn.SiLU(),
            nn.Linear(head_channels, head_channels),
            nn.SiLU(),
            nn.Linear(head_channels, out_channels),
        )

        self.critical_path = critical_path

    def forward(self, data):
        x, t, e, batch = data.x, data.node_type, data.edge_index, data.batch
        v = data.node_value
        if self.critical_path:
            x = torch.concat(
                (t.unsqueeze(1), v.unsqueeze(1), data.node_critical.unsqueeze(1), x),
                dim=1,
            )
        else:
            x = torch.concat((t.unsqueeze(1), v.unsqueeze(1), x), dim=1)
        edata = data.edge_id
        x_init = x

        x = self.conv1(x, e, edge_attr=edata)
        pooled = self.pool1(x, batch)[batch]

        x = torch.concat((x_init, pooled), dim=1)

        x = self.conv2(x, e, edge_attr=edata)
        x = self.final_pool(x, batch)

        x = self.out(x)
        return x


class GNNRelative(nn.Module):
    def __init__(
        self,
        in_channels,
        inner_channels: int = 32,
        hidden_layers: int = 2,
        head_channels: int = 64,
        out_channels: int = 1,
        critical_path: bool = False,
    ):
        super().__init__()
        self.conv1 = tgnn.EdgeCNN(
            in_channels + 2 + (1 if critical_path else 0),
            inner_channels,
            hidden_layers,
            inner_channels,
            act=nn.SiLU(),
        )
        self.pool1 = tgnn.AttentionalAggregation(nn.Linear(inner_channels, 1))

        self.conv2 = tgnn.EdgeCNN(
            in_channels + 2 + (1 if critical_path else 0) + inner_channels,
            inner_channels,
            hidden_layers,
            inner_channels,
            act=nn.SiLU(),
        )

        self.final_pool = tgnn.AttentionalAggregation(nn.Linear(inner_channels, 1))

        self.out = torch.nn.Sequential(
            nn.Linear(2 * inner_channels, head_channels),
            nn.SiLU(),
            nn.Linear(head_channels, out_channels),
        )

        self.critical_path = critical_path

    def embed_graph(self, data):
        x, t, e, batch = data.x, data.node_type, data.edge_index, data.batch
        v = data.node_value
        if self.critical_path:
            x = torch.concat(
                (t.unsqueeze(1), v.unsqueeze(1), data.node_critical.unsqueeze(1), x),
                dim=1,
            )
        else:
            x = torch.concat((t.unsqueeze(1), v.unsqueeze(1), x), dim=1)
        edata = data.edge_id
        x_init = x

        x = self.conv1(x, e, edge_attr=edata)
        pooled = self.pool1(x, batch)[batch]

        x = torch.concat((x_init, pooled), dim=1)

        x = self.conv2(x, e, edge_attr=edata)
        x = self.final_pool(x, batch)

        return x

    def forward(self, A, B):
        xA = self.embed_graph(A)
        xB = self.embed_graph(B)

        x = torch.concat((xA, xB), dim=1)

        x = self.out(x)
        return x


def load_models(
    path: Path, load_weights: bool = False
) -> tuple[AdderEmbedding, GNNRegressor | GNNRelative, list[str], list[str]]:
    """Loads a model pair in `path`.
    
    :return: Embedding model, regression model, list of output targets, and list of input metrics.
    :rtype: `tuple[AdderEmbedding, GNNRegressor | GNNRelative, list[str], list[str]]`"""
    if not (path / "config.json").exists():
        raise RuntimeError(f"config.json file not found in {path}")

    with open(path / "config.json") as f:
        config: dict = json.load(f)

    if config["embed"]:
        embed: torch.nn.Module | None = AdderEmbedding(
            config["features"],
            config["embed_inner_channels"],
            config["embed_hidden_layers"],
        )

        if load_weights:
            if (path / "best_embed.pt").exists():
                embed_path = path / "best_embed.pt"
            elif (path / "embed.pt").exists():
                embed_path = path / "embed.pt"
            else:
                raise RuntimeError(
                    "Embedding model not found in directory (best_embed.pt|embed.pt)"
                )

            embed.load_state_dict(torch.load(embed_path, weights_only=True))
    else:
        embed = None

    if config["task"] == "absolute":
        regress: torch.nn.Module = GNNRegressor(
            config["features"],
            config["inner_channels"],
            config["hidden_layers"],
            config["head_channels"],
            config["out_channels"] if "out_channels" in config else 1,
            config["critical_path"] if "critical_path" in config else False,
        )
    elif config["task"] in ["relative", "classify"]:
        regress = GNNRelative(
            config["features"],
            config["inner_channels"],
            config["hidden_layers"],
            config["head_channels"],
            config["out_channels"] if "out_channels" in config else 1,
            config["critical_path"] if "critical_path" in config else False,
        )
    else:
        raise RuntimeError(f"Invalid task in config: {config['task']}")

    if load_weights:
        if (path / "best_regress.pt").exists():
            regress_path = path / "best_regress.pt"
        elif (path / "regress.pt").exists():
            regress_path = path / "regress.pt"
        else:
            raise RuntimeError(
                "Regression model not found in directory (best_regress.pt|regress.pt)"
            )

        regress.load_state_dict(torch.load(regress_path, weights_only=True))

    targets: list[str] = config["targets"] if "targets" in config else ["psnr"]
    metrics: list[str] = (
        config["metrics"]
        if "metrics" in config
        else ["ep%", "mae%", "mre%", "wce%", "wcre%"]
    )

    return (
        embed,
        regress,
        targets,
        metrics,
    )
