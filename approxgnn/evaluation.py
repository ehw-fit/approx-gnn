from pathlib import Path
import json
from datetime import datetime

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from scipy.optimize import curve_fit
from scipy.stats import pearsonr

from approxgnn.training import create_adder_embeddings
from approxgnn.utils import prepare_pyplot, save_figure


def _prepare_output(output_path: Path):
    if not output_path.exists():
        output_path.mkdir()

    if not output_path.is_dir():
        raise RuntimeError("Output root path isn't a directory.")


def evaluate_absolute(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    loader: DataLoader,
    adder_loader: DataLoader,
    n_features: int,
    output_path: Path,
    name: str,
    use_embed: bool,
    offset: float = 0.0,
    color: str = "tab:blue",
):
    _prepare_output(output_path)

    with torch.no_grad():
        d = next(iter(loader))
        real: torch.Tensor = d.y.unsqueeze(-1)

        if use_embed:
            embeddings = create_adder_embeddings(embed_model, adder_loader, n_features)
            d.x = embeddings[d.adder_id]

        start_time = datetime.now()
        predicted: torch.Tensor = regress_model(d) + offset
        end_time = datetime.now()

        mse = F.mse_loss(predicted, real).item()
        mae = F.l1_loss(predicted, real).item()
        r2 = (
            1 - ((real - predicted) ** 2).sum() / ((real - real.mean()) ** 2).sum()
        ).item()
        pcc = float(pearsonr(real, predicted).statistic[0])
        print(f"Evaluation results -> MAE: {mae}, MSE: {mse}, R2: {r2}, PCC: {pcc}")

        fig, ax = prepare_pyplot(square=True)
        ax.scatter(
            real, predicted, s=3.0, alpha=0.25, rasterized=True, zorder=1, c=color
        )

        guide_min = min(real.min().item(), predicted.min().item())
        guide_max = max(real.max().item(), predicted.max().item())
        ax.plot([guide_min, guide_max], [guide_min, guide_max], linestyle="--", alpha=0.75, color="black", zorder=2)
        ax.set_xlabel("True PSNR [dB]")
        ax.set_ylabel("Predicted PSNR [dB]")
        # ax.set_xticks([10, 20, 30, 40, 50, 60])
        # ax.set_yticks([10, 20, 30, 40, 50, 60])

        fig_path = output_path / f"eval_abs_{name}_viz.pdf"
        save_figure(fig, fig_path)

    json_path = output_path / f"eval_abs_{name}_results.json"
    print(f"Saving statistics to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(
            {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "pcc": pcc,
                "duration": (end_time - start_time).total_seconds(),
                "name": name,
                "data": [[float(x), float(y)] for x, y in zip(real, predicted)],
            },
            f,
            indent=2,
        )


def evaluate_relative(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    loader: DataLoader,
    adder_loader: DataLoader,
    n_features: int,
    output_path: Path,
    name: str,
    use_embed: bool,
    offset: float = 0.0,
    color: str = "tab:blue",
):
    _prepare_output(output_path)

    with torch.no_grad():

        d_a, d_b = next(iter(loader))
        real: torch.Tensor = (d_a.y - d_b.y).unsqueeze(-1)

        if use_embed:
            embeddings = create_adder_embeddings(embed_model, adder_loader, n_features)
            d_a.x = embeddings[d_a.adder_id]
            d_b.x = embeddings[d_b.adder_id]

        start_time = datetime.now()
        predicted: torch.Tensor = regress_model(d_a, d_b) + offset
        end_time = datetime.now()

        mse = F.mse_loss(predicted, real).item()
        mae = F.l1_loss(predicted, real).item()
        r2 = (
            1 - ((real - predicted) ** 2).sum() / ((real - real.mean()) ** 2).sum()
        ).item()
        pcc = float(pearsonr(real, predicted).statistic[0])
        print(f"Evaluation results -> MAE: {mae}, MSE: {mse}, R2: {r2}, PCC: {pcc}")

        fig, ax = prepare_pyplot(square=True)
        # if not small_fig:
        #     fig.suptitle(f"True vs. Predicted accuracy ({name})")
        ax.set_xlabel("True difference [dB]")
        ax.set_ylabel("Predicted difference [dB]")

        ax.scatter(
            real, predicted, s=3.0, alpha=0.25, rasterized=True, zorder=1, c=color
        )
        ax.plot(
            [-30, 30], [-30, 30], linestyle="--", alpha=0.75, color="black", zorder=2
        )
        ax.set_xticks([-20, 0, 20])
        ax.set_yticks([-20, 0, 20])

        fig_path = output_path / f"eval_rel_{name}_viz.pdf"
        save_figure(fig, fig_path)

    json_path = output_path / f"eval_rel_{name}_results.json"
    print(f"Saving statistics to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(
            {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "pcc": pcc,
                "duration": (end_time - start_time).total_seconds(),
                "name": name,
            },
            f,
            indent=2,
        )


def evaluate_classifier(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    loader: DataLoader,
    adder_loader: DataLoader,
    n_features: int,
    output_path: Path,
    name: str,
    use_embed: bool,
    cutoff: float = 0.5,
):
    _prepare_output(output_path)

    with torch.no_grad():
        d_a, d_b = next(iter(loader))

        diff: torch.Tensor = (d_a.y - d_b.y).unsqueeze(-1)
        real: torch.Tensor = ((d_a.y - d_b.y) > 0).unsqueeze(-1).float()

        if use_embed:
            embeddings = create_adder_embeddings(embed_model, adder_loader, n_features)
            d_a.x = embeddings[d_a.adder_id]
            d_b.x = embeddings[d_b.adder_id]

        start_time = datetime.now()
        predicted_logits = regress_model(d_a, d_b)
        predicted: torch.Tensor = F.sigmoid(predicted_logits)
        end_time = datetime.now()

        correct = [
            (r, p)
            for r, p in zip(diff, predicted)
            if r <= 0 and p <= cutoff or r > 0 and p > cutoff
        ]
        incorrect = [
            (r, p)
            for r, p in zip(diff, predicted)
            if r <= 0 and p > cutoff or r > 0 and p <= cutoff
        ]

        correct_percent = 100 * len(correct) / len(real)

        bce = F.binary_cross_entropy_with_logits(predicted_logits, real).item()
        print(f"Evaluation results -> BCE: {bce}, Correct: {correct_percent} %")

        fig, ax = prepare_pyplot(square=True)
        ax.set_xlabel("PSNR difference [dB]")
        ax.set_ylabel("Classifier output")

        def sigmoid(x, x0, k):
            return 1 / (1 + torch.exp(-k * (x - x0)))

        popt, _ = curve_fit(sigmoid, diff.squeeze(-1), predicted.squeeze(-1), p0=[0, 1])

        sx = torch.linspace(-40, 40, 100)
        sy = sigmoid(sx, popt[0], popt[1])

        ax.plot(sx, sy, alpha=0.9, linestyle="--", color="gray", zorder=1)

        ax.scatter(
            [r[0] for r in correct],
            [r[1] for r in correct],
            s=3.0,
            alpha=0.1,
            rasterized=True,
            zorder=2,
        )
        ax.scatter(
            [r[0] for r in incorrect],
            [r[1] for r in incorrect],
            s=3.0,
            alpha=0.1,
            c="red",
            rasterized=True,
            zorder=3,
        )

        fig_path = output_path / f"eval_class_{name}_viz.pdf"
        save_figure(fig, fig_path)

    json_path = output_path / f"eval_class_{name}_results.json"
    print(f"Saving statistics to: {json_path}")
    with open(json_path, "w") as f:
        json.dump(
            {
                "bce": bce,
                "correct%": correct_percent,
                "duration": (end_time - start_time).total_seconds(),
                "name": name,
            },
            f,
            indent=2,
        )
