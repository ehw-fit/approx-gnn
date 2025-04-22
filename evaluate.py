from tqdm import tqdm
from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from pathlib import Path
import json

import torch
import torch.nn.functional as F
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch, Data
import random
import sys

from approxgnn.models import (
    load_models,
)
from approxgnn.evaluation import (
    evaluate_absolute,
    evaluate_relative,
    evaluate_classifier,
)
from approxgnn.convert import convert_to_graph
from approxgnn.training import ClassPairDataset, create_adder_embeddings
from approxgnn.utils import (
    get_verilog_paths,
    load_component_parameters,
    load_dataset,
)


TASK_ABSOLUTE = "absolute"
TASK_RELATIVE = "relative"
TASK_CLASSIFIER = "classify"


def arg_type_check(arg: str, values: list[str]):
    if arg not in values:
        raise ArgumentTypeError(
            f"{arg} is not a valid value. Has to be one of {', '.join(values)}."
        )
    return arg


argp = ArgumentParser()
argp.add_argument("dataset", type=Path, help="Dataset folder.")
argp.add_argument("models", nargs="+", type=Path, help="Model directory paths.")

argp.add_argument(
    "--embed", action=BooleanOptionalAction, default=True, help="Use embeddings."
)

argp.add_argument("--critical-path", action="store_true", help="Use critical paths.")
argp.add_argument("--offset", action="store_true", help="Add corrective offset.")
argp.add_argument(
    "--offset-samples",
    type=int,
    default=20,
    help="Number of corrective offset samples.",
)

argp.add_argument(
    "--components",
    type=Path,
    help="Components JSON file.",
    default=Path("components/components.json"),
)

argp.add_argument(
    "--task",
    type=lambda x: arg_type_check(x, [TASK_ABSOLUTE, TASK_RELATIVE, TASK_CLASSIFIER]),
    default="absolute",
    help="Model type.",
)

argp.add_argument(
    "-r", "--targets", nargs="+", default=["psnr"], help="Network outputs."
)

argp.add_argument("--features", type=int, default=8)

argp.add_argument("--name", help="Dataset name.")

argp.add_argument("--output", type=Path, help="Output directory root.")
argp.add_argument(
    "--split",
    action=BooleanOptionalAction,
    default=True,
    help="Split dataset into validation set, same as in training script.",
)

argp.add_argument("-c", "--color", type=str, default="tab:blue", help="Point colors")

args = argp.parse_args()

if not args.components.exists():
    print("Components path doesn't exist.")
    exit(-1)

if not args.dataset.exists() or not args.dataset.is_dir():
    print("Dataset path doesn't exist or isn't a directory.")
    exit(-1)

if args.output is not None:
    output_path = args.output
else:
    # TODO: fix
    output_path = args.models[0]

verilog_paths = get_verilog_paths(args.components)

adder_name_to_id = {k: i for i, k in enumerate(verilog_paths)}
adder_graphs_nx = {
    k: convert_to_graph(
        Path(v).stem,
        verilog_path=args.components.parent / v,
        n_features=0,
        as_networkx=True,
    )
    for k, v in tqdm(
        verilog_paths.items(),
        total=len(verilog_paths),
        ncols=100,
        desc="Loading components",
    )
}
adder_graphs = {k: from_networkx(v) for k, v in adder_graphs_nx.items()}

adder_loader = DataLoader(
    list(adder_graphs.values()), batch_size=len(adder_graphs), shuffle=False
)

if args.embed:
    if args.critical_path:
        dataset, wiring_ids = load_dataset(
            args.dataset,
            adder_to_id=adder_name_to_id,
            add_critical_paths=True,
            component_graphs=adder_graphs_nx,
            targets=args.targets,
        )
    else:
        dataset, wiring_ids = load_dataset(
            args.dataset, adder_to_id=adder_name_to_id, targets=args.targets
        )

else:
    component_dict = load_component_parameters(args.components)
    if args.critical_path:
        dataset, wiring_ids = load_dataset(
            args.dataset,
            components=component_dict,
            add_critical_paths=True,
            component_graphs=adder_graphs_nx,
            targets=args.targets,
        )
    else:
        dataset, wiring_ids = load_dataset(
            args.dataset, components=component_dict, targets=args.targets
        )

if args.split:
    total_count = len(dataset)
    validation_count = total_count // 5
    validation, training = dataset[:validation_count], dataset[validation_count:]
    val_wirings, train_wirings = (
        wiring_ids[:validation_count],
        wiring_ids[validation_count:],
    )
else:
    validation = dataset
    val_wirings = wiring_ids
    if args.offset:
        print("When using corrective offset, --split should be used.", file=sys.stderr)
        exit(-1)


def get_mean_offset(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    samples: list[Data],
    adder_loader: DataLoader,
    n_features: int,
    use_embed: bool,
):
    with torch.no_grad():
        d = Batch.from_data_list(samples)
        real: torch.Tensor = d.y.unsqueeze(-1)

        if use_embed:
            embeddings = create_adder_embeddings(embed_model, adder_loader, n_features)
            d.x = embeddings[d.adder_id]

        predicted: torch.Tensor = regress_model(d)

        offset = (real - predicted).mean().item()
        print(f"Offset -> {offset}")

    return offset


def get_mean_offset_relative(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    samples: list[tuple[Data, Data]],
    adder_loader: DataLoader,
    n_features: int,
    use_embed: bool,
):
    with torch.no_grad():
        d_a = Batch.from_data_list([x[0] for x in samples])
        d_b = Batch.from_data_list([x[1] for x in samples])
        real: torch.Tensor = (d_a.y - d_b.y).unsqueeze(-1)

        if use_embed:
            embeddings = create_adder_embeddings(embed_model, adder_loader, n_features)
            d_a.x = embeddings[d_a.adder_id]
            d_b.x = embeddings[d_b.adder_id]

        predicted: torch.Tensor = regress_model(d_a, d_b)

        offset = (real - predicted).mean().item()
        print(f"Offset -> {offset}")

    return offset


def get_mean_offset_classify(
    regress_model: torch.nn.Module,
    embed_model: torch.nn.Module,
    samples: list[tuple[Data, Data]],
    adder_loader: DataLoader,
    n_features: int,
    use_embed: bool,
):
    with torch.no_grad():
        d_a = Batch.from_data_list([x[0] for x in samples])
        d_b = Batch.from_data_list([x[1] for x in samples])

        real: torch.Tensor = (d_a.y - d_b.y).unsqueeze(-1)

        if use_embed:
            embeddings = create_adder_embeddings(embed_model, adder_loader, n_features)
            d_a.x = embeddings[d_a.adder_id]
            d_b.x = embeddings[d_b.adder_id]

        predicted: torch.Tensor = F.sigmoid(regress_model(d_a, d_b))
        s_pred = sorted(zip(predicted, real))

        best_correct = 0
        best_cutoff = 0.0

        for a, b in zip(s_pred, s_pred[1:]):
            cutoff = 0.5 * (a[0] + b[0])
            n_correct = len([p for p, r in s_pred if p <= cutoff and r <= 0.0]) + len(
                [p for p, r in s_pred if p > cutoff and r > 0.0]
            )
            if n_correct > best_correct:
                best_correct = n_correct
                best_cutoff = cutoff.item()

        print(f"Cutoff -> {best_cutoff}")

    return best_cutoff


for model in args.models:
    embed_model, regress_model, _, _ = load_models(model, load_weights=True)
    if embed_model is not None:
        embed_model = embed_model.eval()
    regress_model = regress_model.eval()

    if (model / "config.json").exists():
        with open(model / "config.json") as f:
            config = json.load(f)

        print("Loaded configuration.")
        args.task = config["task"]
        args.embed = config["embed"]
        args.features = config["features"]
        args.name = config["output_name"]

    if args.task == TASK_ABSOLUTE:
        val_loader = DataLoader(validation, batch_size=len(validation))
    else:
        val_loader = DataLoader(
            ClassPairDataset(validation, val_wirings), batch_size=len(validation)
        )

    if args.offset and args.task != TASK_ABSOLUTE:
        offset_dataset = ClassPairDataset(training, train_wirings)
        offset_sources = [
            offset_dataset[random.randint(0, len(offset_dataset) - 1)]
            for _ in range(args.offset_samples)
        ]

    if args.task == TASK_ABSOLUTE:
        if args.offset:
            offset = get_mean_offset(
                regress_model,
                embed_model,
                random.sample(training, args.offset_samples),
                adder_loader,
                args.features,
                args.embed,
            )
        else:
            offset = 0.0

        evaluate_absolute(
            regress_model,
            embed_model,
            val_loader,
            adder_loader,
            n_features=args.features,
            output_path=output_path,
            name=args.name,
            use_embed=args.embed,
            offset=offset,
            color=args.color,
        )
    elif args.task == TASK_RELATIVE:
        if args.offset:
            offset = get_mean_offset_relative(
                regress_model,
                embed_model,
                offset_sources,
                adder_loader,
                args.features,
                args.embed,
            )
        else:
            offset = 0.0

        evaluate_relative(
            regress_model,
            embed_model,
            val_loader,
            adder_loader,
            n_features=args.features,
            output_path=output_path,
            name=args.name,
            use_embed=args.embed,
            offset=offset,
            color=args.color,
        )
    elif args.task == TASK_CLASSIFIER:
        if args.offset:
            cutoff = get_mean_offset_classify(
                regress_model,
                embed_model,
                offset_sources,
                adder_loader,
                args.features,
                args.embed,
            )
        else:
            cutoff = 0.5

        evaluate_classifier(
            regress_model,
            embed_model,
            val_loader,
            adder_loader,
            n_features=args.features,
            output_path=output_path,
            name=args.name,
            use_embed=args.embed,
            cutoff=cutoff,
        )
