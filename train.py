from argparse import ArgumentParser, ArgumentTypeError, BooleanOptionalAction
from tqdm import tqdm
from pathlib import Path

import torch
from torch_geometric.utils import from_networkx
from torch_geometric.loader import DataLoader
import random
import math
import numpy as np

from approxgnn.models import (
    AdderEmbedding,
    GNNRegressor,
    GNNRelative,
)

from approxgnn.convert import convert_to_graph
from approxgnn.utils import (
    get_verilog_paths,
    load_component_parameters,
    load_dataset,
)

from approxgnn.training import (
    train_absolute,
    train_relative,
    train_classifier,
    save_run,
    ClassPairDataset,
)
from approxgnn.evaluation import (
    evaluate_absolute,
    evaluate_relative,
    evaluate_classifier,
)
from datetime import datetime


def arg_type_check(arg: str, values: list[str]):
    if arg not in values:
        raise ArgumentTypeError(
            f"{arg} is not a valid value. It has to be one of {', '.join(values)}."
        )
    return arg


def arg_range_check(arg: str, lb: float, ub: float):
    try:
        value = float(arg)
    except ValueError as e:
        raise ArgumentTypeError(
            f"Provided value has to be a float in range [{lb}, {ub}]"
        ) from e

    if value < lb or value > ub:
        raise ArgumentTypeError(f"Provided value has to be in range [{lb}, {ub}]")

    return value


TASK_ABSOLUTE = "absolute"
TASK_RELATIVE = "relative"
TASK_CLASSIFIER = "classify"

argp = ArgumentParser()
argp.add_argument("dataset", type=Path, help="Dataset folder.")
argp.add_argument(
    "--components",
    type=Path,
    help="Components JSON file.",
    default=Path("components/components.json"),
)

argp.add_argument(
    "-t",
    "--task",
    type=lambda x: arg_type_check(x, [TASK_ABSOLUTE, TASK_RELATIVE, TASK_CLASSIFIER]),
    default="absolute",
    help="Model task.",
)

argp.add_argument(
    "-r", "--targets", nargs="+", default=["psnr"], help="Network outputs to train."
)

argp.add_argument(
    "-m",
    "--metrics",
    nargs="+",
    default=["ep%", "mae%", "mre%", "wce%", "wcre%"],
    help="Metrics for non-embedding models",
)

argp.add_argument("--limit", type=int, help="Limit number of training items.")

argp.add_argument("--lr", type=float, default=8e-4)
argp.add_argument("--elr", "--embed-lr", dest="embed_lr", type=float, default=8e-4)
argp.add_argument("-e", "--epochs", type=int, default=150)
argp.add_argument("--features", type=int, default=8)
argp.add_argument("--inner-channels", type=int, default=32)
argp.add_argument("--hidden-layers", type=int, default=2)
argp.add_argument("--head-channels", type=int, default=256)
argp.add_argument("--embed-inner-channels", type=int, default=16)
argp.add_argument("--embed-hidden-layers", type=int, default=2)

argp.add_argument(
    "-l",
    "--loss",
    type=lambda x: arg_type_check(x, ["mse", "huber", "bce", "mae"]),
    default="mse",
    help="Loss criterion. One of mse, huber, bce.",
)

argp.add_argument("-n", "--name", help="Dataset name.")

argp.add_argument(
    "-o", "--output", type=Path, default=Path("outputs"), help="Output directory root."
)

argp.add_argument(
    "--regress-pretrained", type=Path, help="Path to pretrained regression network."
)
argp.add_argument(
    "--embed-pretrained", type=Path, help="Path to pretrained embedding network."
)

argp.add_argument(
    "--train-regress",
    action=BooleanOptionalAction,
    default=True,
    help="Train regression network.",
)
argp.add_argument(
    "--train-embed",
    action=BooleanOptionalAction,
    default=True,
    help="Train embedding network.",
)
argp.add_argument(
    "--embed", action=BooleanOptionalAction, default=True, help="Use embeddings."
)
argp.add_argument(
    "--split",
    type=lambda x: arg_range_check(x, 0.01, 0.99),
    default=0.2,
    help="Validation split - fraction of total data used for validation.",
)
argp.add_argument("--critical-path", action="store_true", help="Use critical paths.")
argp.add_argument("-s", "--seed", type=int, help="RNG seed.")

args = argp.parse_args()

now = datetime.now()
output_name = (
    f"{args.task}_{now.year}-{now.month}-{now.day}_{now.hour}-{now.minute}-{now.second}"
)

if args.name is not None:
    name = args.name
else:
    name = args.dataset.stem

output_name = f"{name}_{output_name}"
output_path = args.output / output_name

if not args.embed:
    args.features = len(args.metrics)

if args.seed is None:
    args.seed = random.randint(0, 2147483647)

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

try:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
except Exception as e:
    print(f"Failed to make torch deterministic: {e}")

config = {
    "components": str(args.components.resolve()),
    "dataset": str(args.dataset.resolve()),
    "task": args.task,
    "lr": args.lr,
    "epochs": args.epochs,
    "features": args.features,
    "inner_channels": args.inner_channels,
    "hidden_layers": args.hidden_layers,
    "head_channels": args.head_channels,
    "embed_inner_channels": args.embed_inner_channels,
    "embed_hidden_layers": args.embed_hidden_layers,
    "loss": args.loss,
    "output": str(args.output.resolve()),
    "output_name": output_name,
    "name": args.name,
    "train_regress": args.train_regress,
    "train_embed": args.train_embed,
    "regress_pretrained": (
        str(args.regress_pretrained) if args.regress_pretrained is not None else None
    ),
    "embed_pretrained": (
        str(args.embed_pretrained) if args.embed_pretrained is not None else None
    ),
    "embed": args.embed,
    "seed": args.seed,
    "critical_path": args.critical_path,
    "targets": args.targets,
    "limit": args.limit,
    "split": args.split,
    "metrics": args.metrics,
}

if not args.components.exists():
    print("Components path doesn't exist.")
    exit(-1)

if not args.dataset.exists() or not args.dataset.is_dir():
    print("Dataset path doesn't exist or isn't a directory.")
    exit(-1)

if not args.output.exists():
    args.output.mkdir()

if not args.output.is_dir():
    print("Output root path isn't a directory.")
    exit(-1)

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
            metrics=args.metrics,
        )
    else:
        dataset, wiring_ids = load_dataset(
            args.dataset,
            adder_to_id=adder_name_to_id,
            targets=args.targets,
            metrics=args.metrics,
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
            metrics=args.metrics,
        )
    else:
        dataset, wiring_ids = load_dataset(
            args.dataset,
            components=component_dict,
            targets=args.targets,
            metrics=args.metrics,
        )

total_count = len(dataset)
validation_count = max(min(math.ceil(total_count * args.split), total_count - 1), 1)
validation, training = dataset[:validation_count], dataset[validation_count:]
val_wirings, train_wirings = (
    wiring_ids[:validation_count],
    wiring_ids[validation_count:],
)

if args.limit is not None:
    indices = random.sample(list(range(len(training))), k=args.limit)
    training = [training[i] for i in indices]
    train_wirings = [train_wirings[i] for i in indices]

if args.task == TASK_ABSOLUTE:
    loader = DataLoader(training, batch_size=512, shuffle=True)
    val_loader = DataLoader(validation, batch_size=len(validation))
else:
    loader = DataLoader(ClassPairDataset(training, train_wirings), batch_size=512)
    val_loader = DataLoader(
        ClassPairDataset(validation, val_wirings), batch_size=len(validation)
    )

if args.loss == "mse":
    criterion: torch.nn.Module = torch.nn.MSELoss()
elif args.loss == "huber":
    criterion = torch.nn.HuberLoss()
elif args.loss == "bce":
    criterion = torch.nn.BCEWithLogitsLoss()
elif args.loss == "mae":
    criterion = torch.nn.L1Loss()
else:
    raise ValueError("Invalid loss criterion.")

embed_model = AdderEmbedding(
    args.features, args.embed_inner_channels, args.embed_hidden_layers
)
if args.embed_pretrained is not None:
    embed_model.load_state_dict(torch.load(args.embed_pretrained, weights_only=True))

if args.task == TASK_ABSOLUTE:
    regress_model: torch.nn.Module = GNNRegressor(
        args.features,
        args.inner_channels,
        args.hidden_layers,
        args.head_channels,
        critical_path=args.critical_path,
    )
elif args.task == TASK_RELATIVE:
    regress_model = GNNRelative(args.features)
    # regress_model = GNNRelative(args.features, args.inner_channels, args.hidden_layers, args.head_channels)
elif args.task == TASK_CLASSIFIER:
    regress_model = GNNRelative(args.features)
    criterion = torch.nn.BCEWithLogitsLoss()
    # regress_model = GNNClassifier(args.features, args.inner_channels, args.hidden_layers, args.head_channels)

if args.regress_pretrained is not None:
    regress_model.load_state_dict(
        torch.load(args.regress_pretrained, weights_only=True)
    )


regress_optimizer = torch.optim.AdamW(regress_model.parameters(), lr=args.lr)
regress_scheduler = torch.optim.lr_scheduler.LinearLR(
    regress_optimizer, 1.0, 0.1, args.epochs
)
embed_optimizer = torch.optim.AdamW(embed_model.parameters(), lr=args.embed_lr)
embed_scheduler = torch.optim.lr_scheduler.LinearLR(
    embed_optimizer, 1.0, 0.1, args.epochs
)

if args.task == TASK_ABSOLUTE:
    results = train_absolute(
        regress_model,
        embed_model,
        criterion,
        regress_optimizer,
        embed_optimizer,
        loader,
        adder_loader,
        val_loader,
        args.features,
        args.epochs,
        regress_scheduler=regress_scheduler,
        embed_scheduler=embed_scheduler,
        train_embed=args.train_embed and args.embed,
        train_regress=args.train_regress,
        use_embed=args.embed,
    )

    save_run(config, results, output_path)

    regress_model.load_state_dict(results["best_regress"])
    regress_model = regress_model.eval()
    embed_model.load_state_dict(results["best_embed"])
    embed_model = embed_model.eval()

    evaluate_absolute(
        regress_model,
        embed_model,
        val_loader,
        adder_loader,
        args.features,
        output_path,
        args.name,
        args.embed,
    )
elif args.task == TASK_RELATIVE:
    results = train_relative(
        regress_model,
        embed_model,
        criterion,
        regress_optimizer,
        embed_optimizer,
        loader,
        adder_loader,
        val_loader,
        args.features,
        args.epochs,
        regress_scheduler=regress_scheduler,
        embed_scheduler=embed_scheduler,
        train_embed=args.train_embed and args.embed,
        train_regress=args.train_regress,
        use_embed=args.embed,
    )

    save_run(config, results, output_path)

    regress_model.load_state_dict(results["best_regress"])
    regress_model = regress_model.eval()
    embed_model.load_state_dict(results["best_embed"])
    embed_model = embed_model.eval()

    evaluate_relative(
        regress_model,
        embed_model,
        val_loader,
        adder_loader,
        args.features,
        output_path,
        args.name,
        args.embed,
    )
elif args.task == TASK_CLASSIFIER:
    results = train_classifier(
        regress_model,
        embed_model,
        criterion,
        regress_optimizer,
        embed_optimizer,
        loader,
        adder_loader,
        val_loader,
        args.features,
        args.epochs,
        regress_scheduler=regress_scheduler,
        embed_scheduler=embed_scheduler,
        train_embed=args.train_embed and args.embed,
        train_regress=args.train_regress,
        use_embed=args.embed,
    )

    save_run(config, results, output_path)

    regress_model.load_state_dict(results["best_regress"])
    regress_model = regress_model.eval()
    embed_model.load_state_dict(results["best_embed"])
    embed_model = embed_model.eval()

    evaluate_classifier(
        regress_model,
        embed_model,
        val_loader,
        adder_loader,
        args.features,
        output_path,
        args.name,
        args.embed,
    )
