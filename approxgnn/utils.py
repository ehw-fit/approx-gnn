import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
import networkx as nx
from typing import Iterable

import random
from pathlib import Path
from tqdm import tqdm

import json
import csv
from zipfile import ZipFile

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from .convert import MODULE_ATTRIBUTES, OpWord


def load_component_parameters(
    components_path: str | Path, objective: str = "mse"
) -> dict[str, dict[str, float]]:
    """Loads component error metrics from EvoApproxLib.

    :param components_path: EvoApproxLib meta JSON path.
    :param objective: EvoApproxLib objective.
    :return: Mapping of component names to metric dictionary.
    :rtype: `dict[str, dict[str, float]]`
    """
    components_path = Path(components_path)

    with open(components_path) as f:
        meta = json.load(f)

    components: dict[str, dict[str, float]] = {}

    for group in meta:
        if group["folder"] == "adders":
            for dataset in group["datasets"]:
                for subdataset in dataset["datasets"]:
                    if subdataset["folder"].split("/")[-1] != f"pareto_pwr_{objective}":
                        continue

                    for instance in subdataset["instances"]:
                        name = instance["name"]
                        params = instance["params"]
                        components[name] = {
                            attr: float(params[attr])
                            for attr in [
                                "mae%",
                                "mre%",
                                "ep%",
                                "wce%",
                                "wcre%",
                                "area",
                                "pwr",
                                "delay",
                            ]
                            if attr in params
                        }
    return components


def get_verilog_paths(
    components_path: str | Path,
    objective: str = "mse",
    unsigned: bool = True,
    signed: bool = False,
) -> dict[str, Path]:
    """Gets a mapping of component names to Verilog paths from EvoApproxLib.

    :param components_path: EvoApproxLib meta JSON path.
    :param objective: EvoApproxLib objective.
    :param unsigned: Load unsigned components.
    :param signed: Load signed components.
    :return: Mapping of component names to Verilog paths.
    :rtype: `dict[str, Path]`
    """

    components_path = Path(components_path)
    with open(components_path) as f:
        meta = json.load(f)

    components: dict[str, Path] = {}

    for group in meta:
        if group["folder"] == "adders":
            for dataset in group["datasets"]:
                if not signed and dataset["signed"]:
                    continue
                elif not unsigned and not dataset["signed"]:
                    continue

                for subdataset in dataset["datasets"]:
                    if subdataset["folder"].split("/")[-1] != f"pareto_pwr_{objective}":
                        continue

                    for instance in subdataset["instances"]:
                        name = instance["name"]
                        files = instance["files"]
                        try:
                            verilog = next(
                                x["file"] for x in files if x["type"] == "Verilog file"
                            )
                        except StopIteration:
                            continue

                        components[name] = Path(subdataset["folder"]) / verilog
    return components


def _instantiate_template(
    instance_id: str,
    wirings: dict[str, nx.DiGraph],
    configs: dict[str, dict[str, str | list[str]]],
    qor: dict[str, dict[str, float]],
    metrics: list[str],
    targets: str | list[str] = "psnr",
    components: dict[str, dict[str, float]] | None = None,
    adder_to_id: dict[str, int] | None = None,
    critical_path: Iterable[str] | None = None,
) -> tuple[Data, str]:
    """Instantiates an accelerator from a networkx graph and a component assignment.

    Sets node_critical when critical_path is used.

    Sets adder_id to assigned components' IDs when adder_to_id is used.

    :param instance_id: Configuration ID.
    :param wirings: Mapping of accelerator IDs to networkx graphs.
    :param configs: Configuration data (ID -> {assignments, wiring ID}).
    :param qor: Mapping of configuration IDs to QoR results.
    :param components: Component parameters.
    :param adder_to_id: Mapping of component names to IDs.
    :param critical_path: List of nodes IDs on the critical path.
    :return: PyG graph and wiring ID.
    :rtype: `tuple[Data, str]`
    """
    if components is None and adder_to_id is None:
        raise ValueError("Either components or adder_to_id has to be defined.")

    wiring_id: str = configs[instance_id]["wiring"]
    wiring = wirings[wiring_id]

    if critical_path is not None:
        for node, data in wiring.nodes(data=True):
            data["node_critical"] = 1 if node in critical_path else 0

    graph = from_networkx(wiring)
    graph.x = torch.zeros((len(graph.x), len(metrics)), dtype=torch.float32)

    module_indices = {
        index.item(): i for i, index in enumerate(graph.adder_id) if index >= 0
    }

    q = qor[instance_id]
    if type(targets) is str:
        graph.y = torch.tensor(q[targets])
    else:
        graph.y = torch.tensor([q[target] for target in targets])

    if adder_to_id is not None:
        for i, v in enumerate(configs[instance_id]["assignments"]):
            graph.adder_id[module_indices[i]] = adder_to_id[v]
    else:

        def adjust_attr(attr: str, component):
            if attr not in component:
                return 0.0
            if attr.endswith("%"):

                return component[attr] * 0.01
            else:
                return component[attr]

        for i, v in enumerate(configs[instance_id]["assignments"]):
            comp = components[v]
            graph.x[module_indices[i]] = torch.tensor(
                [adjust_attr(attr, comp) for attr in metrics]
            )

    graph.id = instance_id
    return graph, wiring_id


def load_dataset(
    dir: str | Path,
    components: dict[str, dict[str, float]] | None = None,
    adder_to_id: dict[str, int] | None = None,
    add_critical_paths: bool = False,
    component_graphs: dict[str, nx.DiGraph] | None = None,
    targets: str | list[str] = "psnr",
    metrics: list[str] = MODULE_ATTRIBUTES,
) -> tuple[list[Data], list[str]]:
    """Loads a dataset from a directory.

    Sets node_critical when add_critical_paths and component graphs is used.

    Sets adder_id to assigned components' IDs when adder_to_id is used.

    :param dir: Dataset directory.
    :param components: Component parameters. Used for metric-based prediction.
    :param adder_to_id: Mapping of component names to IDs.
    :param add_critical_paths: Set critical paths.
    :param component_graphs: Component networkx graphs.
    :return: List of PyG graphs and accelerator IDs they were instantiated from.
    :rtype: `tuple[list[Data], list[str]]`
    """
    if components is None and adder_to_id is None:
        raise ValueError("Either components or adder_to_id has to be defined.")

    dir = Path(dir)

    with open(dir / "results.csv") as f:
        reader = csv.DictReader(f)
        qor: dict[str, dict[str, float]] = {
            x["config"]: {
                k: float(v) for k, v in x.items() if k not in ["config", "wiring"]
            }
            for x in reader
        }
    with open(dir / "_eval.tsv") as f:
        configs: dict[str, dict[str, str | list[str]]] = {}
        for line in f:
            words = line.split()
            configs[words[0]] = {"wiring": words[1], "assignments": words[4:]}

    with ZipFile(dir / "accelerators.json.zip") as f:
        wirings = json.loads(f.read("accelerators.json").decode())

    wirings = {k: nx.adjacency_graph(v["graph"]) for k, v in wirings.items()}

    if add_critical_paths:
        if component_graphs is None:
            raise ValueError(
                "component_graphs has to be set when adding critical paths."
            )

        component_critical_paths = {
            k: find_longest_paths(v) for k, v in component_graphs.items()
        }
        component_critical_lengths = {
            k: [len(x) for x in v] for k, v in component_critical_paths.items()
        }

        longest_paths = {
            cid: find_longest_paths(
                wirings[config["wiring"]],
                config["assignments"],
                component_critical_lengths,
            )
            for cid, config in configs.items()
        }

        critical_paths = {
            cid: set(max((len(p), p) for p in paths)[1])
            for cid, paths in longest_paths.items()
        }
    else:
        critical_paths = None

    dataset = [
        _instantiate_template(
            instance_id,
            wirings,
            configs,
            qor,
            metrics,
            targets=targets,
            components=components,
            adder_to_id=adder_to_id,
            critical_path=(
                critical_paths[instance_id] if critical_paths is not None else None
            ),
        )
        for instance_id in tqdm(
            configs, total=len(configs), ncols=100, desc="Loading dataset"
        )
        if instance_id in qor
    ]

    graphs = [d[0] for d in dataset]
    wiring_ids = [d[1] for d in dataset]
    return graphs, wiring_ids


def random_color():
    """Creates a random color for matplotlib."""
    r = hex(random.randint(0, 255))[2:].ljust(2, "0")
    g = hex(random.randint(0, 255))[2:].ljust(2, "0")
    b = hex(random.randint(0, 255))[2:].ljust(2, "0")
    return f"#{r}{g}{b}"


def prepare_pyplot(
    square: bool = False,
    width: float | None = None,
    height: float | None = None,
    grayscale: bool = False,
    rows: int = 1,
    cols: int = 1,
    shared_x: bool = False,
    shared_y: bool = False,
) -> tuple[Figure, Axes | list[Axes] | list[list[Axes]]]:
    """Prepares matplotlib for publication figures.

    :param square: Create a square figure.
    :param height: Height override.
    :param grayscale: Set to grayscale.
    :return: Figure and axis.
    """
    if grayscale:
        plt.style.use("grayscale")

    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 14

    if square:
        size = (
            3.5 if width is None else width,
            3.5 if height is None else height,
        )
    else:
        size = (7 if width is None else width, 4 if height is None else height)

    fig, ax = plt.subplots(
        nrows=rows,
        ncols=cols,
        figsize=size,
        sharex=shared_x,
        sharey=shared_y,
    )

    return fig, ax


def save_figure(
    fig: Figure,
    fig_path: Path,
    joined: bool = False,
    top: float | None = None,
    bottom: float | None = None,
    right: float | None = None,
    left: float | None = None,
):
    print(f"Saving figure to: {fig_path}")
    if joined:
        fig.subplots_adjust(wspace=0, hspace=0)

    if top is not None:
        fig.subplots_adjust(top=top)
    if bottom is not None:
        fig.subplots_adjust(bottom=bottom)
    if right is not None:
        fig.subplots_adjust(right=right)
    if left is not None:
        fig.subplots_adjust(left=left)

    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def find_loops(
    node: str, G: nx.DiGraph, in_use: dict[str, bool], loops: set[tuple[str, str]]
):
    """Finds edges creating loops and adds them to the `loops` argument.

    :param node: Root node name.
    :param G: Input graph.
    :param in_use: Table marking nodes on currently explored path.
    :param loops: Resulting set of edges.
    """
    in_use[node] = True
    for _, child in G.out_edges(node):
        if in_use[child]:
            loops.add((node, child))
        else:
            find_loops(child, G, in_use, loops)
    in_use[node] = False


def find_longest_paths(
    G: nx.DiGraph,
    assignments: list[str] | None = None,
    modules: dict[str, list[int]] | None = None,
):
    """Finds longest paths in a given graph.

    :param G: Input graph.
    :param assignments: List of adder assignments.
    :param modules: Mapping of component names to critical paths from their inputs.
    """
    input_nodes = [
        node[0]
        for node in G.nodes(data=True)
        if OpWord(node[1]["node_type"]) == OpWord.INPUT
    ]
    output_nodes = [
        node[0]
        for node in G.nodes(data=True)
        if OpWord(node[1]["node_type"]) == OpWord.OUTPUT
    ]

    paths: list[list[str]] = []

    for input_node in input_nodes:
        loops: set[tuple[str, str]] = set()
        find_loops(input_node, G, {k: False for k in G.nodes}, loops)

        pqueue: list[(int, str, str)] = [(0, input_node, "")]
        distances: dict[str, tuple[str, int]] = {input_node: ("", 0)}

        while len(pqueue) > 0:
            current_dist, node, _ = pqueue.pop()

            if current_dist < distances[node][1]:
                # replaced by other path
                continue

            for _, child in G.out_edges(node):
                if (node, child) in loops:
                    continue

                if OpWord(G.nodes[child]["node_type"]) == OpWord.MODULE:
                    if modules is not None and assignments is not None:
                        adder_id = G.nodes[child]["adder_id"]
                        edge_id = G.edges[(node, child)]["edge_id"]
                        adjusted_dist = (
                            current_dist + modules[assignments[adder_id]][edge_id]
                        )
                else:
                    adjusted_dist = current_dist

                if child not in distances or adjusted_dist >= distances[child][1]:
                    new_dist = adjusted_dist + 1
                    distances[child] = (node, new_dist)
                    pqueue.append((new_dist, child, node))

            pqueue = sorted(pqueue)

        target_node = output_nodes[0]
        path = []

        while target_node != "":
            path.append(target_node)
            target_node = distances[target_node][0]

        paths.append(list(path[::-1]))

    return paths
