import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import Batch
import numpy as np
import networkx as nx

from approxgnn.dse import create_nsga2, AutoaxProblem
from approxgnn.utils import load_component_parameters, get_verilog_paths
from approxgnn.convert import convert_to_graph
from approxgnn.models import load_models
from approxgnn.generate import generate_verilog, get_components_and_code

from pymoo.optimize import minimize
from pymoo.core.population import Population
from pymoo.core.individual import Individual

from collections import defaultdict
from zipfile import ZipFile
import json
import csv
from pathlib import Path
from argparse import ArgumentParser


argp = ArgumentParser()
argp.add_argument("accelerator", type=Path, help="Path to an accelerator ZIP file.")
argp.add_argument("qor", type=Path, help="Path to a QoR model directory.")
argp.add_argument("hw", type=Path, help="Path to a HW model directory.")
argp.add_argument("-o", "--output", type=Path, default=Path("outputs"))
argp.add_argument("-g", "--generations", type=int, default=40)
argp.add_argument("-p", "--population", type=int, default=100)
argp.add_argument("-m", "--mutation", type=float, default=0.25)
argp.add_argument("--components", type=Path, default=Path("components/components.json"))

args = argp.parse_args()

component_params = load_component_parameters(args.components)
verilog_paths = get_verilog_paths(args.components)
adder_name_to_id = {k: i for i, k in enumerate(verilog_paths)}
components, component_c_code = get_components_and_code(
    [8, 12, 16], args.components, component_params
)

adder_graphs_nx = {
    k: convert_to_graph(
        Path(v).stem,
        verilog_path=args.components.parent / v,
        n_features=0,
        as_networkx=True,
    )
    for k, v in verilog_paths.items()
}
adder_graphs = {k: from_networkx(v) for k, v in adder_graphs_nx.items()}

adder_batch = Batch.from_data_list(list(adder_graphs.values()))

qor_embed, qor_model, _, _ = load_models(args.qor, load_weights=True)
hw_embed, hw_model, _, _ = load_models(args.hw, load_weights=True)

qor_embeddings = torch.concat(
    (
        qor_embed(adder_batch),
        torch.zeros((1, 8)),
    ),
    dim=0,
)
hw_embeddings = torch.concat(
    (
        hw_embed(adder_batch),
        torch.zeros((1, 8)),
    ),
    dim=0,
)

with ZipFile(args.accelerator) as f:
    accelerators = json.loads(f.read("accelerators.json").decode())

accelerator = next(iter(accelerators.values()))
accelerator_id = next(iter(accelerators.keys()))
graph_nx = nx.adjacency_graph(accelerator["graph"])
graph = from_networkx(graph_nx)

adder_counts: dict[int, int] = defaultdict(int)
for k in component_params:
    if "8u" in k:
        adder_counts[8] += 1
    elif "12u" in k:
        adder_counts[12] += 1
    elif "16u" in k:
        adder_counts[16] += 1

num_components = graph.adder_id.max().item() + 1


name_to_index = {k: i for i, k in enumerate(verilog_paths)}

component_sampling_data = {
    w: (
        # indices
        [name_to_index[i] for i in v],
        # weights
        [
            np.exp(-4 * num_components * component["wce%"] / 100)
            for component in v.values()
        ],
    )
    for w, v in components.items()
}

component_sampling_data = {
    k: (v[0], np.array(v[1]) / sum(v[1])) for k, v in component_sampling_data.items()
}

index_to_bitwidth = {i: k for k, v in component_sampling_data.items() for i in v[0]}

condition = graph.adder_id != -1
component_bitwidths = graph.node_bitwidth[condition][graph.adder_id[condition]].numpy()

adder_names = list(component_params.keys())

# ---------------------------------------------- Known configurations

accurate_components = {
    8: "add8u_0FP",
    12: "add12u_19A",
    16: "add16u_1E2",
}

accurate_solution = [name_to_index[accurate_components[x]] for x in component_bitwidths]

horrible_components = {
    8: "add8u_88L",
    12: "add12u_2MB",
    16: "add16u_0MH",
}

horrible_solution = [name_to_index[horrible_components[x]] for x in component_bitwidths]

initial_pop = Population(
    [
        Individual(X=accurate_solution, F=np.array([1.0, -4.0])),
        Individual(X=accurate_solution, F=np.array([4.0, -70.0])),
    ]
)

# ---------------------------------------------- Problem

problem = AutoaxProblem(
    graph, num_components, qor_embeddings, qor_model, hw_embeddings, hw_model
)

nsga2 = create_nsga2(
    args.population, component_bitwidths, component_sampling_data, args.mutation
)

# ---------------------------------------------- Evaluation

result = minimize(
    problem,
    nsga2,
    ("n_gen", args.generations),
    verbose=True,
    X=initial_pop,
)

pop_F = np.array([i.F for i in result.pop])
pop_X = np.array([i.X for i in result.pop])

with open(args.output / "_eval.tsv", "w") as f:
    for i, x in enumerate(pop_X):
        print(
            f"C{i}",
            accelerator_id,
            num_components,
            255,
            *[adder_names[j] for j in x],
            file=f,
        )

    adder_assignments = [
        [(adder_names[i], index_to_bitwidth[i]) for i in x] for x in pop_X
    ]

result_verilog = [
    generate_verilog(graph_nx, f"C{i}", assignment)
    for i, assignment in enumerate(adder_assignments)
]

with open(args.output / "dse.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, ["config", "hw", "qor"])
    writer.writeheader()
    for i, x in enumerate(pop_F):
        writer.writerow({"config": f"C{i}", "hw": x[0], "qor": -x[1]})
