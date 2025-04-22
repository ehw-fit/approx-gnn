import networkx as nx
import random
import numpy as np
import math
from pathlib import Path
import json
import re
import uuid
from collections import defaultdict
from typing import Iterable, Collection

from .convert import Module, OpWord

R_MODULE_INDEX = re.compile(r"\d+")


def _calculate_bitwidths(
    graph: nx.DiGraph, allowed_bitwidths: list[int] | None = None
) -> tuple[int, list[int]]:
    """Calculates required bit widths and writes them to graph nodes.

    :param graph: Input graph. Modified by this function.
    :param allowed_bitwidths: Allowed bitwidths to round up to.
    :return: Output width and bitwidths for each external component.
    :rtype: tuple[int, list[int]]
    """
    nodes = [x for x in nx.topological_sort(graph)]
    inputs = {n: [0, 0] for n in graph.nodes}

    module_widths: dict[int, int] = {}
    output_width = 0

    for node in nodes:
        nval = graph.nodes[node]
        node_id = OpWord(nval["node_type"])

        value = 0
        if node_id == OpWord.MODULE:
            value = sum(inputs[node])
        elif node_id == OpWord.LSHF:
            shifted = inputs[node][0]
            shift_by = inputs[node][1]
            value = int(shifted * 2**shift_by)
        elif node_id == OpWord.RSHF:
            shifted = inputs[node][0]
            shift_by = inputs[node][1]
            value = int(shifted * 2 ** (-shift_by))
        elif node_id == OpWord.CONST:
            value = nval["node_value"]
        elif node_id == OpWord.INPUT:
            value = 255
        elif node_id == OpWord.OUTPUT:
            value = inputs[node][0]
        else:
            value = max(inputs[node])

        for neigh, edge in graph[node].items():
            inputs[neigh][edge["edge_id"]] = value

        width = math.ceil(math.log2(max(value, 1)))

        if allowed_bitwidths is not None:
            try:
                width = min(
                    [x for x in allowed_bitwidths if x + 1 >= width]
                )  # based on output width
            except Exception:
                raise Exception(
                    "Allowed bitwidths can't accomodate a component in the graph."
                )

        if node_id == OpWord.MODULE:
            module_widths[nval["adder_id"]] = width
        elif node_id == OpWord.OUTPUT:
            output_width = width

        graph.nodes[node]["node_bitwidth"] = width

    module_widths_list = [x[1] for x in sorted(module_widths.items())]
    return output_width, module_widths_list


class Accelerator:
    """Accelerator instance."""

    def __init__(
        self,
        id: str,
        graph: nx.DiGraph,
        kernel: np.ndarray,
        component_bitwidths: list[int],
    ):
        """Initializer.

        :param id: Identifier.
        :param graph: Graph of this accelerator.
        :param kernel: Accelerator graphics kernel.
        :param component_bitwidths: Widths on each component of this accelerator."""
        self.id = id
        """Identifier."""
        self.graph = graph
        """Circuit graph."""
        self.kernel = kernel
        """Graphics kernel performed by the accelerator."""
        self.component_bitwidths = component_bitwidths
        """Width of each external component of this accelerator."""

    @property
    def divisor(self) -> int:
        """Kernel divisor for normalization."""
        return np.abs(self.kernel).sum()

    def serialize(self):
        """Creates a dictionary representation of this accelerator."""
        return {
            "id": self.id,
            "graph": nx.adjacency_data(self.graph),
            "kernel": [[int(x) for x in row] for row in self.kernel],
            "component_bitwidths": self.component_bitwidths,
        }

    @classmethod
    def deserialize(self, data: dict) -> "Accelerator":
        """Creates an accelerator out of a dictionary representation."""
        return Accelerator(
            data["id"],
            nx.adjacency_graph(data["graph"]),
            np.array(data["kernel"]),
            data["component_bitwidths"],
        )


def generate_random_kernel(
    n_features: int,
    maximum=20,
    kernel_size=3,
    allowed_bitwidths=[8, 12, 16],
    kernel=None,
    minimum=2,
) -> Accelerator:
    """Creates a random accelerator.

    Kernel elements get chosen up to a maximum value, which is uniformly chosen between the provided minimum and maximum.

    :param n_features: Number of node X features.
    :param maximum: Maximum of maximum values.
    :param minimum: Minimum of maximum values.
    :param kernel_size: Size of the generated kernel.
    :param allowed_bitwidths: Component widths to round up to.
    :param kernel: Kernel to implement.
    """
    if kernel is None:
        kernel_max = random.randint(minimum, maximum)
        kernel = np.random.randint(0, kernel_max, size=(kernel_size, kernel_size))
        while np.abs(kernel).sum() == 0:
            kernel = np.random.randint(0, kernel_max, size=(kernel_size, kernel_size))

    digit_groups: dict[int, list[int]] = defaultdict(list)

    module = Module(n_features, "random")

    components = []

    adder_id = 0

    def index_to_var(i):
        return f"X{i // kernel_size}{i % kernel_size}"

    for i, x in enumerate(kernel.reshape(-1)):
        if x == 0:
            continue

        module.add_node(index_to_var(i), type=OpWord.INPUT)

        if random.random() > 0.75:
            n = 0
            while (x & 1) == 0:
                if random.random() > 0.5:
                    break
                x >>= 1
                n += 1

            values = []
            for b in range(0, 6):
                if (x & (1 << b)) != 0:
                    if b != 0:
                        shift_name = f"<<i:{i}_{b}"
                        const_name = f"const:i:{i}_{b}"
                        module.add_node(shift_name, type=OpWord.LSHF)
                        module.add_node(const_name, type=OpWord.CONST, value=b)
                        module.add_edge(index_to_var(i), shift_name, edge_id=0)
                        module.add_edge(const_name, shift_name, edge_id=1)
                        values.append(shift_name)
                    else:
                        values.append(index_to_var(i))

            if len(values) > 1:
                for j in range(len(values) - 1):
                    a = random.randint(0, len(values) - 1)
                    b = random.randint(0, len(values) - 1)
                    while a == b:
                        b = random.randint(0, len(values) - 1)

                    added_node = f"add:i:{i}_{j}"
                    module.add_node(added_node, type=OpWord.MODULE, adder_id=adder_id)
                    adder_id += 1
                    module.add_edge(values[a], added_node, edge_id=0)
                    module.add_edge(values[b], added_node, edge_id=1)
                    values[min(a, b)] = added_node
                    values[max(a, b)] = values[-1]
                    values = values[:-1]
            if n != 0:
                shift_name = f"<<n:{i}_{n}"
                const_name = f"const_n:{i}_{n}"
                module.add_node(shift_name, type=OpWord.LSHF)
                module.add_node(const_name, type=OpWord.CONST, value=n)
                module.add_edge(values[0], shift_name, edge_id=0)
                module.add_edge(const_name, shift_name, edge_id=1)
                components.append(shift_name)
            else:
                components.append(values[0])
        else:
            for b in range(0, 16):
                if (x & (1 << b)) != 0:
                    digit_groups[b].append(i)

    for shift, indices in digit_groups.items():
        values = [index_to_var(i) for i in indices]
        if len(values) > 1:
            for i in range(len(values) - 1):
                a = random.randint(0, len(values) - 1)
                b = random.randint(0, len(values) - 1)
                while a == b:
                    b = random.randint(0, len(values) - 1)

                added_node = f"+c:{shift}{i}"
                module.add_node(added_node, type=OpWord.MODULE, adder_id=adder_id)
                adder_id += 1
                module.add_edge(values[a], added_node, edge_id=0)
                module.add_edge(values[b], added_node, edge_id=1)
                values[min(a, b)] = added_node
                values[max(a, b)] = values[-1]
                values = values[:-1]

        if shift != 0:
            shift_name = f"<<c:{shift}"
            const_name = f"const_c:{shift}"
            module.add_node(shift_name, type=OpWord.LSHF)
            module.add_node(const_name, type=OpWord.CONST, value=shift)
            module.add_edge(values[0], shift_name, edge_id=0)
            module.add_edge(const_name, shift_name, edge_id=1)
            components.append(shift_name)
        else:
            components.append(values[0])

    module.add_node("Y", type=OpWord.OUTPUT)

    if len(components) > 1:
        for i in range(len(components) - 1):
            a = random.randint(0, len(components) - 1)
            b = random.randint(0, len(components) - 1)
            while a == b:
                b = random.randint(0, len(components) - 1)

            added_node = f"+comp:{i}"
            module.add_node(added_node, type=OpWord.MODULE, adder_id=adder_id)
            adder_id += 1
            module.add_edge(components[a], added_node, edge_id=0)
            module.add_edge(components[b], added_node, edge_id=1)
            components[min(a, b)] = added_node
            components[max(a, b)] = components[-1]
            components = components[:-1]

    module.add_edge(components[0], "Y", edge_id=0)

    id = f"W{str(uuid.uuid4())[:8]}"
    _, widths = _calculate_bitwidths(module.graph, allowed_bitwidths)

    return Accelerator(id, module.graph, kernel, widths)


def generate_accelerator_code(accelerators: Iterable[Accelerator]) -> str:
    """Generates C code implementing the given accelerators."""
    ret_signatures: list[str] = []

    header = """#include \"common.hpp\"\n\n"""

    for accelerator in accelerators:
        graph = accelerator.graph
        id = accelerator.id

        nodes: list[str] = [x for x in nx.topological_sort(graph)]
        inputs: dict[str, list[int | str]] = {n: [0, 0] for n in graph.nodes}

        signature = f"""int {id}(int X00,int X01,int X02,int X10,int X11,int X12,int X20,int X21,int X22,
            tAdd adders[])\n"""

        for node in nodes:
            nval = graph.nodes[node]
            node_id = OpWord(nval["node_type"])

            if node_id == OpWord.MODULE:
                module_id = nval["adder_id"]
                value: str | int = (
                    f"adders[{module_id}]({inputs[node][0]},{inputs[node][1]})"
                )
            elif node_id == OpWord.LSHF:
                value = f"({inputs[node][0]}) << {inputs[node][1]}"
            elif node_id == OpWord.RSHF:
                value = f"({inputs[node][0]}) >> {inputs[node][1]}"
            elif node_id == OpWord.CONST:
                value = int(nval["node_value"])
            elif node_id == OpWord.INPUT:
                value = node  # name
            elif node_id == OpWord.XOR:
                value = f"({inputs[node][0]}) ^ ({inputs[node][1]})"
            elif node_id == OpWord.NXOR:
                value = f"~(({inputs[node][0]}) ^ ({inputs[node][1]}))"
            elif node_id == OpWord.AND:
                value = f"({inputs[node][0]}) & ({inputs[node][1]})"
            elif node_id == OpWord.NAND:
                value = f"~(({inputs[node][0]}) & ({inputs[node][1]}))"
            elif node_id == OpWord.OR:
                value = f"({inputs[node][0]}) | ({inputs[node][1]})"
            elif node_id == OpWord.NOR:
                value = f"~(({inputs[node][0]}) | ({inputs[node][1]}))"
            elif node_id == OpWord.NEG:
                value = f"~({inputs[node][0]})"
            elif node_id == OpWord.OUTPUT:
                ret_signatures.append(
                    "".join([signature, "{", f"return {inputs[node][0]};", "}"])
                )
                break
            else:
                value = inputs[node][0]

            for neigh, edge in graph[node].items():
                inputs[neigh][edge["edge_id"]] = value

    return header + "\n\n".join(ret_signatures)


def generate_mappings(adder_ids: Iterable[str], wirings_ids: Iterable[str]):
    """Generates map assignment C code for use in reference evaluation."""
    header = """#include \"configs.hpp\"\n\nvoid Configs::load() {\n"""

    bodyA = "".join([f'\tadders["{i}"] = {i};\n' for i in adder_ids])
    bodyW = "".join([f'\twirings["{i}"] = {i};\n' for i in wirings_ids])

    footer = "}"

    return f"{header}{bodyA}{bodyW}{footer}"


R_INCLUDE = re.compile(r"^#include.+", re.MULTILINE)
R_COMMENT = re.compile(r"//.*$", re.MULTILINE)
R_MULTILINE_COMMENT = re.compile(r"/\*.+?\*/", re.MULTILINE | re.DOTALL)
R_SIGNATURE = re.compile(
    r"^\w+\s+(\w+)\s*\(\s*(?:const\s*)?\w+\s+(\w+)\s*,\s*(?:const\s*)?\w+\s+(\w+)\s*\)",
    re.MULTILINE,
)


def get_components_and_code(
    required_widths: Iterable[int],
    meta_path: str | Path,
    component_params: dict[str, dict[str, float]],
    pareto_target: str = "mse",
):
    """Loads components and combines them into C code.

    :param required_widths: Widths to load.
    :param meta_path: EvoApproxLib JSON file.
    :param component_params: Component parameters.
    :param pareto_target: EvoApproxLib target.
    :return: Dictionary of widths, adders and parameters.
    :rtype: tuple[dict[int, dict[str, dict[str, float]]], str]
    """
    meta_path = Path(meta_path)

    try:
        with open(meta_path) as f:
            meta = json.load(f)
        source_path: Path = meta_path.parent
    except Exception:
        raise Exception("Failed to open meta file:", meta_path)

    target_adders = [f"{width}_unsigned" for width in required_widths]

    return_c_code = []
    adders_of_width: dict[int, dict[str, dict[str, float]]] = defaultdict(dict)

    for group in meta:
        for dataset in group["datasets"]:
            if dataset["folder"].split("/")[-1] in target_adders:
                adders_verilog = dict()
                bitwidth = dataset["bitwidth"]

                for subdataset in dataset["datasets"]:
                    if (
                        subdataset["folder"].split("/")[-1]
                        != f"pareto_pwr_{pareto_target.lower()}"
                    ):
                        continue

                    for instance in subdataset["instances"]:
                        name = instance["name"]
                        verilog_path = source_path / subdataset["folder"] / f"{name}.v"
                        adders_verilog[name] = verilog_path.read_text()

                        c_path = source_path / subdataset["folder"] / f"{name}.c"
                        
                        with open(c_path, "r") as f:
                            cfile = f.read()

                        # strip and adjust
                        cfile = R_INCLUDE.sub("", cfile)
                        cfile = R_COMMENT.sub("", cfile)
                        cfile = R_MULTILINE_COMMENT.sub("", cfile)

                        # unify inconsistent signature types
                        cfile = R_SIGNATURE.sub(
                            r"int \g<1>(int \g<2>, int \g<3>)", cfile
                        )
                        cfile = cfile.replace("\n\n", "")
                        return_c_code.append(cfile)

                        adders_of_width[bitwidth][name] = component_params[name]

                        params = {k: float(v) for k, v in instance["params"].items()}
                        params["cfun"] = name
                        params["verilog"] = name + ".v"
                        params["verilog_entity"] = name

    flattened = "\n".join(return_c_code)
    return dict(adders_of_width), f'#include "common.hpp"\n{flattened}'


def generate_configs(
    accelerator: Accelerator,
    components: dict[int, dict[str, dict[str, float]]],
    count: int = 10,
    weight_exponent: float = 4,
) -> list[str]:
    """Generate configurations of a given accelerator.

    :param accelerator: Accelerator to instantiate.
    :param components: Component parameters by bitwidth.
    :param count: Number of configurations to generate.
    :param weight_exponent: Weight exponent for components based on their accuracy. Used to improve accuracy variation for large graphs.
    :return: Configuration list for a TSV file.
    :rtype: list[str]
    """
    component_count = len(accelerator.component_bitwidths)
    divisor = accelerator.divisor

    ret: list[str] = []
    for _ in range(count):
        comps = [
            random.choices(
                list(components[w]),
                [
                    np.exp(-weight_exponent * component_count * component["wce%"] / 100)
                    for component in components[w].values()
                ],
                k=1,
            )[0]
            for w in accelerator.component_bitwidths
        ]

        id = f"C{str(uuid.uuid4())[:8]}"
        comps_str = " ".join(comps)
        ret.append(f"{id} {accelerator.id} {component_count} {divisor} {comps_str}")
    return ret


_R_NAME_CLEANUP = re.compile(r"[+:]")


def generate_verilog(
    graph: nx.DiGraph, id: str, adder_assignments: list[tuple[str, int]]
):
    """Generates Verilog file from a circuit graph.

    :param graph: Graph to use.
    :param id: Resulting module name.
    :param adder_assignments: Tuples of component names and widths.
    :return: Verilog code.
    :rtype: str
    """
    nodes = list(nx.topological_sort(graph))

    w_inputs: dict[str, dict[int, int]] = {n: defaultdict(int) for n in nodes}
    widths: dict[str, int] = {n: 0 for n in nodes}
    wires: dict[int, list[str]] = defaultdict(list)

    for node in nodes:
        nval = graph.nodes[node]
        node_id = OpWord(nval["node_type"])

        if node_id == OpWord.MODULE:
            value: str | int = 0
            width = adder_assignments[nval["adder_id"]][1] + 1
        elif node_id == OpWord.LSHF:
            value = widths[node] + w_inputs[node][1]
            width = value
        elif node_id == OpWord.RSHF:
            value = widths[node] - w_inputs[node][1]
            width = value
        elif node_id == OpWord.CONST:
            value = int(nval["node_value"])
            width = 1
        elif node_id == OpWord.INPUT:
            value = 0
            width = 8
        elif node_id == OpWord.ID:
            value = w_inputs[node][0]
            width = widths[node]
            wires[width].append(node)
        else:
            value = w_inputs[node][0]
            width = widths[node]

        widths[node] = width

        for neigh, edge in graph[node].items():
            w_inputs[neigh][edge["edge_id"]] = value
            widths[neigh] = max(widths[neigh], width)

    inputs: dict[str, dict[int, int | str]] = {n: defaultdict(str) for n in nodes}
    signature = f"module {id}(X00, X01, X02, X10, X11, X12, X20, X21, X22, Y);"
    input_wires = "  input [7:0] X00, X01, X02, X10, X11, X12, X20, X21, X22;"
    output_wires = f"  output [{widths['Y'] - 1}:0] Y;"
    statements = []
    wire_id = 0

    adder_signature = ["A", "B", "O"]

    for node in nodes:
        nval = graph.nodes[node]
        node_id = OpWord(nval["node_type"])

        if node_id == OpWord.ID:
            if inputs[node][0] is not None:
                statements.append(f"assign {node} = {inputs[node][0]};")
            value = str(node)
        elif node_id == OpWord.MODULE:
            adder = adder_assignments[nval["adder_id"]]
            module_id = _R_NAME_CLEANUP.sub("", node)
            module_inputs = []
            sorted_inputs = [x[1] for x in sorted(inputs[node].items())]

            for i, (input, label) in enumerate(zip(sorted_inputs, adder_signature)):
                input_wire = f"w{wire_id}_{i}"
                # width is set to output width
                wires[widths[node] - 1].append(input_wire)
                module_inputs.append(f".{label}({input_wire})")
                statements.append(f"  assign {input_wire} = {input};")

            output = f"w{wire_id}"
            wires[widths[node]].append(output)
            module_inputs.append(f".{adder_signature[-1]}({output})")
            wire_id += 1

            module_type = adder[0]
            statements.append(
                f"  {module_type} {module_id}({', '.join(module_inputs)});"
            )
            value = output
        elif node_id == OpWord.LSHF:
            #  {A, 1'b0};
            shift = inputs[node][1]
            value = f"({inputs[node][0]}) << {shift}"
            value = "{" + value + f", {shift}'b0" + "}"
        elif node_id == OpWord.RSHF:
            value = f"({inputs[node][0]}) >> {inputs[node][1]}"
        elif node_id == OpWord.CONST:
            value = int(nval["node_value"])
        elif node_id == OpWord.INPUT:
            value = node  # name
        elif node_id == OpWord.XOR:
            value = f"({inputs[node][0]}) ^ ({inputs[node][1]})"
        elif node_id == OpWord.NXOR:
            value = f"~(({inputs[node][0]}) ^ ({inputs[node][1]}))"
        elif node_id == OpWord.AND:
            value = f"({inputs[node][0]}) & ({inputs[node][1]})"
        elif node_id == OpWord.NAND:
            value = f"~(({inputs[node][0]}) & ({inputs[node][1]}))"
        elif node_id == OpWord.OR:
            value = f"({inputs[node][0]}) | ({inputs[node][1]})"
        elif node_id == OpWord.NOR:
            value = f"~(({inputs[node][0]}) | ({inputs[node][1]}))"
        elif node_id == OpWord.NEG:
            value = f"~({inputs[node][0]})"
        elif node_id == OpWord.OUTPUT:
            if inputs[node][0] != "Y":
                statements.append(f"  assign Y = {inputs[node][0]};")
            break
        else:
            value = inputs[node][0]

        for neigh, edge in graph[node].items():
            inputs[neigh][edge["edge_id"]] = value

    combined = "\n".join(statements)
    combined_wires = "\n".join(
        [f"  wire [{k-1}:0] {', '.join(v)};" for k, v in wires.items()]
    )
    return f"{signature}\n{input_wires}\n{output_wires}\n{combined_wires}\n\n{combined}\nendmodule"
