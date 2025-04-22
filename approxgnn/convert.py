import sys
from .VerilogLexer import VerilogLexer
from .VerilogParser import VerilogParser
from .VerilogParserVisitor import VerilogParserVisitor
from antlr4 import FileStream, CommonTokenStream, InputStream
import networkx as nx
import torch_geometric.data
from torch_geometric.utils import from_networkx
from collections import defaultdict
from pathlib import Path
import re
from typing import Iterable
from enum import Enum

MODULE_ATTRIBUTES = [
    "ep%",
    "mae%",
    "mre%",
    "wce%",
    "wcre%",
]

R_CONST = re.compile(r"-?(\d+)?'([hHbBoOdD])(\d+)")


class OpWord(Enum):
    """Enum of node types."""

    NULL = 0
    ID = 1
    INPUT = 2
    OUTPUT = 3
    CONST = 4
    ASSIGN = 5
    CONCAT = 6
    MODULE = 7
    TO_BIT = 8
    FROM_BIT = 9
    ADD = 10
    SUB = 11
    MUL = 12
    DIV = 13
    MOD = 14
    EXP = 15
    NAND = 16
    AND = 17
    LAND = 18
    LOR = 19
    OR = 20
    NOR = 21
    NOT = 22
    NEG = 23
    XOR = 24
    NXOR = 25
    LT = 26
    LE = 27
    EQ = 28
    EQS = 29
    NES = 30
    NE = 31
    GE = 32
    GT = 33
    LSHF = 34
    LSHFA = 35
    RSHFA = 36
    RSHF = 37
    TERNARY = 38
    RANGE = 39


class ModuleInstance:
    """Internal class. Represents a known module, that's yet to be inserted."""

    def __init__(self, name: str, parent, connections: list[str] | dict[str, str]):
        """Initializer.

        :param name: Module name.
        :param parent: Parent Module.
        :param connections: List of positional inputs or dictionary of named inputs.
        """
        self.name = name
        self.parent = parent
        self.connections = connections


class Module:
    """Parsed Verilog representation."""

    def __init__(self, n_features: int, name: str):
        """Initializer.

        :param n_features: Number of node X features.
        :param name: Top-level module name.
        """
        self.graph = nx.DiGraph(y=0.0)
        """Verilog node graph."""
        self.n_features = n_features
        """Number of node X features."""
        self.inputs: list[str] = []
        """Module input ports."""
        self.outputs: list[str] = []
        """Module output ports."""
        self.ports: list[str] = []
        """All module ports."""
        self.unresolved_instances: list[ModuleInstance] = []
        """Known modules yet to be inserted."""
        self.external_modules: list[str] = []
        """Unknown modules, assumed to be external components."""
        self.name = name
        """Top-level module name."""

    def add_port(self, name: str, input=False, output=False):
        """Adds a port.

        :param name: Port name.
        :param input: Is an input port. Exclusive.
        :param output: Is an output port. Exclusive.
        """
        if input:
            self.inputs.append(name)
        elif output:
            self.outputs.append(name)
        self.ports.append(name)

    def add_external(self, name: str):
        """Adds an external module name.

        :param name: Module name.
        """
        self.external_modules.append(name)

    def add_instance(self, instance: ModuleInstance):
        """Adds a known instance to insert later.

        :param instance: Module instance.
        """
        self.unresolved_instances.append(instance)

    def add_node(
        self,
        node: str,
        type: OpWord = OpWord.NULL,
        label: str = "",
        value: float = 0.0,
        adder_id: int = -1,
    ):
        """Adds a node to the graph.

        :param node: Node name to add.
        :param type: Node type.
        :param label: Node label for printing.
        :param value: Node value.
        :param adder_id: Component index, if the node is a module.
        """
        if node not in self.graph.nodes:
            self.graph.add_node(
                node,
                node_label=label,
                node_type=type.value,
                node_value=value,
                adder_id=adder_id,
                node_bitwidth=0,
                x=[0.0 for _ in range(self.n_features)],
            )

    def add_edge(self, A, B, edge_id: int = 0):
        """Adds an edge to the graph.

        :param A: Starting node.
        :param B: Ending node.
        :param edge_id: Edge index.
        """
        self.graph.add_edge(
            A,
            B,
            edge_id=edge_id,
        )

    def resolve_instances(self):
        """Inserts unresoled ModuleInstances."""
        for mod in self.unresolved_instances:
            to_add: Module = mod.parent
            ports = set(to_add.ports)

            if type(mod.connections) is list:
                connections = {
                    to_add.ports[i]: e for i, e in enumerate(mod.connections)
                }
            else:
                connections = mod.connections

            # TODO: turn connection list into dict
            def replace_connection(a: str, b: str, v):
                if a in connections:
                    a = connections[a]
                else:
                    a = f"{mod.name}_{a}"

                if b in connections:
                    b = connections[b]
                else:
                    b = f"{mod.name}_{b}"

                return (a, b, v)

            self.graph.add_nodes_from(
                [
                    (f"{mod.name}_{k}", v)
                    for k, v in to_add.graph.nodes.data()
                    if k not in ports
                ]
            )

            self.graph.add_edges_from(
                [replace_connection(a, b, v) for a, b, v in to_add.graph.edges.data()]
            )

    def type_correction(self):
        """Fixes node types for inputs and outputs."""
        for input in self.inputs:
            if input in self.graph.nodes:
                self.graph.nodes[input]["node_type"] = OpWord.INPUT.value

        for output in self.outputs:
            if output in self.graph.nodes:
                self.graph.nodes[output]["node_type"] = OpWord.OUTPUT.value


# Parsing -------------------------------------------------------------------------------


class Visitor(VerilogParserVisitor):
    """Verilog parser visitor."""

    def __init__(
        self,
        n_features: int,
        includes: dict[str, Module] | None = None,
        external_adder_signatures: (
            Iterable[str | int] | dict[str, Iterable[str | int]] | None
        ) = None,
    ):
        """Initializer.

        :param n_features: Number of node X features.
        :param includes: External module mappings for included Verilog files.
        :param external_adder_signatures: External adder output port names or indices common for all components or for individual components.
        """

        if n_features < 0:
            raise ValueError("Number of features has to be >= 0.")

        if includes is not None:
            self.modules: dict[str, Module] = {k: v for k, v in includes.items()}
            """Modules found in Verilog file."""
        else:
            self.modules = {}

        self._active_graph: Module | None = None
        """Currently processed module."""
        self.n_features = n_features
        """Number of node X features."""

        if external_adder_signatures is not None:
            if isinstance(external_adder_signatures, dict):
                self.adder_signatures: dict[str, Iterable[str | int]] = (
                    external_adder_signatures
                )
                """External adder port names or indices per component."""
                self.common_adder_outputs: Iterable[str | int] = []
                """External adder port names or indices for all components."""
            else:
                self.adder_signatures = dict()
                self.common_adder_outputs = external_adder_signatures
        else:
            self.adder_signatures = dict()
            self.common_adder_outputs = [2, "Y", "O", "YS", "YC"]

        self.op_name_map = {
            "+": OpWord.ADD,
            "-": OpWord.SUB,
            "*": OpWord.MUL,
            "/": OpWord.DIV,
            "%": OpWord.MOD,
            "**": OpWord.EXP,
            "~&": OpWord.NAND,
            "&": OpWord.AND,
            "&&": OpWord.LAND,
            "||": OpWord.LOR,
            "|": OpWord.OR,
            "~|": OpWord.NOR,
            "!": OpWord.NOT,
            "~": OpWord.NEG,
            "^": OpWord.XOR,
            "^~": OpWord.NXOR,
            "~^": OpWord.NXOR,
            "<": OpWord.LT,
            "<=": OpWord.LE,
            "==": OpWord.EQ,
            "===": OpWord.EQS,
            "!==": OpWord.NES,
            "!=": OpWord.NE,
            ">=": OpWord.GE,
            ">": OpWord.GT,
            "<<": OpWord.LSHF,
            "<<<": OpWord.LSHFA,
            ">>>": OpWord.RSHFA,
            ">>": OpWord.RSHF,
        }
        """Opeartor to Enum map."""

    def clean_module_name(self, name):
        """Fixes module name for autoax compatibility."""
        # autoax compatibility
        return name.replace('"', "")

    def create_label(self, ctx, name):
        """Creates a unique label based on a position in the parsing context."""
        interval = ctx.getSourceInterval()
        return f"{interval[0]}_{interval[1]}_{name}"

    # Modules and ports -----------------------------------------------------------------

    def visitSource_text(self, ctx):
        """Creates initial empty graphs for each module and visits them."""
        for desc in ctx.description():
            if desc.module_declaration() is not None:
                mod_name: str = desc.module_declaration().module_identifier().getText()
                mod_name = self.clean_module_name(mod_name)
                self.modules[mod_name] = Module(self.n_features, mod_name)

        for desc in ctx.description():
            self.visit(desc)

    def visitModule_declaration(self, ctx):
        """Processes module ports and contents."""
        module_id = self.clean_module_name(ctx.module_identifier().getText())
        self._active_graph = self.modules[module_id]

        if ctx.list_of_port_declarations() is not None:
            self.visit(ctx.list_of_port_declarations())

        for line in ctx.module_item():
            self.visit(line)

    def visitList_of_port_declarations(self, ctx):
        if len(ctx.port_declaration()) > 0:
            for decl in ctx.port_declaration():
                self.visit(decl)
        elif ctx.port_implicit() is not None:
            raise Exception("Implicit ports not supported.")

        # Other ports are resolved later

    def visitPort_declaration(self, ctx):
        if ctx.inout_declaration() is not None:
            raise Exception("InOut ports not supported.")
        elif ctx.input_declaration() is not None:
            ports = self.visit(ctx.input_declaration())
            for k in ports:
                self._active_graph.add_port(k, input=True, output=False)
        elif ctx.output_declaration() is not None:
            ports = self.visit(ctx.output_declaration())
            for k in ports:
                self._active_graph.add_port(k, input=False, output=True)

    def visitInput_declaration(self, ctx):
        return self.visit(ctx.list_of_port_identifiers())

    def visitOutput_declaration(self, ctx):
        if ctx.list_of_port_identifiers() is not None:
            return self.visit(ctx.list_of_port_identifiers())
        elif ctx.list_of_variable_port_identifiers() is not None:
            self.visit(ctx.list_of_variable_port_identifiers())

    def visitList_of_port_identifiers(self, ctx):
        return [x.getText() for x in ctx.port_identifier()]

    def visitList_of_variable_port_identifiers(self, ctx):
        return [x.port_identifier().getText() for x in ctx.var_port_id()]
        # = constant_expression

    # Assignments -----------------------------------------------------------------------

    def visitNet_assignment(self, ctx):
        lvalue = self.visit(ctx.net_lvalue())
        ret = self.visitChildren(ctx)
        name = lvalue["name"]

        self._active_graph.add_node(name, label=name, type=OpWord.ID)

        if "select" in lvalue:
            for i, range_node in enumerate(lvalue["select"]):
                assign_label = self.create_label(ctx.net_lvalue(), f"to_{i}")
                self._active_graph.add_node(assign_label, OpWord.TO_BIT, assign_label)

                # selection
                self._active_graph.add_edge(ret, assign_label, edge_id=0)
                self._active_graph.add_edge(range_node, assign_label, edge_id=1)

                self._active_graph.add_edge(assign_label, name)
        else:
            self._active_graph.add_edge(ret, name)
        return lvalue

    def visitNet_lvalue(self, ctx):
        """Assignment L-value.

        :return: Dictionary containing name and optionally bit selection.
        :rtype: dict[str, str | Any]"""
        name = ctx.hierarchical_identifier().getText()
        if ctx.const_select() is not None:
            selection = self.visit(ctx.const_select())
            return {"name": name, "select": selection}
        else:
            return {"name": name}

    def visitConst_select(self, ctx):
        """L-value bit selection.

        :return: List of selected bits.
        :rtype: list"""
        if ctx.const_bit_select() is not None:
            selections = self.visit(ctx.const_bit_select())
        else:
            selections = []

        selections.append(self.visit(ctx.constant_range_expression()))
        return selections

    def visitSelect_(self, ctx):
        """R-value bit selection.

        Only a single selection is supported.

        :return: List of selected bits (of length 1).
        :rtype: list"""
        if ctx.bit_select() is not None:
            selections = self.visit(ctx.bit_select())
        else:
            selections = []

        selections.append(self.visit(ctx.range_expression()))

        # TODO:
        if len(selections) > 1:
            raise Exception("Only one bit-select is supported on R-values.")
        return selections

    def visitConst_bit_select(self, ctx):
        """L-value bit selection.

        :return: List of integer bit selections.
        :rtype: list[int]
        """
        try:
            return [int(expr.getText()) for expr in ctx.constant_expression()]
        except ValueError:
            raise Exception("Only integer bit-selection is supported.")

    def visitBit_select(self, ctx):
        """R-value bit selection.

        :return: List of integer bit selections.
        :rtype: list[int]
        """
        try:
            return [int(expr.getText()) for expr in ctx.expression()]
        except ValueError:
            raise Exception("Only integer bit-selection is supported.")

    def visitConstant_range_expression(self, ctx):
        """L-value bit range. Only integer selection is supported.

        :return: Selection node.
        :rtype: str"""
        try:
            if ctx.constant_expression() is not None:  # expression
                at = int(ctx.getText())
                label = self.create_label(ctx, "at")
                self._active_graph.add_node(
                    label, label=label, type=OpWord.CONST, value=at
                )
                return label
            else:
                if ctx.CL() is not None:  # range
                    a = int(ctx.msb_constant_expression().getText())
                    b = int(ctx.lsb_constant_expression().getText())
                    label_a = self.create_label(
                        ctx.msb_constant_expression(), "range_A"
                    )
                    label_b = self.create_label(
                        ctx.lsb_constant_expression(), "range_B"
                    )
                elif ctx.PLCL() is not None:  # positive width
                    b = int(self.visit(ctx.constant_base_expression()))
                    a = b + int(self.visit(ctx.width_constant_expression()))
                    label_a = self.create_label(
                        ctx.constant_base_expression(), "range_A"
                    )
                    label_b = self.create_label(
                        ctx.width_constant_expression(), "range_B"
                    )
                elif ctx.MICL() is not None:  # negative width
                    a = int(self.visit(ctx.constant_base_expression()))
                    b = a - int(self.visit(ctx.width_constant_expression()))
                    label_a = self.create_label(
                        ctx.constant_base_expression(), "range_A"
                    )
                    label_b = self.create_label(
                        ctx.width_constant_expression(), "range_B"
                    )
                else:
                    raise Exception("Grammar error in range expression.")

                label_range = self.create_label(ctx, "range")
                self._active_graph.add_node(
                    label_range, label=label_range, type=OpWord.RANGE
                )
                self._active_graph.add_node(
                    label_a, label=label_a, type=OpWord.CONST, value=a
                )
                self._active_graph.add_node(
                    label_b, label=label_b, type=OpWord.CONST, value=b
                )

                self._active_graph.add_edge(label_a, label_range, edge_id=0)
                self._active_graph.add_edge(label_b, label_range, edge_id=1)

                return label_range

        except ValueError:
            raise Exception("Only integer bit and range selection is supported.")

    def visitRange_expression(self, ctx):
        """R-value bit range. Only integer selection is supported."""
        try:
            if ctx.expression() is not None:  # expression
                at = int(ctx.getText())
                label = self.create_label(ctx, "at")
                self._active_graph.add_node(
                    label, label=label, type=OpWord.CONST, value=at
                )
                return label
            else:
                if ctx.CL() is not None:  # range
                    a = int(ctx.msb_constant_expression().getText())
                    b = int(ctx.lsb_constant_expression().getText())
                    label_a = self.create_label(
                        ctx.msb_constant_expression(), "range_A"
                    )
                    label_b = self.create_label(
                        ctx.lsb_constant_expression(), "range_B"
                    )
                elif ctx.PLCL() is not None:  # positive width
                    b = int(self.visit(ctx.base_expression()))
                    a = b + int(self.visit(ctx.width_constant_expression()))
                    label_a = self.create_label(ctx.base_expression(), "range_A")
                    label_b = self.create_label(
                        ctx.width_constant_expression(), "range_B"
                    )
                elif ctx.MICL() is not None:  # negative width
                    a = int(self.visit(ctx.base_expression()))
                    b = a - int(self.visit(ctx.width_constant_expression()))
                    label_a = self.create_label(ctx.base_expression(), "range_A")
                    label_b = self.create_label(
                        ctx.width_constant_expression(), "range_B"
                    )
                else:
                    raise Exception("Grammar error in range expression.")

                label_range = self.create_label(ctx, "range")
                self._active_graph.add_node(
                    label_range, label=label_range, type=OpWord.RANGE
                )
                self._active_graph.add_node(
                    label_a, label=label_a, type=OpWord.CONST, value=a
                )
                self._active_graph.add_node(
                    label_b, label=label_b, type=OpWord.CONST, value=b
                )

                self._active_graph.add_edge(label_a, label_range, edge_id=0)
                self._active_graph.add_edge(label_b, label_range, edge_id=1)

                return label_range

        except ValueError:
            raise Exception("Only integer bit and range selection is supported.")

    # Expressions -----------------------------------------------------------------------

    def visitUnaryExpression(self, ctx):
        """All unary expressions."""
        operator = ctx.unary_operator().getText()
        label = self.create_label(ctx, self.op_name_map[operator].name)
        self._active_graph.add_node(label, label=label, type=self.op_name_map[operator])

        primary = self.visit(ctx.primary())
        self._active_graph.add_edge(primary, label)
        return label

    def visitPrimaryExpression(self, ctx):
        """Constants, IDs, concatenations, and calls."""
        return self.visit(ctx.primary())

    def visitBinaryExpression(self, ctx):
        """All binary expressions."""
        exprs = ctx.expression()
        op = ctx.children[1].getText()
        label = self.create_label(ctx, self.op_name_map[op])
        self._active_graph.add_node(label, label=label, type=self.op_name_map[op])

        self._active_graph.add_edge(self.visit(exprs[0]), label, edge_id=0)
        self._active_graph.add_edge(self.visit(exprs[1]), label, edge_id=1)
        return label

    def visitTernaryExpression(self, ctx):
        exprs = ctx.expression()

        label = self.create_label(ctx, "ternary")
        self._active_graph.add_node(label, label=label, type=OpWord.TERNARY)

        self._active_graph.add_edge(self.visit(exprs[0]), label, edge_id=0)
        self._active_graph.add_edge(self.visit(exprs[1]), label, edge_id=1)
        self._active_graph.add_edge(self.visit(exprs[2]), label, edge_id=2)
        return label

    def visitConstPrimary(self, ctx):
        """Constant values."""
        val_text = ctx.number().getText()
        if m := R_CONST.match(val_text):
            # width = int(m.group(1))
            base = m.group(2).lower()
            if base == "h":
                base = 16
            elif base == "b":
                base = 2
            elif base == "o":
                base = 8
            else:
                base = 10
            val = int(m.group(3), base=base)
        else:
            val = int(val_text)

        label = self.create_label(ctx, f"{val}_const")
        self._active_graph.add_node(label, label=label, type=OpWord.CONST, value=val)
        return label

    def visitConcatPrimary(self, ctx):
        """Concatenation expression."""
        label = self.create_label(ctx, "concat")
        self._active_graph.add_node(label, label=label, type=OpWord.CONCAT)

        for i, expr in enumerate(ctx.concatenation().expression()):
            self._active_graph.add_edge(self.visit(expr), label, edge_id=i)
        return label

    def visitIdPrimary(self, ctx):
        """Wire, input or output elements."""
        ret = ctx.hierarchical_identifier().getText()
        self._active_graph.add_node(ret, label=ret, type=OpWord.ID)

        if ctx.select_():
            select = self.visit(ctx.select_())
            for i, select_id in enumerate(select):
                assign_label = self.create_label(ctx.select_(), f"from_{i}")
                self._active_graph.add_node(assign_label, OpWord.FROM_BIT, assign_label)
                self._active_graph.add_edge(ret, assign_label, edge_id=0)
                self._active_graph.add_edge(select_id, assign_label, edge_id=1)

                # TODO: multiple selections
                break
            return assign_label
        else:
            return ret

    def visitMintypmaxPrimary(self, ctx):
        """Parentheses."""
        return self.visit(ctx.mintypmax_expression())

    def visitMintypmax_expression(self, ctx):
        """Parentheses. Only selects first element."""
        return self.visit(ctx.expression()[0])

    def visitCallPrimary(self, ctx):
        return super().visitCallPrimary(ctx)

    def visitMultConcatPrimary(self, ctx):
        return super().visitMultConcatPrimary(ctx)

    def visitSysCallPrimary(self, ctx):
        return super().visitSysCallPrimary(ctx)

    # Instantiation ---------------------------------------------------------------------

    def visitModule_instantiation(self, ctx):
        """Creates either a module node or instance."""
        module_type: str = ctx.module_identifier().getText()
        # autoax compatibility
        module_type = self.clean_module_name(module_type)

        for _instance in ctx.module_instance():
            instance: VerilogParser.Module_instanceContext = _instance
            instance_name = instance.name_of_module_instance().getText()

            ports = self.visitList_of_port_connections(
                instance.list_of_port_connections()
            )

            if module_type in self.modules:
                # Known modules get inserted afterwards
                self._active_graph.add_instance(
                    ModuleInstance(instance_name, self.modules[module_type], ports)
                )
            else:
                # Unknown modules become nodes
                label = self.create_label(instance, f"module_{instance_name}")

                self._active_graph.add_node(
                    label,
                    label=module_type,
                    type=OpWord.MODULE,
                    adder_id=len(self._active_graph.external_modules),
                )

                self._active_graph.add_external(module_type)

                if module_type in self.adder_signatures:
                    possible_ports = self.adder_signatures[module_type]
                else:
                    possible_ports = self.common_adder_outputs

                # TODO: support multiple output ports
                if type(ports) is dict:
                    out_port_idx = next((i for i in possible_ports if i in ports), None)
                    if out_port_idx is None:
                        raise Exception(
                            f"External module doesn't have known output ports: {', '.join(possible_ports)} [{module_type}, {instance_name}, {self._active_graph.name}]"
                        )
                    output_port = ports[out_port_idx]
                else:
                    out_port_idx = next(
                        (
                            i
                            for i in possible_ports
                            if type(i) is int and i < len(ports)
                        ),
                        None,
                    )
                    if out_port_idx is None:
                        raise Exception(
                            f"External module doesn't have known output ports: {', '.join(possible_ports)} [{module_type}, {instance_name}, {self._active_graph.name}]"
                        )
                    output_port = ports[out_port_idx]

                self._active_graph.add_node(
                    output_port, label=output_port, type=OpWord.ID
                )
                self._active_graph.add_edge(label, output_port)

                index = 0
                if type(ports) is dict:
                    for i, p in ports.items():
                        if i == out_port_idx:  # skip output
                            continue
                        self._active_graph.add_edge(p, label, edge_id=index)
                        index += 1
                else:
                    for i, p in enumerate(ports):
                        if i == out_port_idx:  # skip output
                            continue
                        self._active_graph.add_edge(p, label, edge_id=index)
                        index += 1

    def visitList_of_port_connections(self, ctx):
        """Port list.

        :return: List of unnamed ports or dictionary of named ports and expressions.
        :rtype: `list[tuple[str, Any]] | dict[str, tuple[str, Any]]`"""
        if ctx.ordered_port_connection():
            # Expression in ordered_port_connection gets returned
            return [self.visit(x) for x in ctx.ordered_port_connection()]
        else:
            ports = [self.visit(x) for x in ctx.named_port_connection()]
            return {k: v for k, v in ports}

    def visitNamed_port_connection(self, ctx):
        """Single named port.

        :return: Tuple of port name and expression.
        :rtype: `tuple[str, Any]`"""
        if ctx.expression():
            return (ctx.port_identifier().getText(), self.visit(ctx.expression()))
        else:
            return (ctx.port_identifier().getText(), None)


# Conversion ----------------------------------------------------------------------------


def create_includes(verilog_path: str | Path, n_features: int = 0) -> dict[str, Module]:
    """Loads sub-modules from a file. Intended for convert_to_graph includes.
    
    :param verilog_path: Verilog source path.
    :param n_features: Number of node X features.
    :return: Dictionary of modules.
    :rtype: dict[str, Module]
    """
    stream = FileStream(str(verilog_path))

    lexer = VerilogLexer(stream)
    token_stream = CommonTokenStream(lexer)
    parser = VerilogParser(token_stream)
    tree = parser.source_text()

    visitor = Visitor(n_features)
    visitor.visit(tree)

    return visitor.modules


def convert_to_graph(
    module_name: str,
    verilog_path: str | Path | None = None,
    verilog_data: str | None = None,
    includes: dict[str, Module] | None = None,
    n_features: int = 0,
    as_networkx: bool = False,
) -> torch_geometric.data.Data | nx.DiGraph:
    """Converts a verilog file to graph.
    
    Either verilog_path or verilog_data has to be defined.

    :param module_name: Top-level module name.
    :param verilog_path: Path to a Verilog source file.
    :param verilog_data: Verilog data to parse.
    :param includes: Dictionary of modules to insert. Can be created using create_includes.
    :param n_features: Number of node X features.
    :param as_networkx: Return as networkx graph.
    :return: Verilog node graph either as networkx or PyG data.
    :rtype: Data | DiGraph"""
    if verilog_data is not None and verilog_path is not None:
        raise ValueError("Only one of verilog or verilog_path should to be defined.")

    if verilog_data is not None:
        stream = InputStream(verilog_data)
    elif verilog_path is not None:
        stream = FileStream(str(verilog_path))
    else:
        raise ValueError("One of verilog or verilog_path has to be defined.")

    lexer = VerilogLexer(stream)
    token_stream = CommonTokenStream(lexer)
    parser = VerilogParser(token_stream)
    tree = parser.source_text()

    visitor = Visitor(n_features, includes=includes)
    visitor.visit(tree)

    try:
        output_module = visitor.modules[module_name]
    except KeyError:
        raise KeyError(
            f"Requested module ({module_name}) not found in file, options are: {', '.join(visitor.modules)}"
        )

    output_module.resolve_instances()
    output_module.type_correction()

    if as_networkx:
        return output_module.graph
    else:
        return from_networkx(output_module.graph)


# Layouts -------------------------------------------------------------------------------


def tree_layout(G: nx.Graph):
    nodes = [x for x in nx.topological_sort(G)]
    x = 0
    y = 0
    used: set[str] = set()
    positions = {k: (0, 0) for k in nodes}
    depth = 0
    const_buffer = defaultdict(list)

    for node in nodes:
        if node.startswith("const"):
            const_buffer[next(G.neighbors(node))].append(node)
            continue

        if node in used:
            used = set()
            depth += 1
            y += 10
            x = 0

        positions[node] = (x, y)
        x += 15

        if node in const_buffer:
            subx = x - 10
            suby = y - 5
            for const in const_buffer[node]:
                positions[const] = (subx, suby)
                subx += 10

        for neighbor in G.neighbors(node):
            used.add(neighbor)

    return positions


def acyclic_layout(G: nx.Graph):
    nodes = [x for x in nx.topological_sort(G)][::-1]
    x = 0
    y = 0
    used = {k: False for k in nodes}
    inverse_map = {f: t for f, t in G.edges}
    positions = {k: (0, 0) for k in nodes}

    for node in nodes:
        if used[node]:
            continue

        positions[node] = (x, y)
        used[node] = True

        n = node
        while n in inverse_map:
            n = inverse_map[n]
            y += 10
            positions[n] = (x, y)
            used[n] = True

        y = 0
        x += 15

    return positions


def _branch_layout_rec(G: nx.DiGraph, parent: str, positions, depth, offset):
    for child, _ in G.in_edges(parent):
        positions[child] = (offset, depth * 10 - 0.02 * offset)
        offset = _branch_layout_rec(G, child, positions, depth + 1, offset)
    return offset + 20


def branch_layout(G: nx.Graph):
    root_node = [x for x in nx.topological_sort(G)][-1]
    positions = {k: (0.0, 0.0) for k in G.nodes()}
    _ = _branch_layout_rec(G, root_node, positions, 1, 0)
    return positions
