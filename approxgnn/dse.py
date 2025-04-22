import pymoo

import torch
import numpy as np
import pymoo.core.crossover
import pymoo.core.mutation
import pymoo.core.sampling

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2

from torch import no_grad
from torch_geometric.data import Batch, Data


def convert_to_torch(graph: Data, adder_parameters: torch.Tensor, genome: torch.Tensor):
    condition = graph.adder_id != -1
    selection = graph.adder_id[condition]
    graph.x = torch.zeros((graph.x.shape[0], adder_parameters.shape[1]))
    graph.x[condition] = adder_parameters[genome[selection]]
    return graph.clone()


class AutoaxProblem(Problem):
    def __init__(
        self,
        graph: Data,
        num_components: int,
        qor_parameters: torch.Tensor,
        qor_model: torch.nn.Module,
        hw_parameters: torch.Tensor | None = None,
        hw_model: torch.nn.Module | None = None,
    ):
        super().__init__(n_var=num_components, n_obj=2, n_constr=0, vtype=int)
        self.qor_model = qor_model
        self.hw_model = hw_model
        self.graph = graph
        self.qor_parameters = qor_parameters
        self.hw_parameters = hw_parameters

    def _evaluate(self, X, out, *args, **kwargs):
        n_individuals, n_genes = X.shape

        with no_grad():
            if self.hw_model is not None:
                fitnesses = torch.zeros((n_individuals, 2), dtype=torch.float32)
                batch = Batch.from_data_list(
                    [
                        convert_to_torch(self.graph, self.hw_parameters, individual)
                        for individual in X
                    ]
                )
                fitnesses[:, 0] = self.hw_model(batch).squeeze(-1)
                batch = Batch.from_data_list(
                    [
                        convert_to_torch(self.graph, self.qor_parameters, individual)
                        for individual in X
                    ]
                )
                fitnesses[:, 1] = -self.qor_model(batch).squeeze(-1)
            else:
                batch = Batch.from_data_list(
                    [
                        convert_to_torch(self.graph, self.qor_parameters, individual)
                        for individual in X
                    ]
                )
                fitnesses = self.model(batch)
                fitnesses[:, 1] *= -1  # maximization

        out["F"] = fitnesses


class GraphSampling(pymoo.core.sampling.Sampling):
    def __init__(
        self,
        bitwidths: list[int],
        sampling_data: dict[int, tuple[list[int], np.ndarray[float]]],
    ):
        super().__init__()
        self.sampling_data = sampling_data
        self.bitwidths = bitwidths

    def _do(self, problem, n_samples, **kwargs):
        ret = np.array(
            [self.generate_random_genome() for _ in range(n_samples)], dtype=np.int32
        )
        return ret

    def generate_random_genome(self):
        return np.array(
            [
                np.random.choice(
                    self.sampling_data[w][0],
                    1,
                    p=self.sampling_data[w][1],
                    replace=True,
                )[0]
                for w in self.bitwidths
            ]
        )


class GraphMutation(pymoo.core.mutation.Mutation):
    def __init__(
        self,
        bitwidths: list[int],
        sampling_data: dict[int, tuple[list[int], np.ndarray[float]]],
        prob: float = 0.25,
    ):
        super().__init__()
        self.prob = prob
        self.sampling_data = sampling_data
        self.bitwidths = bitwidths

    def _do(self, problem, X: np.ndarray, **kwargs):
        X = X.copy()
        n_individuals, n_genes = X.shape

        for i in range(n_individuals):
            X[i] = self.mutate_genome(X[i])

        return X

    def mutate_genome(self, genome: np.ndarray):
        next_genome = genome.copy()
        for _ in range(int(np.ceil(self.prob * len(genome)))):
            pos = np.random.randint(0, len(self.bitwidths))

            next_genome[pos] = np.random.choice(
                self.sampling_data[self.bitwidths[pos]][0],
                1,
                p=self.sampling_data[self.bitwidths[pos]][1],
                replace=True,
            )[0]

        return next_genome


class DummyCrossover(pymoo.core.crossover.Crossover):
    def __init__(self):
        super().__init__(2, 2, 0.5)

    def _do(self, problem, parents, **kwargs):
        return parents


def create_nsga2(
    pop_size: int,
    bitwidths: list[int],
    sampling_data: dict[int, tuple[list[int], np.ndarray[float]]],
    mutation_probability: float,
):
    return NSGA2(
        pop_size=pop_size,
        sampling=GraphSampling(bitwidths, sampling_data),
        crossover=DummyCrossover(),
        mutation=GraphMutation(bitwidths, sampling_data, mutation_probability),
        eliminate_duplicates=True,
    )
