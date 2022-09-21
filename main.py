from copy import deepcopy
from typing import Iterable, Union, Callable

import numpy as np
import tqdm
import random
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor


class Problem(object):
    def __init__(self):
        self.feval = 0

    def __call__(self, *args, **kwargs):
        self.feval += 1
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return

    def categorical_bounds(self) -> Union[None, Iterable]:
        return None

    def float_bounds(self) -> Union[None, Iterable]:
        return None

    def integer_bounds(self) -> Union[None, Iterable]:
        return None


class SphereProblem(Problem):
    def forward(self, x):
        x, _, _ = x
        return np.sum(np.square(x))


class Population(object):
    def __init__(self, problem: Problem = SphereProblem(), size=128):
        self.problem = problem
        self.size = size
        self.__init_float_population()
        self.__init_categorical_population()
        self.__init_integer_population()
        self.__objective = None

    @property
    def has_float(self):
        return self.problem.float_bounds() is not None

    @property
    def has_integer(self):
        return self.problem.integer_bounds() is not None

    @property
    def has_categorical(self):
        return self.problem.categorical_bounds() is not None

    @property
    def feval(self):
        return self.problem.feval

    def __init_float_population(self):
        bounds = self.problem.float_bounds()
        if bounds is not None:
            self.__float_pop = np.stack([(np.random.uniform(*i, size=self.size).astype(np.float64)) for i in bounds],
                                        axis=-1)
        else:
            self.__float_pop = None

    def __init_integer_population(self):
        bounds = self.problem.integer_bounds()
        if bounds is not None:
            self.__integer_pop = np.stack([(np.random.randint(*i, size=self.size).astype(np.int64)) for i in bounds],
                                          axis=-1)
        else:
            self.__integer_pop = None

    def __init_categorical_population(self):
        bounds = self.problem.categorical_bounds()
        if bounds is not None:
            self.__categorical_pop = [[random.choice(i) for i in bounds] for _ in range(self.size)]
        else:
            self.__categorical_pop = None

    def get_f(self):
        if self.__objective is None:
            with MPIPoolExecutor(root=0) as executor:
                self.__objective = np.stack(list(executor.map(
                    self.problem,
                    self.get_x(),
                    chunksize=64
                )), axis=0)
            self.problem.feval += self.size
            self.sort()
        return self.__objective

    def get_x(self):
        zipper = []
        if self.has_float:
            zipper.append(self.__float_pop)
        else:
            zipper.append([None] * self.size)
        if self.has_integer:
            zipper.append(self.__integer_pop)
        else:
            zipper.append([None] * self.size)
        if self.has_categorical:
            zipper.append(self.__categorical_pop)
        else:
            zipper.append([None] * self.size)
        return deepcopy(list(zip(*zipper)))

    def get_fx(self):
        return self.__float_pop

    def get_ix(self):
        return self.__integer_pop

    def get_cx(self):
        return self.__categorical_pop

    def get_best_xf(self):
        return self.get_f()[0], self.get_x()[0]

    def set_pop_objective(self, new_objective, new_population):
        self.__objective = new_objective
        self.__float_pop, self.__integer_pop, self.__categorical_pop = new_population
        self.sort()

    def sort(self):
        sorted_idx = np.argsort(self.get_f())
        self.__objective = self.__objective[sorted_idx]
        if self.has_float:
            self.__float_pop = self.__float_pop[sorted_idx]
        if self.has_integer:
            self.__integer_pop = self.__integer_pop[sorted_idx]
        if self.has_categorical:
            self.__categorical_pop = [deepcopy(self.__categorical_pop[idx]) for idx in sorted_idx]

    def clone(self):
        new_pop = Population(self.problem, self.size)
        new_pop.set_pop_objective(self.__objective, [self.__float_pop, self.__integer_pop, self.__categorical_pop])
        return new_pop


class Algorithm(object):
    def __init__(self):
        return

    def evolve(self, population: Population, iter: int = 1):
        pop = population
        with MPIPoolExecutor(root=0) as executor:
            with tqdm.tqdm(range(iter)) as tq_iter:
                for _ in tq_iter:
                    prev_pop = pop

                    def map_func(*args, **kwargs):
                        kwargs.setdefault("chunksize", 64)
                        return executor.map(*args, **kwargs)

                    pop = self.step(pop, map_func=map_func)
                    tq_iter.set_description(
                        f"min(f)={pop.get_best_xf()[0]:.4e}, d(f)={pop.get_best_xf()[0] - prev_pop.get_best_xf()[0]:.4e}, feval={pop.feval}")
        return pop

    def __str__(self):
        return "<Algorithm>"

    def __repr__(self):
        return "Algorithm()"

    def step(self, population: Population, map_func: Callable = None):
        return population.clone()


def where_along_axis(mask, arr1, arr2):
    assert (arr1.shape == arr2.shape)
    ret = np.copy(arr1)
    ret[mask] = arr2[mask]
    return ret


def get_initial(float_bounds, integer_bounds, count=5):
    float_population = np.stack([np.random.uniform(*i, size=count) for i in float_bounds], axis=1)
    integer_population = np.stack([np.random.randint(*i, size=count) for i in integer_bounds], axis=1)
    return float_population, integer_population


def update_mutation_rate(F, lower_f=0.1, upper_f=1.0, tau=0.1):
    rand2 = np.random.uniform(0.0, 1.0)
    if rand2 < tau:
        return lower_f + (upper_f - lower_f) * np.random.uniform(0.0, 1.0)
    else:
        return F


def update_crossover_rate(CR, lower_cr=0.1, upper_cr=1.0, tau=0.1):
    rand2 = np.random.uniform(0.0, 1.0)
    if rand2 < tau:
        return lower_cr + (upper_cr - lower_cr) * np.random.uniform(0.0, 1.0)
    else:
        return CR


def mutate_float(population, bounds, F=0.2):
    idx_a, idx_b, idx_c = np.random.choice(np.arange(population.shape[0]), size=3, replace=False)
    a, b, c = population[idx_a], population[idx_b], population[idx_c]
    new_x = a + F * (b - c)
    new_x = np.stack([np.clip(new_x[idx], *bounds[idx]) for idx in range(len(new_x))], axis=0)
    return new_x


def mutate_integer(x, bounds, F=0.2):
    t = np.random.uniform(0.0, 1.0, size=x.shape[0])
    dx = np.copy(x)
    dx[t <= F] = [np.random.randint(i[0], i[1]) for i, local_t in zip(bounds, t) if local_t <= F]
    return dx


def mutate_categorical(x, categories, F=0.2):
    t = np.random.uniform(0.0, 1.0, size=len(x))
    dx = deepcopy(x)
    for idx, t in enumerate(np.random.uniform(0.0, 1.0, size=len(x))):
        if t <= F:
            dx[idx] = random.choice(categories[idx])
    return dx


def crossover_float(mutated, target, CR=0.2):
    mask = (np.random.uniform(0.0, 1.0, size=mutated.shape) < CR).astype(np.float64)
    return (1 - mask) * mutated + mask * target


def crossover_integer(mutated, target, CR=0.2):
    mask = (np.random.uniform(0.0, 1.0, size=mutated.shape) < CR)
    return np.where(mask, mutated, target)


def crossover_categorical(mutated, target, CR=0.2):
    mask = (np.random.uniform(0.0, 1.0, size=len(mutated)) < CR)
    dx = deepcopy(mutated)
    for idx, t in enumerate(mask):
        if t:
            dx[idx] = target[idx]
    return dx


def client_update(obj_func, population, bounds, my_target, F=0.2, CR=0.2, lower_f=0.1, upper_f=1.0, lower_cr=0.1,
                  upper_cr=1.0):
    fpop, ipop, cpop = population
    fbound, ibound, cvals = bounds
    fx, ix, cx = my_target

    F = update_mutation_rate(F, lower_f, upper_f)
    CR = update_crossover_rate(CR, lower_cr, upper_cr)

    if fbound is not None:
        mfx = mutate_float(fpop, fbound, F=F)
        cfx = crossover_float(mfx, fx, CR=CR)
    else:
        cfx = None
    if ibound is not None:
        mix = mutate_integer(ix, ibound, F=F)
        cix = crossover_integer(mix, ix, CR=CR)
    else:
        cix = None
    if cvals is not None:
        mcx = mutate_categorical(cx, cvals, F=F)
        ccx = crossover_categorical(mcx, cx, CR=CR)
    else:
        ccx = None

    nx = [cfx, cix, ccx]

    new_obj = obj_func(nx)

    return cfx, cix, ccx, new_obj, F, CR


class SaDEwithSGA(Algorithm):
    def __init__(self, F=0.8, CR=0.5):
        super().__init__()
        self.__mutation_rate_initial = F
        self.__crossover_rate_initial = CR
        self.mutation_rate = None
        self.crossover_rate = None

    def step(self, population, map_func: Callable = None):
        f_pop, i_pop, c_pop = population.get_fx(), population.get_ix(), population.get_cx()
        if self.mutation_rate is None or len(self.mutation_rate) != population.size:
            self.mutation_rate = np.ones(population.size, dtype=np.float64) * self.__mutation_rate_initial
        if self.crossover_rate is None or len(self.crossover_rate) != population.size:
            self.crossover_rate = np.ones(population.size, dtype=np.float64) * self.__crossover_rate_initial
        res = list(map_func(
            client_update,
            [population.problem] * population.size,
            [(f_pop, i_pop, c_pop)] * population.size,
            [(
                population.problem.float_bounds(),
                population.problem.integer_bounds(),
                population.problem.categorical_bounds()
            )] * population.size,
            population.get_x(),
            self.mutation_rate,
            self.crossover_rate
        ))
        population.problem.feval += population.size

        nfx = np.stack([i[0] for i in res], axis=0) if population.has_float else None
        nix = np.stack([i[1] for i in res], axis=0) if population.has_integer else None
        ncx = np.stack([i[2] for i in res], axis=0) if population.has_categorical else None
        n_obj = np.stack([i[3] for i in res], axis=0)
        n_F = np.stack([i[4] for i in res], axis=0)
        n_CR = np.stack([i[5] for i in res], axis=0)

        # print(population.get_f())
        insertion_indices = np.searchsorted(population.get_f(), n_obj)
        # print(insertion_indices)

        first_set_idx, second_set_idx = pop.size//2, pop.size//2
        if pop.size % 2 == 1:
            first_set_idx += 1

        obj_f = np.insert(population.get_f(), insertion_indices, n_obj, axis=0)

        if nfx is not None:
            fxn = np.insert(population.get_fx(), insertion_indices, nfx, axis=0)
            fxn = np.concatenate([fxn[:first_set_idx], fxn[-second_set_idx:]], axis=0)
        else:
            fxn = None
        if nix is not None:
            ixn = np.insert(population.get_ix(), insertion_indices, nix, axis=0)
            ixn = np.concatenate([ixn[:first_set_idx], ixn[-second_set_idx:]], axis=0)
        else:
            ixn = None
        if ncx is not None:
            cxn = population.get_cx()
            for idx, insert_idx in enumerate(insertion_indices[::-1]):
                cxn.insert(insert_idx, ncx[::-1][idx])
            cxn = cxn[:first_set_idx] + cxn[-second_set_idx:]
        else:
            cxn = None

        # print(nfx, nix, ncx)
        # print(population.get_fx(), population.get_ix(), population.get_cx())
        # print(fxn, ixn, cxn)

        self.mutation_rate = np.insert(self.mutation_rate, insertion_indices, n_F, axis=0)
        self.crossover_rate = np.insert(self.crossover_rate, insertion_indices, n_CR, axis=0)

        obj_f = np.concatenate([obj_f[:first_set_idx], obj_f[:second_set_idx]], axis=0)
        self.mutation_rate = np.concatenate([self.mutation_rate[:first_set_idx], self.mutation_rate[:second_set_idx]], axis=0)
        self.crossover_rate = np.concatenate([self.crossover_rate[:first_set_idx], self.crossover_rate[:second_set_idx]], axis=0)

        new_pop = population.clone()
        new_pop.set_pop_objective(obj_f, [fxn, ixn, cxn])

        return new_pop


class RosenbrockProblem(Problem):
    def __init__(self, dim=2, a=1.0, b=100.0):
        super().__init__()
        self.dim = max(2, dim)
        self.a = a
        self.b = b

    def float_bounds(self):
        return [[-8, 8]] * self.dim

    def forward(self, x):
        x, _, _ = x
        return np.sum(self.b * np.square(x[1:] - np.square(x[:-1])) + np.square(self.a - x[:-1])) + np.square(
            self.a - x[-1])


class PNormProblem(Problem):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def float_bounds(self):
        return [[-8, 8]] * self.dim

    def categorical_bounds(self):
        return [[2, 4, 6, 8]]

    def forward(self, x):
        fx, _, cx = x
        return np.sum(fx ** cx)


class GriewankProblem(Problem):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def float_bounds(self):
        return [[-8, 8]] * self.dim

    def integer_bounds(self):
        return [[2, 3]]

    def forward(self, x):
        fx, ix, _ = x
        """Griewank's function multimodal, symmetric, inseparable """
        return np.abs(1 + (np.sum(fx ** ix) / 4000.0) - np.prod(np.cos(fx) / np.sqrt(np.arange(self.dim) + 1)))


if __name__ == "__main__":
    if MPI.COMM_WORLD.Get_rank() == 0:
        my_udp = PNormProblem(dim=2)
        pop = Population(my_udp, size=1024)
        my_algo = SaDEwithSGA()

        print(len(pop.get_x()))
        # print(pop.get_x())
        print(pop.get_best_xf())
        print(pop.get_f())

        new_pop = my_algo.evolve(pop, iter=200)

        print(new_pop.get_best_xf())
