import numpy as np
import tqdm
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor


def get_initial(bounds, count=5):
    return np.stack([np.random.uniform(*i, size=count) for i in bounds], axis=1)


def mutate(population, bounds, F=0.2):
    idx_a, idx_b, idx_c = np.random.choice(np.arange(population.shape[0]), size=3, replace=False)
    a, b, c = population[idx_a], population[idx_b], population[idx_c]
    new_x = a + F * (b - c)
    new_x = np.stack([np.clip(new_x[idx], *bounds[idx]) for idx in range(len(new_x))], axis=0)
    return new_x


def crossover(mutated, target, CR=0.2):
    mask = (np.random.uniform(0.0, 1.0, size=mutated.shape) < CR).astype(np.float64)
    return (1 - mask) * mutated + mask * target


def client_update(obj_func, population, bounds, my_target, F=0.2, CR=0.2):
    mutated_x = mutate(population, bounds, F=F)
    crossover_x = crossover(mutated_x, my_target, CR=CR)
    new_obj = obj_func(crossover_x)
    return crossover_x, new_obj


def rosenbrock(x):
    return (2.0 - x[0]) ** 2 + 1000 * (x[1] - x[0] ** 2) ** 2


def sphere(x):
    return x[0] ** 2 + x[1] ** 2


def sort_pop_obj(pop, obj):
    sorted_idx = np.argsort(obj)
    return pop[sorted_idx], obj[sorted_idx]


if __name__ == "__main__":
    with MPIPoolExecutor(root=0) as executor:
        if executor is not None:
            bounds = [[-8, 8], [-8, 8]]
            population = get_initial(bounds, count=8192)
            chk_size = population.shape[0]//MPI.COMM_WORLD.Get_size()
            chk_size = 2**int(np.ceil(np.log2(chk_size)))
            objective_f = np.stack(list(executor.map(rosenbrock, population, chunksize=chk_size)), axis=0)
            population, objective_f = sort_pop_obj(population, objective_f)
            print("Initial x\t\t|\tInitial f", flush=True)
            print(f"{population[0]} | {objective_f[0]}", flush=True)
            num_iterations = 200
            for iter_index in tqdm.tqdm(range(num_iterations)):
                res = list(executor.map(client_update, [rosenbrock]*population.shape[0], [population]*population.shape[0], [bounds]*population.shape[0], population, chunksize=chk_size))

                new_x = np.stack([i[0] for i in res], axis=0)
                new_obj = np.stack([i[1] for i in res], axis=0)

                # mutated_x = np.stack(list(executor.map(mutate, [population]*population.shape[0], [bounds]*population.shape[0], chunksize=chk_size)), axis=0)
                # new_x = np.stack(list(executor.map(crossover, mutated_x, population)), axis=0)
                # new_obj = np.stack(list(executor.map(rosenbrock, new_x, chunksize=chk_size)), axis=0)
                mask = new_obj < objective_f
                prev_pop, prev_obj = population.copy(), objective_f.copy()
                population[mask] = new_x[mask]
                objective_f[mask] = new_obj[mask]
                population, objective_f = sort_pop_obj(population, objective_f)
                if iter_index % 5 == 0:
                    print("Best x\t\t|\tBest f\t|\tDelta f", flush=True)
                print(f"{population[0]} | {objective_f[0]} | {prev_obj[0] - objective_f[0]}", flush=True)


