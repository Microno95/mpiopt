import numpy as np
import tqdm
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor, MPICommExecutor


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


def crossover_float(mutated, target, CR=0.2, lower_cr=0.0, upper_cr=1.0):
    mask = (np.random.uniform(0.0, 1.0, size=mutated.shape) < CR).astype(np.float64)
    return (1 - mask) * mutated + mask * target


def crossover_integer(mutated, target, CR=0.2, lower_cr=0.0, upper_cr=1.0):
    mask = (np.random.uniform(0.0, 1.0, size=mutated.shape) < CR)
    return np.where(mask, mutated, target)


def client_update(obj_func, population, bounds, my_target, F=0.2, CR=0.2, lower_f=0.1, upper_f=1.0, lower_cr=0.1,
                  upper_cr=1.0):
    float_population, integer_population = population
    float_bounds, integer_bounds = bounds
    my_target_float, my_target_integer = my_target
    F = update_mutation_rate(F, lower_f, upper_f)
    CR = update_crossover_rate(CR, lower_cr, upper_cr)
    mutated_x_float = mutate_float(float_population, float_bounds, F=F)
    mutated_x_integer = mutate_integer(my_target_integer, integer_bounds, F=F)
    crossover_x_float = crossover_float(mutated_x_float, my_target_float, CR=CR)
    crossover_x_integer = crossover_integer(mutated_x_integer, my_target_integer, CR=CR)
    new_obj = obj_func(crossover_x_float, crossover_x_integer)
    return crossover_x_float, crossover_x_integer, new_obj, F, CR


def rosenbrock(x, *args):
    return (1.0 - x[0]) ** 2 + 1000 * (x[1] - x[0] ** 2) ** 2


def pnorm(x, integer_x):
    return np.sum(x ** integer_x)


def sphere(x, *args):
    return np.sum(x ** 2)


def griewank_function(x, integer_x):
    """Griewank's function multimodal, symmetric, inseparable """
    return np.abs(1 + (np.sum(x ** integer_x[0]) / 4000.0) - np.prod(np.cos(x) / np.sqrt(np.arange(len(x)) + 1)))


def sort_pop_obj(obj, *args):
    sorted_idx = np.argsort(obj)
    return obj[sorted_idx], tuple(map(lambda x: x[sorted_idx], args))


if __name__ == "__main__":
    target_func = griewank_function

    with MPIPoolExecutor(root=0) as executor:
        if executor is not None:
            float_bounds = [[-600, 600]]*100
            integer_bounds = [[2, 4]]

            pop_size = 128

            population = get_initial(float_bounds, integer_bounds, count=pop_size)
            mutation_rate = np.random.uniform(0.5, 1.0, size=pop_size)
            crossover_rate = np.random.uniform(0.8, 1.0, size=pop_size)
            chk_size = pop_size // MPI.COMM_WORLD.Get_size()
            chk_size = 2 ** int(np.ceil(np.log2(chk_size)))
            objective_f = np.stack(list(executor.map(target_func, *population, chunksize=chk_size)), axis=0)
            np.set_printoptions(precision=4)
            objective_f, (mutation_rate, crossover_rate, *population) = sort_pop_obj(objective_f, mutation_rate,
                                                                                     crossover_rate, *population)
            print("Initial x\t|\tInitial f", flush=True)
            print(f"{population[0][0]}, {population[1][0]} | {objective_f[0]:.4e}", flush=True)
            print(flush=True)
            num_iterations = 200
            with tqdm.tqdm(range(num_iterations)) as tq_iter:
                for iter_index in tq_iter:
                    res = list(executor.map(client_update, [target_func] * pop_size, [population] * pop_size,
                                            [(float_bounds, integer_bounds)] * pop_size, list(zip(*population)),
                                            mutation_rate, crossover_rate, chunksize=chk_size))

                    new_x_float = np.stack([i[0] for i in res], axis=0)
                    new_x_integer = np.stack([i[1] for i in res], axis=0)
                    new_obj = np.stack([i[2] for i in res], axis=0)
                    new_F = np.stack([i[3] for i in res], axis=0)
                    new_CR = np.stack([i[4] for i in res], axis=0)

                    mask = new_obj < objective_f
                    prev_pop, prev_obj = (population[0].copy(), population[1].copy()), objective_f.copy()

                    population[0][mask] = new_x_float[mask]
                    population[1][mask] = new_x_integer[mask]
                    objective_f[mask] = new_obj[mask]
                    mutation_rate[mask] = new_F[mask]
                    crossover_rate[mask] = new_CR[mask]

                    new_obj[~mask] = objective_f[~mask]
                    new_x_float[~mask] = population[0][~mask]
                    new_x_integer[~mask] = population[1][~mask]
                    new_F[~mask] = mutation_rate[~mask]
                    new_CR[~mask] = crossover_rate[~mask]

                    objective_f, (mutation_rate, crossover_rate, *population) = sort_pop_obj(objective_f, mutation_rate,
                                                                                             crossover_rate,
                                                                                             *population)
                    new_obj, (new_F, new_CR, new_x_float, new_x_integer) = sort_pop_obj(new_obj, new_F, new_CR,
                                                                                        new_x_float, new_x_integer)

                    objective_f[pop_size // 2:] = new_obj[:pop_size // 2]
                    population[0][pop_size // 2:] = new_x_float[:pop_size // 2]
                    population[1][pop_size // 2:] = new_x_integer[:pop_size // 2]

                    tq_iter.set_description(
                        f"{objective_f[0]:.4e} | {prev_obj[0] - objective_f[0]:.4e} | {mutation_rate[0]:.4e}, {crossover_rate[0]:.4e} || {objective_f[-1]:.4e}")
                    tq_iter.set_postfix(dict(integer_vals=population[1][0]))
