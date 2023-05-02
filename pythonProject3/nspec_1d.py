import math
import numpy as np
import time
from scipy import interpolate
from copy import deepcopy, copy
import scipy.stats as stats
import matplotlib.pyplot as plt


def initialize_death_spline(death_y: np.array, death_cutoff_r: np.array):
    all_death_splines = []
    for i in range(len(death_cutoff_r)):
        all_death_splines.append([])
        for j in range(len(death_cutoff_r[i])):
            death_grid = np.linspace(0, death_cutoff_r[i][j], len(death_y[i][j]))
            all_death_splines[i].append(interpolate.CubicSpline(death_grid, death_y[i][j]))
    return all_death_splines


def initialize_ircdf_spline(birth_ircdf_y: np.array, birth_cutoff_r: np.array):
    all_birth_splines = []
    for i in range(len(birth_ircdf_y)):
        birth_ircdf_grid = np.linspace(0, birth_cutoff_r[i], len(birth_ircdf_y[i]))
        all_birth_splines.append(interpolate.CubicSpline(birth_ircdf_grid, birth_ircdf_y[i]))
    return all_birth_splines


def generate_random_index(n: int, weights: np.array) -> int:
    assert n > 0
    assert (weights >= 0).all()
    assert (weights != 0).any()

    pk = 1.0 * weights / np.sum(weights)
    try:
        index = stats.rv_discrete(values=(np.arange(n), pk)).rvs()
    except ValueError:
        pk[-1] = 1. - np.sum(pk[:-1])
        index = stats.rv_discrete(values=(np.arange(n), pk)).rvs()

    assert 0 <= index < n
    return index


class Grid:

    def __init__(self,
                 n: int,
                 cell_count_x: int,
                 periodic: bool,
                 area_length_x: np.float64):

        self.num_spec = n

        self.cell_count_x = cell_count_x

        # x_coord of each specimen
        self.cell_coords = []

        # death rate of each specimen
        self.spec_death_rates = []

        # number of specs in cell
        self.cell_spec_population = np.zeros((cell_count_x, n), dtype=int)

        self.periodic = periodic
        self.total_death_rate = 0
        self.total_population = np.zeros(n, dtype=int)
        self.spec_sum_death_rate_in_all_cells = np.zeros(n, dtype=np.float64)
        self.spec_sum_death_rate_in_cell = np.zeros(shape=(cell_count_x, n), dtype=np.float64)
        self.area_length_x = area_length_x

        for i in range(cell_count_x):
            self.spec_death_rates.append([])
            self.cell_coords.append([])
            for j in range(n):
                self.spec_death_rates[i].append(np.zeros(0, dtype=np.float64))
                self.cell_coords[i].append(np.zeros(0, dtype=np.float64))

    def get_correct_index(self, i: int) -> int:
        if self.periodic:
            if i < 0:
                i += self.cell_count_x
            if i >= self.cell_count_x:
                i -= self.cell_count_x
        return i

    # def get_death_rates(self, n_spec, i: int) -> np.array:
    #    i = self.get_correct_index(i)
    #    return self.spec_death_rates[i][n_spec]

    # def get_cell_coords(self, n_spec, i: int) -> np.array:
    #    i = self.get_correct_index(i)
    #    return self.cell_coords[i][n_spec]

    # def get_all_coords(self) -> np.array:
    #     x_coords = np.array([], dtype=int)
    #     for lst in self.cell_coords:
    #         x_coords = np.hstack([x_coords, lst], dtype=int)
    #     return x_coords

    # def get_all_death_rates(self) -> np.array:
    #     all_death_rates = np.array([], dtype=np.float64)
    #     for lst in self.spec_death_rates:
    #         all_death_rates = np.hstack([all_death_rates, lst], dtype=np.float64)
    #     return all_death_rates

    def count_distance(self, cell_i: int, cell_j: int, first_spec: int, second_spec: int, i: int, j: int) -> np.float64:
        cell_i = self.get_correct_index(cell_i)
        cell_j = self.get_correct_index(cell_j)
        distance = abs(self.cell_coords[cell_i][first_spec][i] - self.cell_coords[cell_j][second_spec][j])
        if self.periodic:
            return min(distance, self.area_length_x - distance)
        else:
            return distance


class Poisson_1d:
    def __init__(self,
                 n: int,
                 area_length_x: np.float64,
                 cell_count_x: int,
                 b: np.array,
                 d: np.array,
                 dd: np.array,
                 seed: int,
                 initial_population_x: np.array,
                 init_spec_x: np.array,
                 death_y: np.array,
                 death_cutoff_r: np.array,
                 birth_inverse_rcdf_y: np.array,
                 birth_cutoff_r: np.array,
                 periodic: bool,
                 realtime_limit: np.float64
                 ):

        self.num_spec = n
        self.area_length_x = area_length_x
        self.cell_count_x = cell_count_x
        self.b = b
        self.d = d
        self.dd = dd
        self.seed = seed
        np.random.seed(seed)

        self.initial_population_x = initial_population_x
        self.init_spec_x = init_spec_x  # чтобы узнать вид в координате

        self.death_y = death_y
        self.death_cutoff_r = death_cutoff_r

        self.birth_cutoff_r = birth_cutoff_r
        self.birth_ircdf_y = birth_inverse_rcdf_y

        self.periodic = periodic
        self.realtime_limit = realtime_limit

        self.init_time = time.time()
        self.time = 0.
        self.realtime_limit_reached = False
        self.event_count = 0

        self.death_spline = initialize_death_spline(death_y, death_cutoff_r)
        self.birth_ircdf_spline = initialize_ircdf_spline(birth_inverse_rcdf_y, birth_cutoff_r)

        self.rad_in_cells = np.array(
            list(map(math.ceil, death_cutoff_r.reshape(-1) / (area_length_x / cell_count_x)))).reshape((n, n))
        self.grid = Grid(n, cell_count_x, periodic, area_length_x)

        self.initialize()

    def initialize(self):

        # Spawn all specimens
        for j in range(len(self.initial_population_x)):
            x_coord = self.initial_population_x[j]
            cur_spec = self.init_spec_x[j]
            if x_coord < 0 or x_coord > self.area_length_x:
                continue
            i = int(np.floor(x_coord * self.cell_count_x / self.area_length_x))

            if i >= self.cell_count_x:
                i -= 1
            self.grid.cell_coords[i][cur_spec] = np.append(self.grid.cell_coords[i][cur_spec], x_coord)
            self.grid.spec_death_rates[i][cur_spec] = np.append(self.grid.spec_death_rates[i][cur_spec],
                                                                self.d[cur_spec])
            self.grid.spec_sum_death_rate_in_cell[i][cur_spec] += self.d[cur_spec]
            self.grid.total_death_rate += self.d[cur_spec]
            self.grid.total_population[cur_spec] += 1
            self.grid.spec_sum_death_rate_in_all_cells[cur_spec] += self.d[cur_spec]
            self.grid.cell_spec_population[i][cur_spec] += 1

        # Recalculate death rates
        for outer_cell in range(self.cell_count_x):
            for outer_spec in range(self.num_spec):
                for outer_coord_ind in range(len(self.grid.cell_coords[outer_cell][outer_spec])):
                    self.recalculate_death_rates(outer_cell, outer_spec, outer_coord_ind)


    def recalculate_death_rates(self, outer_cell: int, outer_spec: int, outer_coord_ind: int):
        for inner_spec in range(self.num_spec):
            for inner_cell in range(outer_cell - self.rad_in_cells[outer_spec][inner_spec],
                                    outer_cell + self.rad_in_cells[outer_spec][inner_spec] + 1):
                if (not self.periodic) and (inner_cell < 0 or inner_cell >= self.cell_count_x):
                    continue
                inner_cell = self.grid.get_correct_index(inner_cell)
                for inner_coord_ind in range(len(self.grid.cell_coords[inner_cell][inner_spec])):
                    if outer_cell == inner_cell and inner_spec == outer_spec and inner_coord_ind == outer_coord_ind:
                        continue
                    distance = self.grid.count_distance(inner_cell, outer_cell, inner_spec, outer_spec, inner_coord_ind,
                                                        outer_coord_ind)

                    if distance > self.death_cutoff_r[outer_spec][inner_spec]:
                        # too far to interact
                        continue

                    interaction = self.dd[outer_spec][inner_spec] * self.death_spline[outer_spec][inner_spec](distance)

                    self.grid.total_death_rate += interaction
                    self.grid.spec_sum_death_rate_in_all_cells[inner_spec] += interaction
                    self.grid.spec_sum_death_rate_in_cell[inner_cell][inner_spec] += interaction
                    self.grid.spec_death_rates[inner_cell][inner_spec][inner_coord_ind] += interaction

    def approx(self, cell: int, spec: int, coord: int):
        if abs(self.grid.spec_death_rates[cell][spec][coord]) < 1e-10:
            self.grid.spec_death_rates[cell][spec][coord] = 0
        if abs(self.grid.spec_sum_death_rate_in_cell[cell][spec]) < 1e-10:
            self.grid.spec_sum_death_rate_in_cell[cell][spec] = 0
        if abs(self.grid.spec_sum_death_rate_in_all_cells[spec]) < 1e-10:
            self.grid.spec_sum_death_rate_in_all_cells[spec] = 0
        if abs(self.grid.total_death_rate) < 1e-10:
            self.grid.total_death_rate = 0

    def kill_random(self):
        if self.grid.total_population.sum() == 0:
            return

        spec_death_index = generate_random_index(self.grid.num_spec, self.grid.spec_sum_death_rate_in_all_cells)
        # generate dying specimen
        cell_death_index = generate_random_index(self.cell_count_x,
                                                 self.grid.spec_sum_death_rate_in_cell[:, spec_death_index])
        assert len(self.grid.cell_coords[cell_death_index][spec_death_index]) > 0

        if len(self.grid.cell_coords[cell_death_index][spec_death_index]) <= 0:
            return

        in_cell_death_index = generate_random_index(
            len(self.grid.spec_death_rates[cell_death_index][spec_death_index]),
            self.grid.spec_death_rates[cell_death_index][spec_death_index]
        )

        # recalculate death rates
        for spec in range(self.num_spec):
            for cell in range(cell_death_index - self.rad_in_cells[spec_death_index][spec],
                              cell_death_index + self.rad_in_cells[spec_death_index][spec] + 1):
                if (not self.periodic) and (cell < 0 or cell >= self.cell_count_x):
                    continue
                cell = self.grid.get_correct_index(cell)
                for coord in range(len(self.grid.cell_coords[cell][spec])):
                    if cell_death_index == cell and spec == spec_death_index and coord == in_cell_death_index:
                        continue

                    distance = self.grid.count_distance(cell, cell_death_index, spec, spec_death_index, coord,
                                                        in_cell_death_index)
                    if distance > self.death_cutoff_r[spec_death_index][spec]:
                        # too far to interact
                        continue

                    interaction = self.dd[spec_death_index][spec] * self.death_spline[spec_death_index][spec](distance)
                    self.grid.spec_death_rates[cell][spec][coord] -= interaction
                    self.grid.spec_sum_death_rate_in_cell[cell][spec] -= interaction
                    self.grid.spec_sum_death_rate_in_all_cells[spec] -= interaction
                    self.grid.total_death_rate -= interaction
                    self.approx(cell, spec, coord)

        self.grid.total_death_rate -= self.grid.spec_death_rates[cell_death_index][spec_death_index][
            in_cell_death_index]
        self.grid.spec_sum_death_rate_in_cell[cell_death_index][spec_death_index] -= \
        self.grid.spec_death_rates[cell_death_index][spec_death_index][
            in_cell_death_index]
        self.grid.spec_sum_death_rate_in_all_cells[spec_death_index] -= \
        self.grid.spec_death_rates[cell_death_index][spec_death_index][
            in_cell_death_index]
        self.approx(cell_death_index, spec_death_index, in_cell_death_index)

        self.grid.spec_death_rates[cell_death_index][spec_death_index][in_cell_death_index], \
        self.grid.spec_death_rates[cell_death_index][spec_death_index][-1] = \
            self.grid.spec_death_rates[cell_death_index][spec_death_index][-1], \
            self.grid.spec_death_rates[cell_death_index][spec_death_index][in_cell_death_index]
        self.grid.cell_coords[cell_death_index][spec_death_index][in_cell_death_index], \
        self.grid.cell_coords[cell_death_index][spec_death_index][-1] = \
            self.grid.cell_coords[cell_death_index][spec_death_index][-1], \
            self.grid.cell_coords[cell_death_index][spec_death_index][in_cell_death_index]

        self.grid.spec_death_rates[cell_death_index][spec_death_index] = np.delete(
            self.grid.spec_death_rates[cell_death_index][spec_death_index], [-1])
        self.grid.cell_coords[cell_death_index][spec_death_index] = np.delete(
            self.grid.cell_coords[cell_death_index][spec_death_index], [-1])

        self.grid.total_population[spec_death_index] -= 1
        self.grid.cell_spec_population[cell_death_index][spec_death_index] -= 1

    def spawn_random(self):
        # generate parent specimen
        spec_spawn_index = generate_random_index(self.num_spec, self.grid.total_population * self.b)

        cell_index = generate_random_index(self.cell_count_x, self.grid.cell_spec_population[:, spec_spawn_index])
        parent_index = generate_random_index(
            len(self.grid.cell_coords[cell_index][spec_spawn_index]),
            np.ones(len(self.grid.cell_coords[cell_index][spec_spawn_index]))
        )
        new_coord_x = self.grid.cell_coords[cell_index][spec_spawn_index][parent_index] + \
                      self.birth_ircdf_spline[spec_spawn_index](stats.uniform.rvs()) * (
                              2. * stats.bernoulli.rvs(0.5) - 1.)

        if new_coord_x < 0 or new_coord_x > self.area_length_x:
            if not self.periodic:
                # Specimen failed to spawn and died outside area boundaries
                return
            if new_coord_x < 0:
                new_coord_x += self.area_length_x
            if new_coord_x > self.area_length_x:
                new_coord_x -= self.area_length_x

        new_cell = int(np.floor(new_coord_x * self.cell_count_x / self.area_length_x))
        if new_cell == self.cell_count_x:
            new_cell -= 1

        # New specimen is added to the end of array
        self.grid.spec_death_rates[new_cell][spec_spawn_index] = np.append(
            self.grid.spec_death_rates[new_cell][spec_spawn_index], self.d[spec_spawn_index])
        self.grid.cell_coords[new_cell][spec_spawn_index] = np.append(self.grid.cell_coords[new_cell][spec_spawn_index],
                                                                      new_coord_x)
        self.grid.cell_spec_population[new_cell][spec_spawn_index] += 1
        self.grid.total_population[spec_spawn_index] += 1
        self.grid.spec_sum_death_rate_in_cell[new_cell][spec_spawn_index] += self.d[spec_spawn_index]
        self.grid.spec_sum_death_rate_in_all_cells[spec_spawn_index] += self.d[spec_spawn_index]
        self.grid.total_death_rate += self.d[spec_spawn_index]

        index_of_new_spec = len(self.grid.spec_death_rates[new_cell][spec_spawn_index]) - 1


        # recalculate death rates
        for spec in range(self.num_spec):
            for cell in range(new_cell - self.rad_in_cells[spec_spawn_index][spec],
                              new_cell + self.rad_in_cells[spec_spawn_index][spec] + 1):
                if (not self.periodic) and (cell < 0 or cell >= self.cell_count_x):
                    continue
                cell = self.grid.get_correct_index(cell)
                for coord in range(len(self.grid.cell_coords[cell][spec])):
                    if new_cell == cell and spec == spec_spawn_index and coord == index_of_new_spec:
                        continue

                    distance = self.grid.count_distance(cell, new_cell, spec, spec_spawn_index, coord,
                                                        index_of_new_spec)

                    if distance <= self.death_cutoff_r[spec_spawn_index][spec]:
                        interaction = self.dd[spec_spawn_index][spec] * self.death_spline[spec_spawn_index][spec](
                            distance)
                        self.grid.total_death_rate += interaction
                        self.grid.spec_sum_death_rate_in_all_cells[spec] += interaction
                        self.grid.spec_sum_death_rate_in_cell[cell][spec] += interaction
                        self.grid.spec_death_rates[cell][spec][coord] += interaction

                    if distance <= self.death_cutoff_r[spec][spec_spawn_index]:
                        interaction = self.dd[spec][spec_spawn_index] * self.death_spline[spec][spec_spawn_index](
                            distance)
                        self.grid.total_death_rate += interaction
                        self.grid.spec_sum_death_rate_in_all_cells[spec_spawn_index] += interaction
                        self.grid.spec_sum_death_rate_in_cell[new_cell][spec_spawn_index] += interaction
                        self.grid.spec_death_rates[new_cell][spec_spawn_index][index_of_new_spec] += interaction

    def make_event(self):
        if self.grid.total_population.sum() == 0:
            return
        self.event_count += 1
        self.time += stats.expon.rvs(
            scale=1. / np.array(
                (self.grid.total_population * self.b + self.grid.spec_sum_death_rate_in_all_cells)).sum())
        born_probability = np.array(self.grid.total_population * self.b).sum() / np.array(
            self.grid.total_population * self.b + self.grid.spec_sum_death_rate_in_all_cells).sum()
        if stats.bernoulli.rvs(born_probability) == 0:
            self.kill_random()
        else:
            self.spawn_random()

    def run_events(self, events: int):
        if events > 0:
            for i in range(events):
                if time.time() > self.init_time + self.realtime_limit:
                    self.realtime_limit_reached = True
                    return
                self.make_event()


def run_simulations(sim: Poisson_1d, iterations: int):
    assert iterations > 0
    time = [sim.time]
    pop = [deepcopy(sim.grid.total_population)]

    for i in range(1, iterations + 1):
        sim.run_events(1)
        if sim.realtime_limit_reached:
            break
        pop.append(deepcopy(sim.grid.total_population))
        time.append(sim.time)

    return time, pop, sim.realtime_limit_reached


def test_sim1():
    death_grid = np.linspace(0.0, 5., num=1001)
    birth_grid = np.linspace(0.5, 1. - 1e-10, num=1001)
    init_pop, init_spec = [], []
    for i in range(100):
        coord_in_cell = stats.randint.rvs(0, 600)
        while (coord_in_cell in init_pop):
            coord_in_cell = stats.randint.rvs(0, 600)
        init_pop.append(coord_in_cell)
        init_spec.append(0)
    for i in range(200):
        coord_in_cell = stats.randint.rvs(0, 600)
        while (coord_in_cell in init_pop):
            coord_in_cell = stats.randint.rvs(0, 600)
        init_pop.append(coord_in_cell)
        init_spec.append(1)


    sim = Poisson_1d(
        n=2,
        area_length_x=np.float_(600.0),
        dd=np.array([[0.001, 0.001], [0.001, 0.001]], dtype=np.float64),
        cell_count_x=60,
        b=np.array([0.4, 0.4], dtype=np.float64),
        d=np.array([0.2, 0.2], dtype=np.float64),
        initial_population_x=init_pop,
        init_spec_x=init_spec,
        seed=1234,
        death_y=np.array([[stats.norm.pdf(death_grid, scale=0.04), stats.norm.pdf(death_grid, scale=0.04)],
                          [stats.norm.pdf(death_grid, scale=0.04), stats.norm.pdf(death_grid, scale=0.04)]]),
        birth_inverse_rcdf_y=np.array([stats.norm.ppf(birth_grid, scale=0.04), stats.norm.ppf(birth_grid, scale=0.25)]),
        death_cutoff_r=np.array([[3, 3], [3, 3]]),
        birth_cutoff_r=np.array([3, 3]),
        periodic=True,
        realtime_limit=np.float_(300)
    )
    simulation_time, population, limit_reached = run_simulations(sim, 100000000)
    plt.figure(figsize=(14, 12))
    plt.title("зависимость популяции от времени")
    plt.text(0, np.max(population) * 0.95, "b = [0.4, 0.4], d = [0.2, 0.2], dd = [[0.001, 0.001], [0.001, 0.001]]", fontsize=14.)
    plt.text(0, np.max(population) * 0.9, "$\sigma_m = (0.04, 0.06)$, $\sigma_w = [[0.12, 0.02], [0.02, 0.12]]$ ", fontsize=14.)
    plt.plot(simulation_time, population)
    plt.xlabel("время")
    plt.ylabel("численность")
    plt.legend(['N1', 'N2'])



    # plt.title("расселение по областям")
    # plt.bar(np.arange(1, len(result[3]) + 1), result[3])

    plt.show()


test_sim1()