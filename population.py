import numpy as np
from numba import jit
import logging
from warnings import filterwarnings
from itertools import permutations

filterwarnings("ignore")


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


logger = get_logger()


class Population(object):
    def __init__(
        self,
        size,
        dice_throw,
        cross_rate=0.1,
        mut_rate=0.05,
        elite_rate=0.1,
        num_generations=1000,
    ):
        self.size = size
        self.num_dice = 14
        self.num_pos = 14
        self.dice_throw = np.sort(np.array(dice_throw, dtype=np.int32))
        # Create a matrix of dice values based on the initial dice throw
        self.dice_values = self.get_dice_values()
        self.fitness = np.zeros(self.size, dtype=np.int32)
        self.mask = np.zeros((self.size, 4), dtype=bool)
        self.crossover_rate = (
            cross_rate  # The percentage of the population that will be crossed over
        )
        self.num_crossover = int(
            self.size * self.crossover_rate
        )  # The number of individuals that will be crossed over
        self.mutation_rate = (
            mut_rate  # The percentage of the population that will be mutated
        )
        self.num_mutated = int(
            self.size * self.mutation_rate
        )  # The number of individuals that will be mutated
        self.elitism = elite_rate  # The percentage of the population that will be kept as is for the next generation
        self.num_elite = int(
            self.size * self.elitism
        )  # The number of elite individuals
        self.num_elite_improve = (
            10  # The number of elite individuals that will be improved
        )
        self.equation_positions = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13]]
        self.generation = 0
        self.num_generations = num_generations
        self.best_solution = None
        self.best_fitness = 0
        self.logger = self.get_logger()

    def get_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        return logger

    @jit
    def get_dice_values(self, n=None):
        # A random permutation of the dice throw is used to fill the dice values
        if n is None:
            n = self.size
        return np.array([np.random.permutation(self.dice_throw) for _ in range(n)])

    @jit
    def calc_fitness(self):
        # Calculate the fitness of each self.best_solution by summing the values of the dice that are successfully placed on the equation board
        # Equation 1: dice1 + dice2 = dice3
        self.mask[:, 0] = (
            self.dice_values[:, 0] + self.dice_values[:, 1] == self.dice_values[:, 2]
        )
        self.fitness += self.mask[:, 0] * self.dice_values[:, 0:3].sum(axis=1)
        # Equation 2: dice4 - dice5 = dice6
        self.mask[:, 1] = (
            self.dice_values[:, 3] - self.dice_values[:, 4] == self.dice_values[:, 5]
        )
        self.fitness += self.mask[:, 1] * self.dice_values[:, 3:6].sum(axis=1)
        # Equation 3: dice7 x dice8 = dice9 +/- dice10
        mask3_1 = (
            self.dice_values[:, 6] * self.dice_values[:, 7]
            == self.dice_values[:, 8] + self.dice_values[:, 9]
        )
        mask3_2 = (
            self.dice_values[:, 6] * self.dice_values[:, 7]
            == self.dice_values[:, 8] - self.dice_values[:, 9]
        )
        self.mask[:, 2] = mask3_1 | mask3_2
        self.fitness += self.mask[:, 2] * self.dice_values[:, 6:10].sum(axis=1)
        # Equation 4: dice11 : dice12 = dice13 +/- dice14
        # 4.1: Check if dice7 is divisible by dice8
        mask4_1 = self.dice_values[:, 10] % self.dice_values[:, 11] == 0
        mask4_2 = (
            self.dice_values[:, 10] // self.dice_values[:, 11]
            == self.dice_values[:, 12] + self.dice_values[:, 13]
        )
        mask4_3 = (
            self.dice_values[:, 10] // self.dice_values[:, 11]
            == self.dice_values[:, 12] - self.dice_values[:, 13]
        )
        self.mask[:, 3] = mask4_1 & (mask4_2 | mask4_3)
        self.fitness += self.mask[:, 3] * self.dice_values[:, 6:10].sum(axis=1)
        return

    @jit
    def crossover(self, crossover_indices):
        self.logger.debug("Performing crossover")
        # Steps:
        # 1. Try to see if one equation can be taken from each parent
        children_made = 0
        for i, m in enumerate(crossover_indices[: self.num_crossover // 2]):
            child_values = np.zeros(self.num_pos, dtype=np.int32)
            # Find equation(s) satisfied
            solved = np.where(self.mask[m])[0]
            if len(solved) == 0:
                continue
            self.logger.debug(f"Solved equations (mother): {solved}")
            solved_pos = []
            for s in solved:
                solved_pos.extend(self.equation_positions[s])
            # Get values of the solved equation positions
            self.logger.debug(f"Mother: {self.dice_values[m]}")
            self.logger.debug(f"Solved positions: {solved_pos}")
            mother_values = self.dice_values[m, solved_pos]
            self.logger.debug(f"Mother solved values: {mother_values}")
            # Set the solved values from mother to the child
            child_values[solved_pos] = mother_values
            self.logger.debug(f"Child values: {child_values}")
            # Could loop through all parents, but for now just take the father from the bottom
            father_values = self.dice_values[
                crossover_indices[-i], ~np.isin(np.arange(self.num_pos), solved_pos)
            ]
            self.logger.debug(f"Father values: {father_values}")
            # Make the child
            child_values = np.concatenate((mother_values, father_values))
            # Fill the child with random values

            self.logger.debug(f"Child values: {child_values}")
            # Check if the values match the dice throw
            if np.all(np.sort(child_values) == self.dice_throw):
                # If they do, then make the child
                self.dice_values[m] = child_values
                children_made += 1
        self.logger.debug(f"Children made: {children_made}")
        return

    @jit
    def mutate(self, mutation_indices):
        self.logger.debug("Performing mutation")
        for ind in mutation_indices:
            # Select two random positions swap the dice values
            pos = np.random.randint(0, self.num_pos, size=2)
            # Swap the positions
            self.dice_values[ind, pos[0]], self.dice_values[ind, pos[1]] = (
                self.dice_values[ind, pos[1]],
                self.dice_values[ind, pos[0]],
            )
        return

    @jit
    def improve_elite(self, elite_indices):
        # Try to improve the elite by trying out permutations of the unsolved equation positions
        for i, m in enumerate(elite_indices[: self.num_elite_improve]):
            original_values = self.dice_values[m].copy()
            original_score = self.fitness[m]
            # Find equation(s) not satisfied
            unsolved = np.where(~self.mask[m])[0]
            if len(unsolved) == 0:
                continue
            self.logger.debug(f"Unsolved equations (elite): {unsolved}")
            unsolved_pos = []
            for s in unsolved:
                unsolved_pos.extend(self.equation_positions[s])
            # Try to permute the unsolved positions
            for p in permutations(unsolved_pos):
                self.dice_values[m, unsolved_pos] = self.dice_values[m, p]
                self.calc_fitness()
                if self.fitness[m] > original_score:
                    self.logger.debug(
                        f"Improved elite {i} from {original_score} to {self.fitness[m]}"
                    )
                    break
                else:
                    self.dice_values[m] = original_values
        return

    @jit
    def evolve(self):
        self.logger.debug(f"Generation {self.generation}")
        # Reset the fitness and mask
        self.generation += 1
        self.fitness = np.zeros(self.size)
        self.mask = np.zeros((self.size, 4), dtype=bool)
        # Calculate the fitness of the current generation
        self.calc_fitness()
        self.print_best_solution()
        # Get the elite indices
        fitness_sorted = np.argsort(self.fitness)[::-1]
        elite_indices = fitness_sorted[: self.num_elite]
        # self.improve_elite(elite_indices)
        crossover_indices = fitness_sorted[
            self.num_elite : (self.num_elite + self.num_crossover)
        ]
        self.crossover(crossover_indices)
        # Find mutation indices by choosing from non elite indices and non crossover indices
        mutation_indices = np.random.choice(
            np.setdiff1d(
                np.arange(self.size), np.concatenate((elite_indices, crossover_indices))
            ),
            size=self.num_mutated,
            replace=False,
        )
        self.mutate(mutation_indices)
        # Keep individuals that are in elite, crossover, and mutated
        keep_indices = np.concatenate(
            (elite_indices, crossover_indices, mutation_indices)
        )
        keep_values = self.dice_values[keep_indices].copy()
        fresh_values = self.get_dice_values(
            n=self.size - self.num_elite - self.num_crossover - self.num_mutated
        )
        # Combine the elite, crossover, mutated, and fresh positions
        self.dice_values = np.vstack((keep_values, fresh_values))
        return

    def print_best_solution(self):
        best_index = np.argmax(self.fitness)
        self.best_solution = self.dice_values[best_index]
        self.best_fitness = self.fitness[best_index]
        self.correct_equations = self.mask[best_index]
        # self.prettyprint_solution()
        return

    def prettyprint_solution(self):
        # Print the self.best_solution. Pad with spaces before displaying the correct equation
        self.logger.info(f"Best solution from logger: {self.best_solution}")
        print("-" * 30)
        print(f"Generation: {self.generation}")
        print(f"Best solution: {self.best_solution}")
        print("Best solution:")
        print(f"Fitness: {self.best_fitness}")
        print(
            f"{self.best_solution[0]} + {self.best_solution[1]} = {self.best_solution[2]}".ljust(
                20
            )
            + f"Correct: {self.correct_equations[0]}"
        )
        print(
            f"{self.best_solution[3]} - {self.best_solution[4]} = {self.best_solution[5]}".ljust(
                20
            )
            + f"Correct: {self.correct_equations[1]}"
        )
        print(
            f"{self.best_solution[6]} x {self.best_solution[7]} = {self.best_solution[8]} +/- {self.best_solution[9]}".ljust(
                20
            )
            + f"Correct: {self.correct_equations[2]}"
        )
        print(
            f"{self.best_solution[10]} : {self.best_solution[11]} = {self.best_solution[12]} +/- {self.best_solution[13]}".ljust(
                20
            )
            + f"Correct: {self.correct_equations[3]}"
        )
        print("-" * 30)
        return

    def run(self):
        # self.logger.debug(f"Dice throw: {self.dice_throw}")
        while self.generation < self.num_generations:
            self.evolve()
        return


def check_solution(dice_array):
    # Convert dice_array to a list of ints
    dice_array = [int(i) for i in dice_array]
    # Check if the solution is correct
    correct_equations = [False, False, False, False]
    score = 0
    eq1 = (dice_array[0] + dice_array[1]) == dice_array[2]
    if eq1:
        correct_equations[0] = True
        score += sum(dice_array[:3])
    eq2 = (dice_array[3] - dice_array[4]) == dice_array[5]
    if eq2:
        correct_equations[1] = True
        score += sum(dice_array[3:6])
    eq3 = (dice_array[6] / dice_array[7]) == (
        (dice_array[8] + dice_array[9]) | (dice_array[8]) - (dice_array[9])
    )
    if eq3:
        correct_equations[2] = True
        score += sum(dice_array[6:10])
    eq4 = (dice_array[10] * dice_array[11]) == (dice_array[12] + dice_array[13]) | (
        dice_array[12] - dice_array[13]
    )
    if eq4:
        correct_equations[3] = True
        score += sum(dice_array[10:])
    return score, correct_equations
