from genetic_core import GeneticCore, BaseGenetic
from greedy import KnapsackGreedy
import random


class KnapsackGenetic(BaseGenetic):

    def __init__(self,
                 initial_population_function,
                 fitness_evaluation_function,
                 crossover_function,
                 mutation_function,
                 number_of_random_initial_individuals: int = 50,
                 number_of_greedy_initial_individuals: int = 0,
                 crossover_probability: float = 0.85,
                 mutation_probability: float = 0.1,
                 number_of_iterations: int = 2000,
                 stop_if_without_changes: bool = False,
                 number_of_iterations_without_changes: int = 50,
                 use_elitism: bool = True,
                 use_visualization: bool = False,
                 use_correction_after_each_step: bool = False
                 ):
        super().__init__(
            crossover_probability=crossover_probability,
            number_of_iterations=number_of_iterations,
            stop_if_without_changes=stop_if_without_changes,
            number_of_iterations_without_changes=number_of_iterations_without_changes,
            use_elitism=use_elitism,
            use_visualization=use_visualization
        )
        self.__initial_population_function = getattr(self, "_KnapsackGenetic__" + initial_population_function)
        self.__fitness_evaluation_function = getattr(self, "_KnapsackGenetic__" + fitness_evaluation_function)
        self.__crossover_function = getattr(self, "_KnapsackGenetic__" + crossover_function)
        self.__mutation_function = getattr(self, "_KnapsackGenetic__" + mutation_function)

        self.__number_of_random_initial_individuals = number_of_random_initial_individuals
        self.__number_of_greedy_initial_individuals = number_of_greedy_initial_individuals
        self.__mutation_probability = mutation_probability
        self.__use_correction_after_each_step = use_correction_after_each_step

    def get_solution(self, objects: list[tuple[float, float]], capacity: int) -> tuple[tuple[list[int], float, float], int]:
        self.__objects=objects
        self.__capacity=capacity
        self.__number_of_objects = len(objects)
        genetic_core = GeneticCore()
        best_individual, iterations_count = genetic_core.get_best_individual(self)
        result = self.__get_solution_from_best_individual(best_individual)
        return result, iterations_count

    def __get_solution_from_best_individual(self, best_individual: list[int]) -> tuple[list[int], float, float]:
        sum_value = 0
        sum_weight = 0
        for i, presence in enumerate(best_individual):
            if presence:
                current_value, current_weight = self.__objects[i]
                sum_value = sum_value + current_value
                sum_weight = sum_weight + current_weight
        return best_individual, sum_value, sum_weight

    # Функции начальной инициализации

    def initial_population_function(self):
        return self.__initial_population_function()

    def __get_initial_population(self) -> list[list[int]]:
        initial_population = []
        if self.__number_of_random_initial_individuals > 0:
            for _ in range(self.__number_of_random_initial_individuals):
                new_individual = [random.randint(0, 1) for _ in range(self.__number_of_objects)]
                initial_population.append(new_individual)
        if self.__number_of_greedy_initial_individuals > 0:
            greedy_solver = KnapsackGreedy()
            new_individual = greedy_solver.get_solution(self.__objects, self.__capacity)[0]
            for _ in range(self.__number_of_greedy_initial_individuals):
                initial_population.append(new_individual)

        return initial_population

    # Фитнесс функции

    def fitness_evaluation_function(self, population):
        return self.__fitness_evaluation_function(population)

    def __simple_fitness_evaluation(self, population: list[list[int]]) -> list[float]:
        fitness_scores = []
        for individual in population:
            scores = 0
            free_space = self.__capacity
            for i, presence in enumerate(individual):
                if presence:
                    current_value, current_weight = self.__objects[i]
                    scores = scores + current_value
                    free_space = free_space - current_weight
                    if free_space < 0:
                        scores = 0
                        break

            fitness_scores.append(scores)

        return fitness_scores

    def __fitness_evaluation_without_zeroing_out(self, population: list[list[int]]) -> list[float]:
        fitness_scores = []
        for individual in population:
            self.__individual_correction(individual)
            sum_value = sum(y[0] for x, y in zip(individual, self.__objects) if x == 1)
            fitness_scores.append(sum_value)

        return fitness_scores

    # Функции скрещивания
    def crossover_function(self, first_parent, second_parent):
        if self.__use_correction_after_each_step:
            return self.__crossover_with_correction(first_parent, second_parent)
        else:
            return self.__crossover_function(first_parent, second_parent)

    def __single_point_crossover(self, first_parent: list[int], second_parent:list[int]) -> tuple[list[int], list[int]]:
        crossover_point = random.randint(0, len(first_parent) - 1)

        first_child = first_parent[:crossover_point] + second_parent[crossover_point:]
        second_child = second_parent[:crossover_point] + first_parent[crossover_point:]

        return first_child, second_child

    def __greedy_crossover(self, first_parent: list[int], second_parent: list[int]) -> tuple[list[int], list[int]]:
        new_individual = [presence_first_parent or presence_second_parent for presence_first_parent, presence_second_parent in zip(first_parent, second_parent)]
        existing_objects_indexes = [index for index, presence_new_individual in enumerate(new_individual) if presence_new_individual == 1]
        objects_for_greedy = [self.__objects[index] for index in existing_objects_indexes]
        greedy_solver = KnapsackGreedy()
        greedy_solution = greedy_solver.get_solution(objects_for_greedy, self.__capacity)[0]
        for index, value in enumerate(greedy_solution):
            new_individual[existing_objects_indexes[index]] = value

        return new_individual, new_individual

    def __zigzag_crossover(self, first_parent: list[int], second_parent: list[int]) -> tuple[list[int], list[int]]:
        first_child = []
        second_child = []
        zigzag_direction = 0
        number_of_objects = len(first_parent)
        for i in range(number_of_objects):
            first_gene = first_parent[i]
            second_gene = second_parent[i]
            if zigzag_direction == 0:
                first_child.append(first_gene)
                second_child.append(second_gene)
            else:
                first_child.append(second_gene)
                second_child.append(first_gene)
            zigzag_direction = zigzag_direction ^ 1

        return first_child, second_child

    def __crossover_with_correction(self, first_parent: list[int], second_parent: list[int]) -> tuple[list[int], list[int]]:
        crossover_function = self.__crossover_function
        new_individuals = crossover_function(first_parent, second_parent)
        for individual in new_individuals:
            self.__individual_correction(individual)
        return new_individuals

    # Функции мутации

    def mutation_function(self, individual):
        if self.__use_correction_after_each_step:
            return self.__mutation_with_correction(individual)
        else:
            return self.__mutation_function(individual)

    def __each_gene_mutation(self, individual: list[int]) -> list[int]:
        probably_mutated_individual = individual.copy()
        if self.__mutation_probability > 0:
            for i, presence in enumerate(probably_mutated_individual):
                random_value = random.random()
                if random_value <= self.__mutation_probability:
                    probably_mutated_individual[i] = presence ^ 1

        return probably_mutated_individual

    def __one_gene_mutation(self, individual: list[int]) -> list[int]:
        probably_mutated_individual = individual.copy()
        if self.__mutation_probability > 0:
            random_value = random.random()
            if random_value <= self.__mutation_probability:
                random_gene_index = random.randint(0, len(probably_mutated_individual) - 1)
                probably_mutated_individual[random_gene_index] = probably_mutated_individual[random_gene_index] ^ 1

        return probably_mutated_individual

    def __mutation_with_correction(self,  individual: list[int]) -> list[int]:
        mutation_function = self.__mutation_function
        probably_mutated_individual = mutation_function(individual)
        self.__individual_correction(probably_mutated_individual)
        return probably_mutated_individual

    # Функции представления результата в виде строки

    def solution_to_string(self, solution: tuple[tuple[list[int], float, float], int]) -> str:
        (best_individual, sum_value, sum_weight), iterations_count = solution
        result = (f"\nGenetic algorithm results\n"
                  f"Objects: {best_individual}\n"
                  f"Sum value: {sum_value} Sum weight: {sum_weight}\n"
                  f"Number of iterations for best result: {iterations_count}")
        return result

    # Общее

    def __individual_correction(self, individual):
        sum_weight = sum(y[1] for x, y in zip(individual, self.__objects) if x == 1)
        while sum_weight > self.__capacity:
            index_to_replace = random.choice([i for i, including in enumerate(individual) if including == 1])
            individual[index_to_replace] = 0
            sum_weight = sum_weight - self.__objects[index_to_replace][1]

