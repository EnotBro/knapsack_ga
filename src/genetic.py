from genetic_core import GeneticCore
from greedy import KnapsackGreedy
import random

class KnapsackGenetic:

    def get_solution(self,
                     objects: list[tuple[int, int]],
                     capacity: int,
                     number_of_random_initial_individuals: int = 50,
                     number_of_greedy_initial_individuals: int = 0,
                     crossover_probability: float = 0.85,
                     mutation_probability: float = 0.1,
                     number_of_iterations: int = 2000,
                     stop_if_without_changes: bool = False,
                     number_of_iterations_without_changes: int = 50,
                     use_elitism: bool = True,
                     use_visualization: bool = False
                     ) -> tuple[tuple[list[int], int, int], int]:
        self.objects=objects
        self.capacity=capacity
        number_of_objects = len(objects)
        genetic_core = GeneticCore()
        best_individual, iterations_count = genetic_core.get_best_individual(
            initial_population_function=self.__get_initial_population,
            initial_population_args=(number_of_objects,
                                     number_of_random_initial_individuals,
                                     number_of_greedy_initial_individuals,
                                     (objects, capacity)),
            fitness_evaluation_function=self.__fitness_evaluation_without_zeroing_out,
            fitness_evaluation_args=(objects, capacity),
            crossover_function=self.__single_point_crossover,
            crossover_probability=crossover_probability,
            mutation_function=self.__each_gene_mutation,
            mutation_probability=mutation_probability,
            number_of_iterations=number_of_iterations,
            stop_if_without_changes=stop_if_without_changes,
            number_of_iterations_without_changes=number_of_iterations_without_changes,
            use_elitism=use_elitism,
            use_visualization=use_visualization
            )
        result = self.__get_solution_from_best_individual(best_individual, objects)
        return result, iterations_count

    def __get_solution_from_best_individual(self, best_individual: list[int], objects: list[tuple[int, int]]) -> tuple[list[int], int, int]:
        sum_value = 0
        sum_weight = 0
        for i, presence in enumerate(best_individual):
            if presence:
                current_value, current_weight = objects[i]
                sum_value = sum_value + current_value
                sum_weight = sum_weight + current_weight
        return best_individual, sum_value, sum_weight


    # Функции начальной инициализации
    def __get_initial_population(self, number_of_objects: int, num_random_individuals: int, num_greedy_individuals: int, task_conditions: tuple[list[tuple[int, int]], int]) -> list[list[int]]:
        initial_population = []
        if num_random_individuals > 0:
            for _ in range(num_random_individuals):
                new_individual = [random.randint(0, 1) for _ in range(number_of_objects)]
                initial_population.append(new_individual)
        if num_greedy_individuals > 0:
            greedy_solver = KnapsackGreedy()
            new_individual = greedy_solver.get_solution(*task_conditions)[0]
            for _ in range(num_greedy_individuals):
                initial_population.append(new_individual)

        return initial_population

    # Фитнесс функции

    def __simple_fitness_evaluation(self, population: list[list[int]], objects: list[tuple[int, int]], capacity: int) -> list[int]:
        fitness_scores = []
        for individual in population:
            scores = 0
            free_space = capacity
            for i, presence in enumerate(individual):
                if presence:
                    current_value, current_weight = objects[i]
                    scores = scores + current_value
                    free_space = free_space - current_weight
                    if free_space < 0:
                        scores = 0
                        break

            fitness_scores.append(scores)

        return fitness_scores

    def __fitness_evaluation_without_zeroing_out(self, population: list[list[int]], objects: list[tuple[int, int]], capacity: int)-> list[int]:
        fitness_scores = []
        for individual in population:
            sum_weight = sum(y[1] for x, y in zip(individual, objects) if x == 1)
            while sum_weight > capacity:
                index_to_replace = random.choice([i for i, including in enumerate(individual) if including == 1])
                individual[index_to_replace] = 0
                sum_weight = sum(y[1] for x, y in zip(individual, objects) if x == 1)

            sum_value = sum(y[0] for x, y in zip(individual, objects) if x == 1)
            fitness_scores.append(sum_value)

        return fitness_scores


    # Функции скрещивания

    def __single_point_crossover(self, first_parent: list[int], second_parent:list[int]) -> tuple[list[int], list[int]]:
        crossover_point = random.randint(0, len(first_parent) - 1)

        first_child = first_parent[:crossover_point] + second_parent[crossover_point:]
        second_child = second_parent[:crossover_point] + first_parent[crossover_point:]

        return first_child, second_child

    def __greedy_crossover(self, first_parent: list[int], second_parent: list[int]) -> tuple[list[int], list[int]]:
        new_individual = [presence_first_parent or presence_second_parent for presence_first_parent, presence_second_parent in zip(first_parent, second_parent)]
        existing_objects_indexes = [index for index, presence_new_individual in enumerate(new_individual) if presence_new_individual == 1]
        objects_for_greedy = [self.objects[index] for index in existing_objects_indexes]
        greedy_solver = KnapsackGreedy()
        greedy_solution = greedy_solver.get_solution(objects_for_greedy, self.capacity)[0]
        for index, value in enumerate(greedy_solution):
            new_individual[existing_objects_indexes[index]] = value

        return new_individual, new_individual

    # Функции мутации

    def __each_gene_mutation(self, individual: list[int], mutation_probability: float) -> list[int]:
        probably_mutated_individual = individual.copy()
        if mutation_probability > 0:
            for i, presence in enumerate(probably_mutated_individual):
                random_value = random.random()
                if random_value <= mutation_probability:
                    probably_mutated_individual[i] = presence ^ 1

        return probably_mutated_individual

    def __one_gene_mutation(self, individual: list[int], mutation_probability: float) -> list[int]:
        probably_mutated_individual = individual.copy()
        if mutation_probability > 0:
            random_value = random.random()
            if random_value <= mutation_probability:
                random_gene_index = random.randint(0, len(probably_mutated_individual) - 1)
                probably_mutated_individual[random_gene_index] = probably_mutated_individual[random_gene_index] ^ 1

        return probably_mutated_individual


    # Функции представления результата в виде строки

    def solution_to_string(self, solution: tuple[tuple[list[int], int, int], int]) -> str:
        (best_individual, sum_value, sum_weight), iterations_count = solution
        result = (f"\nGenetic algorithm results\n"
                  f"Objects: {best_individual}\n"
                  f"Sum value: {sum_value} Sum weight: {sum_weight}\n"
                  f"Number of iterations: {iterations_count}")
        return result
