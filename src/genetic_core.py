import random
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class BaseGenetic(ABC):
    def __init__(self,
                 crossover_probability: float,
                 number_of_iterations: int,
                 number_of_iterations_without_changes: int,
                 stop_if_without_changes: bool,
                 use_elitism: bool,
                 use_visualization: bool
                 ):
        self.__crossover_probability = crossover_probability
        self.__number_of_iterations = number_of_iterations
        self.__number_of_iterations_without_changes = number_of_iterations_without_changes
        self.__stop_if_without_changes = stop_if_without_changes
        self.__use_elitism = use_elitism
        self.__use_visualization = use_visualization

    @property
    def crossover_probability(self):
        return self.__crossover_probability

    @property
    def number_of_iterations(self):
        return self.__number_of_iterations

    @property
    def number_of_iterations_without_changes(self):
        return self.__number_of_iterations_without_changes

    @property
    def stop_if_without_changes(self):
        return self.__stop_if_without_changes

    @property
    def use_elitism(self):
        return self.__use_elitism

    @property
    def use_visualization(self):
        return self.__use_visualization

    @abstractmethod
    def initial_population_function(self):
        pass

    @abstractmethod
    def fitness_evaluation_function(self, population):
        pass

    @abstractmethod
    def crossover_function(self, first_parent, second_parent):
        pass

    @abstractmethod
    def mutation_function(self, individual):
        pass


class GeneticCore:

    def get_best_individual(self, genetic_task: BaseGenetic):

        self.__genetic_task = genetic_task

        if self.__genetic_task.use_visualization:
            self.__fitness_values = []

        self.__population = self.__genetic_task.initial_population_function()

        iteration = 0
        current_best_result_iteration = 0
        iterations_without_changes = 0
        fitness_scores = self.__genetic_task.fitness_evaluation_function(self.__population)
        current_max = max(fitness_scores)

        while iteration < self.__genetic_task.number_of_iterations:
            self.__population = self.__get_new_population(fitness_scores)
            fitness_scores = self.__genetic_task.fitness_evaluation_function(self.__population)
            iteration = iteration + 1

            new_max = max(fitness_scores)

            if self.__genetic_task.use_visualization:
                self.__fitness_values.append(new_max)

            if new_max == current_max:
                iterations_without_changes = iterations_without_changes + 1
            else:
                current_max = new_max
                iterations_without_changes = 0
                current_best_result_iteration = iteration

            if self.__genetic_task.stop_if_without_changes:
                if iterations_without_changes >= self.__genetic_task.number_of_iterations_without_changes:
                    break

        best_individual_index = fitness_scores.index(max(fitness_scores))
        best_individual = self.__population[best_individual_index]

        if self.__genetic_task.use_visualization:
            self.__build_fitness_values_plot()

        return best_individual, current_best_result_iteration

    def __get_new_population(self, fitness_scores: list[int]) -> list[list[int]]:
        new_population = []
        size_of_population = len(self.__population)

        if self.__genetic_task.use_elitism:
            best_individual_index = fitness_scores.index(max(fitness_scores))
            best_individual = self.__population[best_individual_index]
            for _ in range(2):
                new_population.append(best_individual)

        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        intervals = self.__get_intervals(probabilities)

        while len(new_population) < size_of_population:
            selected_individuals = self.__get_random_pair_of_individuals(intervals)
            new_individuals = selected_individuals.copy()
            random_value = random.random()
            if random_value <= self.__genetic_task.crossover_probability:
                new_individuals = self.__genetic_task.crossover_function(*selected_individuals)
            for individual in new_individuals:
                probably_mutated_individual = self.__genetic_task.mutation_function(individual)
                new_population.append(probably_mutated_individual)

        return new_population

    def __get_intervals(self, probabilities: list[float]) -> list[float]:
        intervals = []
        cumulative_sum = 0

        for probability in probabilities:
            cumulative_sum = cumulative_sum + probability
            intervals.append(cumulative_sum)

        return intervals

    def __get_random_pair_of_individuals(self, intervals: list[float]) -> list[list[int]]:
        selected_individuals = []
        for _ in range(2):

            random_value = random.random()

            for i, interval in enumerate(intervals):
                if random_value <= interval:
                    selected_individuals.append(self.__population[i])
                    break

        return selected_individuals

    def __build_fitness_values_plot(self) -> None:
        iterations = range(1, len(self.__fitness_values) + 1)

        plt.plot(iterations, self.__fitness_values, marker='o', linestyle='-')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness values')
        plt.title('Fitness function plot by iterations')
        plt.grid(True)
        plt.show()

