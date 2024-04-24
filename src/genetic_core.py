import random
import matplotlib.pyplot as plt


class GeneticCore:

    def get_best_individual(self,
                            initial_population_function,
                            initial_population_args,
                            fitness_evaluation_function,
                            fitness_evaluation_args,
                            crossover_function,
                            crossover_probability,
                            mutation_function,
                            mutation_probability,
                            number_of_iterations,
                            number_of_iterations_without_changes,
                            stop_if_without_changes,
                            use_elitism,
                            use_visualization
                            ):
        self.__crossover_function = crossover_function
        self.__crossover_probability = crossover_probability
        self.__mutation_function = mutation_function
        self.__mutation_probability = mutation_probability
        self.__use_elitism = use_elitism

        if use_visualization:
            self.__fitness_values = []

        self.__population = initial_population_function(*initial_population_args)

        iteration = 0
        fitness_scores = fitness_evaluation_function(self.__population, *fitness_evaluation_args)

        if stop_if_without_changes:
            iterations_without_changes = 0
            current_max = max(fitness_scores)
            while iteration < number_of_iterations and iterations_without_changes < number_of_iterations_without_changes:
                self.__population = self.__get_new_population(fitness_scores)
                fitness_scores = fitness_evaluation_function(self.__population, *fitness_evaluation_args)
                iteration = iteration + 1

                new_max = max(fitness_scores)

                if use_visualization:
                    self.__fitness_values.append(new_max)

                if new_max == current_max:
                    iterations_without_changes = iterations_without_changes + 1
                else:
                    current_max = new_max
                    iterations_without_changes = 0
            if iterations_without_changes == number_of_iterations_without_changes:
                iteration = iteration - iterations_without_changes
        else:
            while iteration < number_of_iterations:
                self.__population = self.__get_new_population(fitness_scores)
                fitness_scores = fitness_evaluation_function(self.__population, *fitness_evaluation_args)
                iteration = iteration + 1

                if use_visualization:
                    new_max = max(fitness_scores)
                    self.__fitness_values.append(new_max)

        best_individual_index = fitness_scores.index(max(fitness_scores))
        best_individual = self.__population[best_individual_index]

        if use_visualization:
            self.__build_fitness_values_plot()

        return best_individual, iteration

    def __get_new_population(self, fitness_scores):
        new_population = []
        size_of_population = len(self.__population)

        if self.__use_elitism:
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
            if random_value <= self.__crossover_probability:
                new_individuals = self.__crossover_function(*selected_individuals)
            for individual in new_individuals:
                probably_mutated_individual = self.__mutation_function(individual, self.__mutation_probability)
                new_population.append(probably_mutated_individual)

        return new_population

    def __get_intervals(self, probabilities):
        intervals = []
        cumulative_sum = 0

        for probability in probabilities:
            cumulative_sum = cumulative_sum + probability
            intervals.append(cumulative_sum)

        return intervals

    def __get_random_pair_of_individuals(self, intervals):
        selected_individuals = []
        for _ in range(2):

            random_value = random.random()

            for i, interval in enumerate(intervals):
                if random_value <= interval:
                    selected_individuals.append(self.__population[i])
                    break

        return selected_individuals

    def __build_fitness_values_plot(self):
        iterations = range(1, len(self.__fitness_values) + 1)

        plt.plot(iterations, self.__fitness_values, marker='o', linestyle='-')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness values')
        plt.title('Fitness function plot by iterations')
        plt.grid(True)
        plt.show()

