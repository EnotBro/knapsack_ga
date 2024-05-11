# Задача о рюкзаке (Knapstack problem)
import string
import os
from pathlib import Path
from greedy import KnapsackGreedy
from genetic import KnapsackGenetic
import random


TASK_CONDITIONS_FILE_PATH = "task_conditions3.txt"
# task_conditions1 110101 33 37 включения/ценность/вес
# task_conditions2 0001011101 82 57 включения/ценность/вес
# task_conditions3 111111111111011 40 50 включения/ценность/вес

USE_TASK_CONDITIONS_FROM_FILE = False


def main() -> None:
    random.seed(100)

    task_conditions = None
    if USE_TASK_CONDITIONS_FROM_FILE:
        task_conditions = read_task_conditions_from_file(TASK_CONDITIONS_FILE_PATH)
    else:
        task_conditions = get_random_task_conditions(50, 5, 20, 1, 14, 36)
    print("Task Conditions")
    print(task_conditions)

    greedy_solver = KnapsackGreedy()
    greedy_result = greedy_solver.get_solution(*task_conditions)
    print(greedy_solver.solution_to_string(greedy_result))

    genetic_solver = KnapsackGenetic(
        initial_population_function="get_initial_population",
        fitness_evaluation_function="fitness_evaluation_without_zeroing_out",
        crossover_function="single_point_crossover",
        mutation_function="each_gene_mutation",
        number_of_random_initial_individuals=100,
        number_of_greedy_initial_individuals=0,
        crossover_probability=0.85,
        mutation_probability=0.1,
        number_of_iterations=2000,
        stop_if_without_changes=False,
        number_of_iterations_without_changes=150,
        use_elitism=True,
        use_visualization=True,
        use_correction_after_each_step=False
    )
    genetic_result = genetic_solver.get_solution(*task_conditions)
    print(genetic_solver.solution_to_string(genetic_result))


def read_task_conditions_from_file(filename: string) -> tuple[list, int]:
    max_weight = 0
    objects = []
    with open(os.path.join(Path(__file__).resolve().parent.parent, filename), 'r') as file:
        max_weight = int(file.readline().strip())

        for line in file:
            values = line.strip().split()
            objects.append((int(values[0]), int(values[1]))) # (ценность, вес)
    return objects, max_weight


def get_random_task_conditions(number_of_objects: int, min_value: int, max_value: int, min_weight: int, max_weight: int,
                               capacity: int) -> tuple[list[tuple[int, int]], int]:
    objects = [(random.randint(min_value, max_value), random.randint(min_weight, max_weight)) for _ in range(number_of_objects)]
    return objects, capacity


if __name__ == '__main__':
    main()
