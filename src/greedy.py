
class KnapsackGreedy:

    def get_solution(self, objects: [(int, int)], capacity: int):
        relative_values = self.__get_relative_values(objects)
        sorted_values = sorted(relative_values, key=lambda x: x[1], reverse=True)
        result = self.__get_greedy_solution(objects, sorted_values, capacity)
        return result

    def __get_relative_values(self, objects: [(int, int)]):
        result = []
        for i, (value, weight) in enumerate(objects):
            result.append((i, value/weight))
        return result

    def __get_greedy_solution(self, objects, sorted_values, capacity):
        objects_presence = [0] * len(sorted_values)
        free_space = capacity
        sum_value = 0
        for i, _ in sorted_values:
            current_value, current_weight = objects[i]
            remainder = free_space - current_weight
            if remainder >= 0:
                sum_value = sum_value + current_value
                objects_presence[i] = 1
                free_space = remainder

                if remainder == 0:
                    break

        sum_weight = capacity - free_space
        return objects_presence, sum_value, sum_weight

    def solution_to_string(self, solution):
        objects_presence, sum_value, sum_weight = solution
        result = (f"\nGreedy algorithm results\n"
                  f"Objects: {objects_presence}\n"
                  f"Sum value: {sum_value} Sum weight: {sum_weight}")
        return result



