import concurrent.futures
import random as rd
import math as math

from shared.knapsack import *
from shared.item import *

def swap(list):
    if len(list) < 2:
        return cp.deepcopy(list)
    index_1, index_2 = rd.sample(range(len(list)), 2)
    new_lst = cp.deepcopy(list)
    new_lst[index_1], new_lst[index_2] = new_lst[index_2], new_lst[index_1]
    return new_lst

class SA():
    def __init__(self, knapsack, items, initial_temperature, final_temperature, alpha, placement_function, max_iterations, stop_filling_rate):
        self.knapsack = cp.deepcopy(knapsack)
        self.items = cp.deepcopy(items)
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.alpha = alpha
        self.placement_function = placement_function
        self.max_iterations = max_iterations
        self.stop_filling_rate = stop_filling_rate
        self.filling_rates_from_random_selection = []
        self.filling_rates = []

    def execute(self):
        M = self.items
        max_layout, max_filling_rate = self.placement_function(self.knapsack, M)
        self.filling_rates.append(max_filling_rate)

        temperature = self.initial_temperature
        iterations = 0

        with concurrent.futures.ProcessPoolExecutor() as executor:
            while temperature > self.final_temperature:
                if max_filling_rate >= self.stop_filling_rate or iterations > self.max_iterations:
                    break

                M_swapped = swap(M)
                future = executor.submit(self.placement_function, self.knapsack, M_swapped)
                layout, filling_rate = future.result()

                if filling_rate > max_filling_rate:
                    max_filling_rate = filling_rate
                    max_layout = layout
                    M = M_swapped
                else:
                    delta = max_filling_rate - filling_rate
                    random_number = rd.random()
                    if math.exp(-delta / temperature) > random_number:
                        M = M_swapped
                        self.filling_rates_from_random_selection.append(filling_rate)  # Store the filling rate

                temperature *= self.alpha
                iterations += 1

                self.filling_rates.append(filling_rate)

        self.knapsack.items = max_layout
        return max_layout, max_filling_rate
