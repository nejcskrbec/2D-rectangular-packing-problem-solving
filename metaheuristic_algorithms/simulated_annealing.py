import concurrent.futures
import random as rd
import math as math
from shared.knapsack import *
from shared.item import *

def swap(list):
    if len(list) < 2:
        return copy.deepcopy(list)
    index_1, index_2 = rd.sample(range(len(list)), 2)
    new_lst = copy.deepcopy(list)
    new_lst[index_1], new_lst[index_2] = new_lst[index_2], new_lst[index_1]
    return new_lst


def SA(knapsack, items, initial_temperature, final_temperature, alpha, max_iterations, method):
    M = sorted(items, key=lambda item: item.shape.area, reverse=True)
    max_layout, max_filling_rate = method(knapsack, M)
    temperature = initial_temperature
    iterations = 0

    while temperature > final_temperature:
        if max_filling_rate == knapsack.shape.area or iterations > max_iterations:
            break

        M_swapped = swap(M)
        layout, filling_rate = method(knapsack, M_swapped)

        if filling_rate > max_filling_rate:
            max_filling_rate = filling_rate
            max_layout = layout
            M = M_swapped
        else:
            delta = max_filling_rate - filling_rate
            random_number = rd.random()
            if math.exp(-delta / temperature) > random_number:
                max_filling_rate = filling_rate
                max_layout = layout
                M = M_swapped

        temperature *= alpha
        iterations += 1

    return max_layout, max_filling_rate

def SA_optimized(knapsack, items, initial_temperature, final_temperature, alpha, max_iterations, method):
    M = sorted(items, key=lambda item: item.shape.area, reverse=True)
    max_layout, max_filling_rate = method(knapsack, M)
    temperature = initial_temperature
    iterations = 0

    with concurrent.futures.ProcessPoolExecutor() as executor:
        while temperature > final_temperature:
            if max_filling_rate == knapsack.shape.area or iterations > max_iterations:
                break

            M_swapped = swap(M)
            future = executor.submit(method, knapsack, M_swapped)
            layout, filling_rate = future.result()

            if filling_rate > max_filling_rate:
                max_filling_rate = filling_rate
                max_layout = layout
                M = M_swapped
            else:
                delta = max_filling_rate - filling_rate
                random_number = rd.random()
                if math.exp(-delta / temperature) > random_number:
                    max_filling_rate = filling_rate
                    max_layout = layout
                    M = M_swapped

            temperature *= alpha
            iterations += 1

    knapsack.items = max_layout
    return max_layout, max_filling_rate




