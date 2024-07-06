from shared.shape_utils import *
from shared.parser import *

from metaheuristic_algorithms.simulated_annealing import *
from metaheuristic_algorithms.evolutionary import *

from placement_algorithms.py.deterministic_heuristic import *
from placement_algorithms.py.constructive_heuristic import *
from placement_algorithms.py.touching_perimeter_heuristic import *

if __name__ == '__main__':
    
    knapsack, items = parse_problem_instance("./tests/GCUT-1.json") 

    GA = GA(500, basic_fitness, roulette_wheel_selection_with_linear_scaling, ox3_crossover_reproduction, m1_m3_mutation, 0.01, touching_perimeter_heuristic, 0.97, 50)
    GA.knapsack = knapsack
    GA.items = items
    layout, score = GA.execute()
    visualize_knapsack_and_items(knapsack, layout, "Touching perimeter heuristic")


    layout, score = SA(knapsack, items, 10, 0.001, 0.9, touching_perimeter_heuristic, 1000, 0.9).execute()
    visualize_knapsack_and_items(knapsack, layout, "Touching perimeter heuristic")


