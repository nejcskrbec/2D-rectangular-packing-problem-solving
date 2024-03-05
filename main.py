from shared.shape_utils import *
from shared.parser import *

from metaheuristic_algorithms.simulated_annealing import *
from metaheuristic_algorithms.evolutionary import *

from placement_algorithms.deterministic_heuristic import *
from placement_algorithms.constructive_heuristic import *
from placement_algorithms.touching_perimiter_heuristic import *

if __name__ == '__main__':
    
    knapsack, items = parse_problem_instance("./tests/HT-2001/1.json") 

    layout, score = GA(knapsack, items, 20, basic_fitness, roulette_wheel_selection_with_linear_scaling, ox3_crossover_reproduction, m1_m3_mutation, 0.01, touching_perimiter_heuristic, 0.97, 50).execute()
    visualize_knapsack_and_items(knapsack, layout, "Genetic algorithm")

    layout, score = SA(knapsack, items, 10, 0.001, 0.9, touching_perimiter_heuristic, 1000, 0.9).execute()
    visualize_knapsack_and_items(knapsack, layout, "Touching perimeter heuristic")


