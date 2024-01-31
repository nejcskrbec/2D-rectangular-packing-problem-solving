from shared.shape_utils import *
from shared.parser import *

from metaheuristic_algorithms.simulated_annealing import *
from placement_algorithms.deterministic_heuristic import *
from placement_algorithms.constructive_heuristic import *

if __name__ == '__main__':
    
    # Example usage:
    knapsack = Knapsack(8, 6)

    items = [
        Item(width=2, height=4),
        Item(width=1, height=5),
        Item(width=3, height=2),
        Item(width=5, height=3),
        Item(width=3, height=2),
        Item(width=3, height=3),
    ]

    knapsack, items = parse_problem_instance("./JSON/1.json")

    layout, score = BFHA_local_optimized(knapsack, items)
    visualize_knapsack_and_items(knapsack, layout, "BFHA local")

    layout, score = constructive_heuristic(knapsack, items)
    visualize_knapsack_and_items(knapsack, layout, "Constructive heuristic")

    layout, score = SA_optimized(knapsack, items, 100, 0.001, 0.9, 1000, constructive_heuristic)
    visualize_knapsack_and_items(knapsack, layout, "Simulated annealing")

