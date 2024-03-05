from scipy import stats

from shared.shape_utils import *
from shared.parser import *

from metaheuristic_algorithms.simulated_annealing import *

from placement_algorithms.deterministic_heuristic import *
from placement_algorithms.constructive_heuristic import *
from placement_algorithms.touching_perimiter_heuristic import *

def plot_all_runs_filling_rates(all_runs_filling_rates):
    num_plots = len(all_runs_filling_rates)
    cols = 2  # Maximum of 2 runs per column
    rows = (num_plots + 1) // cols  # Calculate rows needed, round up if necessary
    
    fig, axs = plt.subplots(rows, cols, figsize=(10, rows * 4), squeeze=False)  # Adjust size as needed and ensure axs is always 2D array

    for i, filling_rates in enumerate(all_runs_filling_rates):
        row, col = divmod(i, cols)
        ax = axs[row, col]
        
        # Calculate the mode, use the first mode if multiple modes exist
        mode_val = stats.mode(filling_rates, nan_policy='omit')[0][0]
        
        # Plot each point, coloring based on its value relative to the mode
        colors = ['red' if rate < mode_val else 'blue' for rate in filling_rates]
        ax.scatter(range(len(filling_rates)), filling_rates, color=colors)
        
        ax.set_title(f'{i+1}')

    # Hide any unused subplots
    for j in range(i+1, rows*cols):
        fig.delaxes(axs.flatten()[j])
    
    plt.tight_layout()
    plt.show()
    
    # Hide any unused subplots
    for j in range(i+1, rows*cols):
        fig.delaxes(axs.flatten()[j])
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    
    knapsack, items = parse_problem_instance("./tests/HT-2001/8.json") 

    all_runs_filling_rates_from_random_selection = []
    all_runs_filling_rates = []
    
    sa = SA(knapsack, items, 1, 0.001, 0.9, touching_perimiter_heuristic, 1000, 0.99)
    layout, score = sa.execute()
    all_runs_filling_rates_from_random_selection.append(sa.filling_rates_from_random_selection)

    sa = SA(knapsack, items, 100, 0.001, 0.9, touching_perimiter_heuristic, 1000, 0.99)
    layout, score = sa.execute()
    all_runs_filling_rates_from_random_selection.append(sa.filling_rates_from_random_selection)

    sa = SA(knapsack, items, 200, 0.001, 0.9, touching_perimiter_heuristic, 1000, 0.99)
    layout, score = sa.execute()
    all_runs_filling_rates_from_random_selection.append(sa.filling_rates_from_random_selection)

    sa = SA(knapsack, items, 1000, 0.001, 0.9, touching_perimiter_heuristic, 1000, 0.99)
    layout, score = sa.execute()
    all_runs_filling_rates_from_random_selection.append(sa.filling_rates_from_random_selection)

    plot_all_runs_filling_rates(all_runs_filling_rates_from_random_selection)

    sa = SA(knapsack, items, 200, 0.001, 0.9, touching_perimiter_heuristic, 1000, 0.99)
    layout, score = sa.execute()
    all_runs_filling_rates.append(sa.filling_rates)

    sa = SA(knapsack, items, 200, 0.5, 0.9, touching_perimiter_heuristic, 1000, 0.99)
    layout, score = sa.execute()
    all_runs_filling_rates.append(sa.filling_rates)

    plot_all_runs_filling_rates(all_runs_filling_rates)

