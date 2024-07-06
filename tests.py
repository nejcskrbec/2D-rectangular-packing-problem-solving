import os
import re
import time
import csv
import pandas as pd

from shared.shape_utils import *
from shared.parser import *

from placement_algorithms.deterministic_heuristic import *
from placement_algorithms.constructive_heuristic import *
from placement_algorithms.touching_perimeter_heuristic import *

from metaheuristic_algorithms.simulated_annealing import *
from metaheuristic_algorithms.evolutionary import *


def save_tests_data_to_csv(filenames, directory, output_csv='tests.csv'):
    # Prepare the CSV file for writing
    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Ime problema', 'Vsebnik (w, h)', 'Število predmetov',
                         'Povprečna širina', 'Povprečna višina', 'Povprečna površina'])

        # Extract names and sort
        test_problems = [(filename[:-5], filename) for filename in filenames]
        test_problems.sort()  # Sort by test problem name (without .json suffix)

        # Process each filename in sorted order
        for test_problem_name, filename in test_problems:
            knapsack, items = parse_problem_instance(f"{directory}/{filename}")
            
            # Compute necessary metrics
            num_items = len(items)
            total_surface = 0
            total_width = 0
            total_height = 0
            
            for item in items:
                width, height = item.width, item.height
                surface = width * height
                total_surface += surface
                total_width += width
                total_height += height
                
            average_surface = total_surface / num_items if num_items else 0
            average_width = total_width / num_items if num_items else 0
            average_height = total_height / num_items if num_items else 0

            # Write row data
            writer.writerow([
                test_problem_name,
                f"({knapsack.width}, {knapsack.height})",
                num_items,
                f"{average_width:.2f}",
                f"{average_height:.2f}",
                f"{average_surface:.2f}"
            ])


def save_execution_data_to_csv(execution_times, target_directory):
    csv_file_path = os.path.join(target_directory, "execution_times.csv")
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Problem Instance", "Execution Time (s)", "Number of Items", "Filling Rate"])
        writer.writerows(execution_times)


def plot_execution_times(execution_times, target_directory):
    plt.figure(figsize=(10, 6))
    instances = [f"{et[0]}" for et in execution_times]
    times = [et[1] for et in execution_times]
    plt.bar(instances, times, color='#93bde5')
    plt.xlabel('Primer')
    plt.ylabel('Čas izvajanja (s)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(target_directory, "execution_times_plot.png"))
    plt.close()

def process_and_plot_data(groups, target_directory):
    """Process and plot data for execution times and filling rates from specified groups, where each group name represents an algorithm."""
    group_data = {key: [] for key in groups}  # Prepare data structure based on provided groups
    all_data_records = []

    # Ensure the target directory exists
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)

    # Read and organize data
    for group_name, directories in groups.items():
        # List to store DataFrames temporarily for this group
        temp_data_frames = []
        for directory in directories:
            filepath = os.path.join(directory, 'execution_times.csv')
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                df['Algorithm'] = os.path.basename(directory).split('_')[3]  # Extract algorithm type from directory name
                temp_data_frames.append(df)
            else:
                print(f"Warning: CSV file not found in directory {directory}")

        # Concatenate all data frames in the list for the current group, if not empty
        if temp_data_frames:
            all_data = pd.concat(temp_data_frames, ignore_index=True)
            first_column_name = all_data.columns[0]
            grouped_data = all_data.groupby(first_column_name)
            subdataframes = {name: group.reset_index(drop=True) for name, group in grouped_data}
            sorted_subdataframes = {key: subdataframes[key] for key in sorted(subdataframes.keys())}
            group_data[group_name].append(list(sorted_subdataframes.values()))

            for df in all_data.itertuples(index=False):
                all_data_records.append([
                    getattr(df, 'Problem_Instance', None),
                    getattr(df, 'Number_of_Items', None),
                    group_name,
                    getattr(df, 'Execution_Time_s', None),
                    getattr(df, 'Filling_Rate', None)
                ])

    bar_colors = ['#d8e8f6','#94bee5', '#81a6c8']

    def extract_number(s):
        """ Extract the first sequence of digits from a string. """
        match = re.search(r'\d+', s)  # This regex matches any sequence of digits in the string
        return int(match.group()) if match else float('inf')  # Convert the match to an integer or return infinity if no match

    def plot_execution_times():
        """Plot filling rates for each algorithm group."""
        for group_name, data in group_data.items():
            if not data:
                print(f"No data to plot for group {group_name}. Check if CSV files are present and correctly formatted.")
                continue

            plt.figure(figsize=(10, 6))
            bar_positions = []
            label_positions = []
            heights = []
            colors = []
            labels = []

            sorted_data = sorted(data[0], key=lambda df: df.iloc[0, 2])
            for problem_results in sorted_data:
                current_position = len(bar_positions)
                problem_data = problem_results.iterrows()
                for index, row in problem_data:
                    bar_positions.append(current_position + index)
                    heights.append(row['Execution Time (s)'])
                    colors.append(bar_colors[index])

                    if index == 1:
                        label_positions.append(current_position + index)
                        labels.append(f"{row['Problem Instance']}")

                # Append additional space after each group
                bar_positions.append(current_position + len(problem_results))
                colors.append('#93bde5')  # Default color for spacing
                heights.append(0)  # Zero height for spacing

            plt.bar(bar_positions, heights, color=colors)
            plt.xticks(label_positions, labels, rotation=45)
            plt.xlabel('Primer')
            plt.ylabel('Čas izvajanja (s))')
            plt.title(f'Čas izvajanja - {group_name}')
            plt.tight_layout()

            color_patches = [
                mpatches.Patch(color=bar_colors[0], label=f'{group_name}'),
                mpatches.Patch(color=bar_colors[1], label=f'{group_name} in SO'),
                mpatches.Patch(color=bar_colors[2], label=f'{group_name} in GA')
            ]
            plt.legend(handles=color_patches)
            
            plt.savefig(os.path.join(target_directory, f'{group_name}_execution_time_plot.png'))
            plt.close()

    def plot_filling_rates():
        """Plot filling rates for each algorithm group."""
        for group_name, data in group_data.items():
            if not data:
                print(f"No data to plot for group {group_name}. Check if CSV files are present and correctly formatted.")
                continue

            plt.figure(figsize=(10, 6))
            bar_positions = []
            label_positions = []
            heights = []
            colors = []
            labels = []

            sorted_data = sorted(data[0], key=lambda df: extract_number(df.iloc[0, 0]))
            for problem_results in sorted_data:
                current_position = len(bar_positions)
                problem_data = problem_results.iterrows()
                for index, row in problem_data:
                    bar_positions.append(current_position + index)
                    heights.append(row['Filling Rate'] * 100)
                    colors.append(bar_colors[index])

                    if index == 1:
                        label_positions.append(current_position + index)
                        labels.append(f"{row['Problem Instance']}")

                # Append additional space after each group
                bar_positions.append(current_position + len(problem_results))
                colors.append('#93bde5')  # Default color for spacing
                heights.append(0)  # Zero height for spacing

            plt.bar(bar_positions, heights, color=colors)
            plt.xticks(label_positions, labels, rotation=45)
            plt.xlabel('Primer')
            plt.ylabel('Zapolnjenost (%)')
            plt.ylim(0, 120)
            plt.title(f'Zapolnjenost - {group_name}')
            plt.legend(bar_colors, ['Algoritem vstavljanja', 'SO', 'GA'])
            plt.tight_layout()

            color_patches = [
                mpatches.Patch(color=bar_colors[0], label=f'{group_name}'),
                mpatches.Patch(color=bar_colors[1], label=f'{group_name} in SO'),
                mpatches.Patch(color=bar_colors[2], label=f'{group_name} in GA')
            ]
            plt.legend(handles=color_patches)

            plt.savefig(os.path.join(target_directory, f'{group_name}_filling_rate_plot.png'))
            plt.close()

    def save_comprehensive_csv(data_records, target_directory, groups):
        csv_file_path = os.path.join(target_directory, "all_data.csv")
        headers = ["Problem Instance", "Number of Items"]
        for group in groups:
            headers.extend([f"{group} Execution Time (s)", f"{group} Filling Rate"])
        
        with open(csv_file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)
            for record in data_records:
                writer.writerow(record)

    save_comprehensive_csv(all_data_records, target_directory, groups)
    plot_execution_times()
    plot_filling_rates()

class TestPA:
    def __init__(self, test_directory, test_filenames, target_directory, placement_algorithm):
        self.test_directory = test_directory
        self.test_filenames = test_filenames
        self.target_directory = target_directory
        self.placement_algorithm = placement_algorithm

    def execute(self):
        # Check if the directory exists; if not, create it
        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)

        knapsacks_with_items = []
        execution_data = []
        titles = []

        for filename in self.test_filenames:
            start_time = time.time()

            knapsack, items = parse_problem_instance(f"{self.test_directory}/{filename}")
            layout, score = self.placement_algorithm(knapsack, items)
            print(f"Finished processing {filename} with filling rate: {score * 100} %")

            end_time = time.time()

            title = os.path.splitext(filename)[0]
            titles.append(title)
            execution_data.append([title, end_time - start_time, len(items), score])
            knapsacks_with_items.append((knapsack, layout))

        # Sort execution_times by the number of items
        execution_data.sort(key=lambda x: x[2])

        results_plot = visualize_knapsacks_and_items(knapsacks_with_items, titles)
        results_plot.savefig(f"{self.target_directory}/layouts.png")

        save_execution_data_to_csv(execution_data, self.target_directory)
        plot_execution_times(execution_data, self.target_directory)

class TestGOA:
    def __init__(self, test_directory, test_filenames, target_directory, global_optimization_algorithm_instance):
        self.test_directory = test_directory
        self.test_filenames = test_filenames
        self.target_directory = target_directory
        self.global_optimization_algorithm_instance = global_optimization_algorithm_instance

    def execute(self):
        # Check if the directory exists; if not, create it
        if not os.path.exists(self.target_directory):
            os.makedirs(self.target_directory)

        knapsacks_with_items = []
        execution_data = []
        titles = []

        for filename in self.test_filenames:
            start_time = time.time()

            knapsack, items = parse_problem_instance(f"{self.test_directory}/{filename}")
            self.global_optimization_algorithm_instance.knapsack = knapsack
            self.global_optimization_algorithm_instance.items = items
            layout, score = self.global_optimization_algorithm_instance.execute()
            print(f"Finished processing {filename} with filling rate: {score * 100} %")

            end_time = time.time()

            title = os.path.splitext(filename)[0]
            titles.append(title)
            execution_data.append([title, end_time - start_time, len(items), score])
            knapsacks_with_items.append((knapsack, layout))

        # Sort execution_times by the number of items
        execution_data.sort(key=lambda x: x[2])

        results_plot = visualize_knapsacks_and_items(knapsacks_with_items, titles)
        results_plot.savefig(f"{self.target_directory}/layouts.png")

        save_execution_data_to_csv(execution_data, self.target_directory)
        plot_execution_times(execution_data, self.target_directory)

if __name__ == '__main__':

    test_filenames = [
        "GCUT-1.json", "GCUT-2.json", "GCUT-3.json", "GCUT-5.json",
        "B-1.json", "B-2.json", "B-3.json", "B-5.json",
        "C-1.json", "C-2.json", "C-3.json", "C-5.json",
        "HT-1.json", "HT-4.json", "HT-8.json", "HT-10.json",
        "OKP-1.json", "OKP-2.json", "OKP-3.json", "OKP-4.json",
    ]

    save_tests_data_to_csv(test_filenames, 'tests/', '11_test_common_results/tests.csv')

    # local_best_fit_first_test = TestPA(
    #     test_directory="tests/",
    #     test_filenames=test_filenames,
    #     target_directory="01_test_local_best_fit_first",
    #     placement_algorithm=local_best_fit_first
    # )
    # local_best_fit_first_test.execute()

    # constructive_heuristic_test = TestPA(
    #     test_directory="tests/",
    #     test_filenames=test_filenames,
    #     target_directory="02_test_constructive_heuristic",
    #     placement_algorithm=constructive_heuristic
    # )
    # constructive_heuristic_test.execute()

    # touching_perimeter_heuristic_heuristic_test = TestPA(
    #     test_directory="tests/",
    #     test_filenames=test_filenames,
    #     target_directory="03_test_touching_perimeter_heuristic",
    #     placement_algorithm=touching_perimeter_heuristic
    # )
    # touching_perimeter_heuristic_heuristic_test.execute()

    # SA_local_best_fit_first_test = TestGOA(
    #     test_directory="tests/",
    #     test_filenames=test_filenames,
    #     target_directory="04_test_SA_local_best_fit_first",
    #     global_optimization_algorithm_instance=SA(1000, 0.01, 0.8, local_best_fit_first, 10000, 1)
    # )
    # SA_local_best_fit_first_test.execute()

    # SA_constructive_heuristic_test = TestGOA(
    #     test_directory="tests/",
    #     test_filenames=test_filenames,
    #     target_directory="05_test_SA_constructive_heuristic",
    #     global_optimization_algorithm_instance=SA(1000, 0.01, 0.8, constructive_heuristic, 10000, 1)
    # )
    # SA_constructive_heuristic_test.execute()

    # SA_touching_perimeter_heuristic_test = TestGOA(
    #     test_directory="tests/",
    #     test_filenames=test_filenames,
    #     target_directory="06_test_SA_touching_perimeter_heuristic",
    #     global_optimization_algorithm_instance=SA(1000, 0.01, 0.8, touching_perimeter_heuristic, 10000, 1)
    # )
    # SA_touching_perimeter_heuristic_test.execute()

    # GA_local_best_fit_first_test = TestGOA(
    #     test_directory="tests/",
    #     test_filenames=test_filenames,
    #     target_directory="07_test_GA_local_best_fit_first",
    #     global_optimization_algorithm_instance=GA(30, num_of_items_fitness, roulette_wheel_selection_with_linear_scaling, ox3_crossover_reproduction, m1_m3_mutation, 0.05, local_best_fit_first, 1, 100)
    # )
    # GA_local_best_fit_first_test.execute()

    GA_constructive_heuristic_test = TestGOA(
        test_directory="tests/",
        test_filenames=test_filenames,
        target_directory="08_test_GA_constructive_heuristic",
        global_optimization_algorithm_instance=GA(30, num_of_items_fitness, roulette_wheel_selection_with_linear_scaling, ox3_crossover_reproduction, m1_m3_mutation, 0.1, constructive_heuristic, 1, 100)
    )
    GA_constructive_heuristic_test.execute()

    # GA_touching_perimeter_heuristic_test = TestGOA(
    #     test_directory="tests/",
    #     test_filenames=test_filenames,
    #     target_directory="09_test_GA_touching_perimeter_heuristic",
    #     global_optimization_algorithm_instance=GA(30, num_of_items_fitness, roulette_wheel_selection_with_linear_scaling, ox3_crossover_reproduction, m1_m3_mutation, 0.1, touching_perimeter_heuristic, 1, 100)
    # )
    # GA_touching_perimeter_heuristic_test.execute()

    groups = {
        "SNP": [
            "01_test_local_best_fit_first",
            "04_test_SA_local_best_fit_first",
            "07_test_GA_local_best_fit_first"
        ],
        "KHA": [
            "02_test_constructive_heuristic",
            "05_test_SA_constructive_heuristic",
            "08_test_GA_constructive_heuristic"
        ],
        "HSO": [
            "03_test_touching_perimeter_heuristic",
            "06_test_SA_touching_perimeter_heuristic",
            "09_test_GA_touching_perimeter_heuristic"
        ]
    }

    target_directory = '11_test_common_results'

    process_and_plot_data(groups, target_directory)
