import random as rd
import bisect as bi
import copy as cp
import random as rd
import itertools as iter
import multiprocessing

from functools import partial
from shared.knapsack import *
from shared.item import *

class Individual:
    def __init__(self, items, knapsack, fitness_function, placement_function):
        self.items = items
        self.knapsack = knapsack
        self.placement_function = placement_function
        self.fitness_function = fitness_function
        self.layout = []
        self.score = 0
        self.fitness = 0
        self.calculated = False
        self.calculate()

    def calculate(self):
        if self.calculated is False:
            layout, score = self.placement_function(self.knapsack, self.items)
            self.layout = layout
            self.score = score
            self.fitness = self.fitness_function(self)
            self.calculated = True

def create_individual(items, knapsack, fitness_function, placement_function):
    return Individual(list(items), knapsack, fitness_function, placement_function)

class GA():
    def __init__(self, population_size, fitness_function, selection_function, crossover_function, mutation_function, mutation_rate, placement_function, stop_filling_rate, max_iterations, knapsack=None, items=None):
        self.knapsack = cp.deepcopy(knapsack) if knapsack is not None else None
        self.items = cp.deepcopy(items) if items is not None else None
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.mutation_rate = mutation_rate
        self.placement_function = placement_function
        self.stop_filling_rate = stop_filling_rate
        self.max_iterations = max_iterations
        self.current_iteration = 1
        self.generated_individuals = []
        self.population = []

    def generate_population(self, initial_population):
        items_copy = cp.deepcopy(self.items)

        existing_permutations = {tuple(individual.items) for individual in self.generated_individuals}
        permutations = set()
        for _ in range(self.population_size - len(initial_population)):
            rd.shuffle(items_copy)
            perm = tuple(items_copy)
            if perm not in existing_permutations:
                existing_permutations.add(perm)
                permutations.add(perm)

        # Use multiprocessing.Pool() for parallel execution of the constructor
        with multiprocessing.Pool() as pool:
            # Use partial to create a function with fixed arguments except for 'items'
            create_individual_partial = partial(
                create_individual,
                knapsack=self.knapsack,
                fitness_function=self.fitness_function,
                placement_function=self.placement_function
            )
            # Execute constructor in parallel for each permutation of items
            new_individuals = pool.map(create_individual_partial, permutations)

        initial_population.extend(new_individuals)
        self.generated_individuals.extend(initial_population)
        self.population = sorted(initial_population, key=lambda x: x.fitness, reverse=True)


    def merge_offspring(self, offspring):
        self.population.extend(offspring)
        self.generate_population(self.population)


    def execute(self):
        if self.knapsack is None or self.items is None:
            raise ValueError("Knapsack and items must be set before executing.")
           
        self.generate_population([])

        best_individual = max(self.population, key=lambda individual: individual.score)

        for _ in range(self.max_iterations):      
            if best_individual.score >= self.stop_filling_rate or self.current_iteration == self.max_iterations:
                break

            # Selection
            selected_individuals = self.selection_function(self.population)

            # Crossover (Reproduction)
            offspring = self.crossover_function(self, selected_individuals)

            # Mutations
            for index, individual in enumerate(offspring):
                if rd.random() < self.mutation_rate:
                    mutated_individual = self.mutation_function(individual, self)
                    self.population[index] = mutated_individual

            self.merge_offspring(offspring)

            current_best_individual = max(self.population, key=lambda individual: individual.score)
            if current_best_individual.score > best_individual.score:
                best_individual = current_best_individual

            self.current_iteration += 1

        self.current_iteration = 1
        self.generated_individuals = []
        self.population = []

        return best_individual.layout, best_individual.score


### Selection functions
NUM_COMBINATIONS_COEFFICIENT = (9/10)
def proportional_selection(population):
    selection_amount = int(NUM_COMBINATIONS_COEFFICIENT * len(population))
    total_fitness = sum(individual.fitness for individual in population)
    selection_probs = [individual.fitness / total_fitness for individual in population]

    # Create a list of cumulative probabilities
    cumulative_probs = [sum(selection_probs[:i+1]) for i in range(len(selection_probs))]

    # Select individuals based on their cumulative probability
    selected_individuals = []
    for _ in range(selection_amount):
        rand_selection = rd.random()
        # Find the individual whose cumulative probability is just above the random selection
        index = bi.bisect(cumulative_probs, rand_selection)
        selected_individuals.append(population[index])

    return selected_individuals
    
TARGET_AVERAGE=0.5
def roulette_wheel_selection_with_linear_scaling(population):
    def linear_scaling(individuals):
        fitness_values = [ind.fitness for ind in individuals]
        avg_fitness = sum(fitness_values) / len(fitness_values)
        max_fitness = max(fitness_values)

        a = (TARGET_AVERAGE * (len(fitness_values) - 1)) / (max_fitness - avg_fitness) if max_fitness != avg_fitness else avg_fitness
        b = TARGET_AVERAGE * (max_fitness - len(fitness_values) * avg_fitness) / (max_fitness - avg_fitness) if max_fitness != avg_fitness else 0

        return [max(a * f + b, 0) for f in fitness_values]  # Ensure fitness is non-negative

    def roulette_wheel_selection(scaled_fitness, individuals):
        total_fitness = sum(scaled_fitness)
        selection_probs = [f / total_fitness for f in scaled_fitness]
        cumulative_probs = [sum(selection_probs[:i+1]) for i in range(len(selection_probs))]

        selection_point = rd.uniform(0, 1)
        index = bi.bisect_left(cumulative_probs, selection_point)
        return individuals[index]

    scaled_fitness = linear_scaling(population)  # Apply linear scaling first
    # Select individuals using roulette wheel selection multiple times
    selected_individuals = list({roulette_wheel_selection(scaled_fitness, population) for _ in range(100)})
    return selected_individuals


### Crossover functions
def ox3_crossover(parent_a, parent_b, instance):
    length = len(parent_a.items)
    p, q = sorted(rd.sample(range(1, length), 2))  # Ensure p < q

    def create_offspring_items(parent_1, parent_2):
        middle = parent_1.items[p:q+1]
        start = [item for item in parent_2.items[:p] if item not in middle]
        end = [item for item in parent_2.items if item not in middle and item not in start]
        return start + middle + end

    # Creating Individual instances for offspring
    offspring_a = Individual(create_offspring_items(parent_a, parent_b), instance.knapsack, instance.fitness_function, instance.placement_function)
    offspring_b = Individual(create_offspring_items(parent_b, parent_a), instance.knapsack, instance.fitness_function, instance.placement_function)

    return offspring_a, offspring_b


### Mutation functions
def m1_mutation(individual, instance):
    knapsack = cp.deepcopy(individual.knapsack)
    original_items = cp.deepcopy(individual.items)

    # Simplify the calculation for max_number_of_elements and items_to_remove
    max_number_of_elements = int(((instance.max_iterations - instance.current_iteration) * len(individual.layout)) / instance.max_iterations)
    items_to_remove = min(rd.randint(0, max_number_of_elements), len(individual.layout))

    items_kept = rd.sample(individual.layout, len(individual.layout) - items_to_remove)

    # Determine items_to_fill using set operations for efficiency
    items_to_fill = list((set(individual.layout) - set(items_kept)) | (set(individual.items) - set(individual.layout)))
    rd.shuffle(items_to_fill)

    if len(items_to_fill) > 0:
        knapsack.items = items_kept
        individual.knapsack = knapsack
        individual.items = items_to_fill
        individual.calculated = False
        individual.calculate()
        individual.items = original_items

    return individual

def m3_mutation(individual, instance):
    if len(individual.layout) == 0:
        return individual
        
    knapsack = cp.deepcopy(individual.knapsack)
    original_items = cp.deepcopy(individual.items)
        
    # Randomly select the items to remove
    max_number_of_elements = len(individual.layout)
    items_to_remove = rd.sample(individual.layout, rd.randint(0, max_number_of_elements))

    # Determine items to keep by subtracting items_to_remove from individual.items
    items_kept = [item for item in individual.layout if item not in items_to_remove]
    knapsack.items = items_kept

    packing_vertices = list(set(knapsack.get_items_combined_shape_vertices()) - set(knapsack.get_vertices()))
    if len(packing_vertices) == 0:
        return individual

    placing_point = min(packing_vertices, key=lambda point: (point[1], point[0]))

    minx, miny, maxx, maxy = knapsack.get_bounds()

    # Calculate the target width and height for the fits
    target_width = maxx - placing_point[0]
    target_height = maxy - placing_point[1]

    # Filter items for horizontal fits and vertical fits
    horizontal_fits = {item for item in individual.layout if item.width == target_width}
    vertical_fits = {item for item in individual.layout if item.height == target_height}
    items_that_fit = list(horizontal_fits.union(vertical_fits) - set(items_to_remove))
    rd.shuffle(items_that_fit)

    # Filter items which can be added later
    items_to_fill = list((set(individual.layout) - set(items_kept)) | set(items_to_remove) )
    rd.shuffle(items_to_fill)

    if len(items_that_fit) > 0:
        item_to_add = items_that_fit[0]
        individual.knapsack.add_item(item_to_add, placing_point, 'bottom_left')
        if item_to_add in items_to_fill: 
            items_to_fill.remove(item_to_add)

    # Apply the placement function to the updated knapsack and update individual
    knapsack.items = items_kept
    individual.knapsack = knapsack
    individual.items = items_to_fill
    individual.calculated = False
    individual.calculate()
    individual.items = original_items

    return individual


MUTATION_PROBABILITY = 0.5
def m1_m3_mutation(individual, instance):
    # If the random number is less than or equal to mutation_prob, use m1_mutation, else use m3_mutation
    if rd.random() <= MUTATION_PROBABILITY:
        return m1_mutation(individual, instance)
    else:
        return m3_mutation(individual, instance)


### Reproduction functions
ELITE_SIZE = 10
def ox3_crossover_reproduction(instance, current_population):
    # Select top ELITE_SIZE individuals for reproduction
    top_individuals = instance.population[:ELITE_SIZE]

    # Create pairs for crossover
    pairs = [(top_individuals[i], top_individuals[i + 1]) for i in range(0, len(top_individuals) - 1, 2)]

    # Use multiprocessing.Pool to construct all_tuples in parallel
    with multiprocessing.Pool() as pool:
        all_tuples = pool.starmap(ox3_crossover, [(pair[0], pair[1], instance) for pair in pairs])

    # Flatten the list of tuples
    individuals = list(chain(*all_tuples))

    # Reconstruct pairs
    reconstructed_pairs = [(individuals[i], individuals[i + 1]) for i in range(0, len(individuals) - 1, 2)]

    # Select individuals with higher fitness from each pair
    higher_fitness_individuals = [
        pair[0] if pair[0].fitness > pair[1].fitness else pair[1]
        for pair in reconstructed_pairs
    ]

    instance.population = instance.population[ELITE_SIZE:]

    return higher_fitness_individuals


### Fitness functions
def basic_fitness(individual):
    return individual.score


def num_of_items_fitness(individual):
    return individual.score - 0.02 * len(individual.layout)
