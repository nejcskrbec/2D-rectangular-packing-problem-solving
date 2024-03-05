import random as rd
import bisect as bi

from shared.knapsack import *
from shared.item import *

class Individual:
    def __init__(self, items, knapsack, fitness_function, placement_function):
        layout, score = placement_function(knapsack, items)
        self.items = layout
        self.score = score
        self.fitness = fitness_function(self)

    def __hash__(self):
        # Hash based on the sequence of item dimensions, maintaining order
        return hash(tuple((item.width, item.height) for item in self.items))

    def __eq__(self, other):
        if len(self.items) != len(other.items):
            return False 

        return all(item1.width == item2.width and item1.height == item2.height
                   for item1, item2 in zip(self.items, other.items))


import copy as cp
import random as rd

class GA():
    def __init__(self, knapsack, items, population_size, fitness_function, selection_function, crossover_function, mutation_function, mutation_rate, placement_function, stop_filling_rate, max_iterations):
        self.knapsack = cp.deepcopy(knapsack)
        self.items = cp.deepcopy(items)
        self.population_size = population_size
        self.fitness_function = fitness_function
        self.selection_function = selection_function
        self.crossover_function = crossover_function
        self.mutation_function = mutation_function
        self.mutation_rate = mutation_rate
        self.placement_function = placement_function
        self.stop_filling_rate = stop_filling_rate
        self.max_iterations = max_iterations
        self.population = []

    def create_individual(self):
        individual_items = rd.sample(self.items, rd.randint(1, len(self.items))) # Ensure variable length
        return Individual(individual_items, self.knapsack, self.fitness_function, self.placement_function)

    def generate_population(self, initial_population):
        # Convert initial_population to a set to avoid duplicates
        current_population = set(initial_population)

        # Generate new individuals and add them to the set to maintain uniqueness
        current_population |= {Individual(rd.sample(self.items, len(self.items)), self.knapsack, self.fitness_function, self.placement_function) 
                            for _ in range(self.population_size - len(current_population))}

        return sorted(list(current_population), key=lambda individual: individual.fitness, reverse=True)

    def execute(self):
        self.population = self.generate_population([])
        
        for _ in range(self.max_iterations):            
            if self.population[0].fitness >= self.stop_filling_rate:
                break

            # Selection
            selected_individuals = self.selection_function(self.population, self.population_size)

            # Crossover (Reproduction)
            offspring = self.crossover_function(self, selected_individuals)

            # Replace the old population with offspring
            self.population = self.generate_population([])

            # Mutation
            for individual in offspring:
                if rd.random() < self.mutation_rate:
                    self.mutation_function(individual, self)
            
            self.population.sort(key=lambda individual: individual.fitness, reverse=True)

        return self.population[0].items, self.population[0].fitness



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
    

def roulette_wheel_selection_with_linear_scaling(population, target_average=35):
    def linear_scaling(individuals, target_average):
        fitness_values = [ind.fitness for ind in individuals]
        avg_fitness = sum(fitness_values) / len(fitness_values)
        max_fitness = max(fitness_values)

        a = (target_average * (len(fitness_values) - 1)) / (max_fitness - avg_fitness) if max_fitness != avg_fitness else 0
        b = target_average * (max_fitness - len(fitness_values) * avg_fitness) / (max_fitness - avg_fitness) if max_fitness != avg_fitness else avg_fitness

        return [max(a * f + b, 0) for f in fitness_values]  # Ensure fitness is non-negative

    def roulette_wheel_selection(scaled_fitness, individuals):
        total_fitness = sum(scaled_fitness)
        selection_probs = [f / total_fitness for f in scaled_fitness]
        cumulative_probs = [sum(selection_probs[:i+1]) for i in range(len(selection_probs))]

        selection_point = rd.uniform(0, 1)
        index = bi.bisect_left(cumulative_probs, selection_point)
        return individuals[index]

    scaled_fitness = linear_scaling(population, target_average)  # Apply linear scaling first
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

    if offspring_a.fitness > offspring_b.fitness:
        return offspring_a
    else:
        return offspring_b


### Mutation functions
def m1_mutation(individual, instance):
    knapsack = cp.deepcopy(instance.knapsack)

    # Simplify the calculation for max_number_of_elements and items_to_remove
    max_number_of_elements = int(((instance.max_iterations - instance.current_iteration + 1) * len(individual.items)) / instance.max_iterations)
    items_to_remove = min(rd.randint(0, max_number_of_elements), len(individual.items))

    # Directly determine items to keep without explicitly calculating indices_to_remove
    items_kept = rd.sample(individual.items, len(individual.items) - items_to_remove)

    # Determine items_to_fill using set operations for efficiency
    items_to_fill = list(set(instance.items) - set(items_kept))
    knapsack.items = items_kept

    # Apply the placement function to the updated knapsack and update individual
    mutated_layout, mutated_score = instance.placement_function(knapsack, items_to_fill)
    individual.items = mutated_layout
    individual.fitness = instance.fitness_function(individual)

    return individual

def m3_mutation(individual, instance):

    if len(instance.knapsack.items) == 0:
        return individual
        
    knapsack = cp.deepcopy(instance.knapsack)
        
    # Randomly select the items to remove
    max_number_of_elements = len(individual.items)
    items_to_remove = rd.sample(individual.items, rd.randint(0, max_number_of_elements))

    # Determine items to keep by subtracting items_to_remove from individual.items
    items_kept = [item for item in individual.items if item not in items_to_remove]
    knapsack.items = items_kept

    packing_vertices = list(set(knapsack.get_items_combined_shape_vertices()) - set(knapsack.get_vertices()))
    placing_point = min(packing_vertices, key=lambda point: (point[1], point[0]))

    minx, miny, maxx, maxy = knapsack.get_bounds()

    # Calculate the target width and height for the fits
    target_width = maxx - placing_point[0]
    target_height = maxy - placing_point[1]

    # Filter items for horizontal fits and vertical fits
    horizontal_fits = {item for item in instance.items if item.width == target_width}
    vertical_fits = {item for item in instance.items if item.height == target_height}
    items_that_fit = horizontal_fits.union(vertical_fits) - set(items_to_remove)
    
    # Filter items which can be added later
    items_to_fill = list((set(instance.items) - set(items_kept)) | set(items_to_remove))
    rd.shuffle(items_to_fill)

    if len(items_that_fit) > 0:
        item_to_add = list(items_that_fit)[0]
        instance.knapsack.add_item(item_to_add, placing_point, 'bottom_left')
        if item_to_add in items_to_fill: 
            items_to_fill.remove(item_to_add)

    # Apply the placement function to the updated knapsack and update individual
    mutated_layout, mutated_score = instance.placement_function(knapsack, items_to_fill)
    individual.items = mutated_layout
    individual.fitness = instance.fitness_function(individual)

    return individual


MUTATION_PROBABILITY = 0.5
def m1_m3_mutation(individual, instance):
    # If the random number is less than or equal to mutation_prob, use m1_mutation, else use m3_mutation
    if rd.random() <= MUTATION_PROBABILITY:
        return m1_mutation(individual, instance)
    else:
        return m3_mutation(individual, instance)


### Reproduction functions
def ox3_crossover_reproduction(instance, current_population):
    pairs = [(current_population[i], current_population[i + 1]) for i in range(0, len(current_population) - 1, 2)]
    return list({ox3_crossover(pair[0], pair[1], instance) for pair in pairs})


### Fitness functions
def basic_fitness(individual):
    return individual.score


def num_of_items_fitness(individual):
    return individual.score - 0.2 * len(individual.items)
