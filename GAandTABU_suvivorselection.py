import TABU
import copy
import math
import random
import numpy
import heapq

distance_matrix = []
number_of_cities = 0
file_path = "bier127.tsp"
def read_data(path):
    global data
    global city
    global distance_matrix
    global number_of_cities
    f = open(path)
    data = f.readlines()
    number_of_cities = int(data[3].split()[2])
    distance_matrix = [0] * number_of_cities
    for i in range(number_of_cities):
        distance_matrix[i] = [0] * number_of_cities
    city = []
    for i in range(7, len(data) - 1):
        city.append([])
        line = data[i].split()
        for j in range(1,len(line)):
            city[i - 7].append(float(line[j]))
    for i in range(number_of_cities):
        for j in range(number_of_cities):
            distance_matrix[i][j] = distance(city[i], city[j])
    f.close()
    distance_matrix = numpy.array(distance_matrix)

def distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)
def fitness_function(solution):
    global distance_matrix
    sum = 0
    for i in range(0,len(solution)-1):
        sum = sum + distance_matrix[solution[i]][solution[i+1]]
    sum = sum + distance_matrix[solution[len(solution)-1]][solution[0]]
    return sum
def Greedy_procedure(distance_matrix):
    array = []
    index_a = 0
    index_b = 0
    min = distance_matrix[0][1]
    for i in range(0,len(distance_matrix)):
        for j in range(i + 1,len(distance_matrix)):
            if distance_matrix[i][j] <= min:
                min = distance_matrix[i][j]
                index_a = i
                index_b = j
    array.append(index_a)
    array.append(index_b)
    while len(array) < number_of_cities:
        now = array[-1]
        set = []
        for i in range(0,len(distance_matrix)):
            set.append(i)
        for i in range(0,len(array)):
            set.remove(array[i])
        min = distance_matrix[now][set[0]]
        index = 0
        for i in range(len(set)):
            if distance_matrix[now][set[i]] <= min:
                min = distance_matrix[now][set[i]]
                index = i
        array.append(set[index])
    return fitness_function(array), array
def different(parent1, parent2):
    index = 0
    for i in range(len(parent2)):
        if parent2[i] == parent1[0]:
            index = i
            break
    fake = []
    fake = parent2[index : len(parent2)] + parent2[0 : index]
    if fake == parent1:
        return False
    else: return True
def initial_population(size):
    population1 = []
    population = []
    first_individual = Greedy_procedure(distance_matrix)[1]
    population1.append(first_individual)
    individual = copy.copy(first_individual)
    for i in range(size - 1):
        random.shuffle(individual)
        while True:
            num = 0
            for j in range(len(population1)):
                if different(individual, population1[j]) == False:
                    random.shuffle(individual)
                else: num += 1
            if num == len(population1): break
        population1.append(copy.deepcopy(individual))
    for i in population1:
        population.append([fitness_function(i), i])
    return population
def roulette_wheel_selection(population):
    total_fitness = 0
    for i in range(len(population)):
        total_fitness = total_fitness + population[i][0]
    probabilities = []
    for i in range(len(population)):
        probabilities.append((total_fitness - population[i][0]))
    probabilities = [p / sum(probabilities) for p in probabilities]
    r = random.random()
    cumulative_probability = 0
    for i in range(len(population)):
        cumulative_probability += probabilities[i]
        if r < cumulative_probability:
            return population[i]
def tournament_selection(population, Tournament_size):
    set1 = random.choices(population, k = Tournament_size)
    set1.sort(key = lambda x: x[0])
    parent1 = set1[0]
    set2 = random.choices(population, k = Tournament_size)
    set2.sort(key = lambda x: x[0])
    parent2 = set2[0]
    return parent1, parent2
def one_point_crossover(parent1, parent2, crossover_rate):
    random_number = random.random()
    if random_number <= crossover_rate:
        child1 = [0]*2
        child2 = [0]*2
        point = random.randint(0,number_of_cities - 1)
        child1[1] = parent1[1][0:point]
        for k in parent2[1]:
            if (k in child1[1]) == False:
                child1[1].append(k)
        child1[0] = fitness_function(child1[1])
        child2[1] = parent2[1][0:point]
        for k in parent1[1]:
            if (k in child2[1]) == False:
                child2[1].append(k)
        child2[0] = fitness_function(child2[1])
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2
def order_crossover(parent1, parent2, crossover_rate):
    random_number = random.random()
    if random_number <= crossover_rate:
        point1 = random.randint(0, len(parent1) - 2)
        point2 = random.randint(point1, len(parent2) - 1)
        p1 = parent1[1][point1 : point2]
        p2 = parent2[1][point1 : point2]
        off_spring1, off_spring2 = [0,[]], [0,[]]
        x1, x2 = [], []
        for i in range(point2, len(parent1[1])):
            x1.append(parent1[1][i])
        for i in range(0, point2):
            x1.append(parent1[1][i])
        for i in x1:
            if (i in p2) == True: x1.remove(i)
        for i in range(point2, len(parent2[1])):
            x2.append(parent2[1][i])
        for i in range(0, point2):
            x2.append(parent2[1][i])
        for i in x2:
            if (i in p1) == True: x2.remove(i)
        off_spring1[1] = x2[len(x2) - len(parent1[1]) + point2 - 1 : len(x2)] + p1 + x2[0 : len(x2) - len(parent1[1]) + point2 - 1]
        off_spring2[1] = x1[len(x1) - len(parent1[1]) + point2 - 1 : len(x1)] + p2 + x1[0 : len(x1) - len(parent1[1]) + point2 - 1]
        off_spring1[0], off_spring2[0] = fitness_function(off_spring1[1]), fitness_function(off_spring2[1])
    else:
        off_spring1 = parent1
        off_spring2 = parent2
    return off_spring1, off_spring2
def two_point_crossover(parent1, parent2, crossover_rate):
    random_number = random.random()
    if random_number <= crossover_rate:
        child1 = [0, [-1] * len(parent1[1])]
        child2 = [0, [-1] * len(parent2[1])]
        point1 = random.randint(0, number_of_cities - 2)
        point2 = random.randint(point1, number_of_cities - 1)
        child1[1][:point1] = parent1[1][:point1]
        child1[1][point2:] = parent1[1][point2:]
        fake1 = []
        for k in range(len(parent2[1])):
            if (parent2[1][k] in child1[1]) == False:
                fake1.append(parent2[1][k])
        child1[1][point1:point2] = fake1
        child1[0] = fitness_function(child1[1])
        child2[1][:point1] = parent2[1][:point1]
        child2[1][point2:] = parent2[1][point2:]
        fake2 = []
        for k in range(len(parent1[1])):
            if (parent1[1][k] in child2[1]) == False:
                fake2.append(parent1[1][k])
        child2[1][point1:point2] = fake2
        child2[0] = fitness_function(child2[1])
    else:
        child1 = parent1
        child2 = parent2
    return child1, child2

def crossover(population, parent1, parent2, crossover_rate):
    child1_one, child2_one = one_point_crossover(parent1, parent2, crossover_rate)
    child1_two, child2_two = two_point_crossover(parent1, parent2, crossover_rate)
    child1_order, child2_order = order_crossover(parent1, parent2, crossover_rate)
    array1 = [child1_one, child1_two, child1_order]
    array2 = [child2_one, child2_two, child2_order]
    array1 = sorted(array1, key = lambda x: x[0])
    array2 = sorted(array2, key = lambda x: x[0])
    child1 = array1[0]
    if child1 in population:
        child1 = array1[1]
        if child1 in population:
            child1 = array1[2]
            if child1 in population:
                child1 = parent1
    child2 = array2[0]
    if child2 in population:
        child2 = array2[1]
        if child2 in population:
            child2 = array2[2]
            if child2 in population:
                child2 = parent2
    return child1, child2
def swap_mutation(chromosome, mutation_rate):
    random_number = random.random()
    child = copy.deepcopy(chromosome)
    if random_number <= mutation_rate:
        point1 = random.randint(0,number_of_cities - 1)
        point2 = random.randint(0,number_of_cities - 1)
        child[1][point1], child[1][point2] = child[1][point2], child[1][point1]
        child[0] = fitness_function(child[1])
    return child
def inversion_mutation(chromosome, mutation_rate):
    random_number = random.random()
    child = copy.deepcopy(chromosome)
    if random_number <= mutation_rate:
        point1 = random.randint(0, len(child) - 2)
        point2 = random.randint(point1 + 1, len(child) - 1)
        b = child[1][point1 : point2 + 1]
        b.reverse()
        child[1][point1 : point2 + 1] = b
    child = [fitness_function(child[1]), child[1]]
    return child
def nearest_city(city):
    min = max(distance_matrix[city][0], distance_matrix[city][1])
    index = 0
    for i in range(number_of_cities):
        if distance_matrix[city][i] <= min and i != city:
            min = distance_matrix[city][i]
            index = i
    return index

def IRGIBNNM_mutation(chrmosome, mutation_rate):
    child = copy.deepcopy(chrmosome)
    child = inversion_mutation(child, mutation_rate)

    point = random.randint(0, number_of_cities - 1)
    near = nearest_city(point)
    array = []
    for i in range(0, number_of_cities):
        array.append(i)
    array.remove(point)
    array.remove(near)
    random_element = random.choice(array)
    child[1][point], child[1][random_element] = child[1][random_element], child[1][point]
    child[0] = fitness_function(child[1])
    return child
def mutation(population, child, mutation_rate):
    child_swap = swap_mutation(child, mutation_rate)
    child_inversion = inversion_mutation(child, mutation_rate)
    child_IRGIBNNM = IRGIBNNM_mutation(child, mutation_rate)
    array = [child_swap, child_inversion, child_IRGIBNNM]
    array = sorted(array, key = lambda x: x[0])
    child1 = array[0]
    if child1 in population:
        child1 = array[1]
        if child1 in population:
            child1 = array[2]
            if child1 in population:
                child1 = child
    return child1

def Genetic_Algorithm(current_population, tournament_size, crossover_rate, mutation_rate, number_iteration):
    global temp
    global number_of_cities
    for i in range(number_iteration):
        print(i)
        new_population = []
        for j in range(int(len(current_population)/2)):
            #Crossover:
            if i <= number_iteration/2: parent1, parent2 = tournament_selection(current_population, tournament_size)
            else: parent1, parent2 = roulette_wheel_selection(current_population), roulette_wheel_selection(current_population)
            child1, child2 = crossover(new_population, parent1, parent2, crossover_rate)
            #Mutation:
            child1, child2 = mutation(new_population, child1, mutation_rate), mutation(new_population, child2, mutation_rate)
            new_population.append(child1)
            new_population.append(child2)
        tick = int(len(current_population) * 80/100)
        new_population1 = heapq.nsmallest(tick, list(new_population), key = lambda x: x[0])
        current_population1 = heapq.nsmallest(min(50, number_of_cities) - len(new_population1), list(current_population), key = lambda x: x[0])
        current_population = current_population1 + new_population1
        length = len(current_population)
        if length != min(50, number_of_cities):
            """current_population = current_population + new_population[:min(50, number_of_cities) - length]"""
            temp = initial_population(min(50, number_of_cities) - length + 1)
            temp.pop(0)
            current_population = current_population1 + temp
        tick = int(len(current_population) * 5 / 100)
        arr = random.choices(current_population, k = tick)
        for i in range(len(arr)):
            arr[i] = list(TABU.tabu_search(file_path, arr[i][1], 40))
        current_population[:tick] = arr

    best = min(current_population, key = lambda x: x[0])
    return best

read_data(file_path)
solution = Genetic_Algorithm(initial_population(min(50, number_of_cities)), 4, 0.95, 1/number_of_cities, 10)

