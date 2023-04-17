import math
import random
import time

INT_MAX = 2147483647
def read_path(path):
    global listnode
    global number_of_cities
    global tabu_tenure
    listnode = []
    f = open(path)
    data=f.readlines()
    number_of_cities=int(data[3].split()[2])
    for i in range(7,len(data)-1):
        arr=[]
        line=data[i].split()
        arr.append(float(line[0]))
        arr.append(float(line[1]))
        arr.append(float(line[2]))
        listnode.append(arr)
    tabu_tenure = int(math.sqrt(number_of_cities))
    '''
        for link in matrix:
            try:
                matrix = listnode[int(link[0])]
                matrix.append(link[1:])
                listnode[int(link[0])] = matrix
            except:
                listnode[int(link[0])] = [link[1:]]
    '''
    f.close()
    return listnode, number_of_cities, tabu_tenure

def euclidean_distance(city1, city2):
    return math.sqrt((city1[1] - city2[1])**2 + ((city1[2] - city2[2])**2))

def calculate_matrix(listnode):
    matrix=[]
    for i in range(len(listnode)):
        arr=[]
        for j in range(len(listnode)):
            if i==j:
                arr.append(10000000)
            else:
                arr.append(euclidean_distance(listnode[i],listnode[j]))
        matrix.append(arr)
    return matrix

def GreedyAlgorithm(tsp):
    route = []
    used = [0] * len(tsp)
    index1=random.randint(0,len(tsp))
#    index1=0
    used[index1] = 1
    now = index1
    for i in range(len(tsp)):
        index = 0
        min = INT_MAX
        if i == 0:
            route.append(0)
        else:
            for j in range(len(tsp)):
                if tsp[now][j] < min and used[j] == 0:
                    index = j
                    min = tsp[now][j]
            route.append(index)
            used[index] += 1
            now = index
    return route

def fitness(route, matrix):
    path_length = 0

    for i in range(len(route)):
        if (i + 1 != len(route)):
            path_length += matrix[route[i]][route[i + 1]]
        else:
            path_length += matrix[route[i]][route[0]]

    return path_length


def two_swap(state):
    neighbors = []

    for i in range(number_of_cities):
        for j in range(i+1, number_of_cities):
            arr=[]
            tmp_state=state[:i]+state[j:j+1]+state[i+1:j]+state[i:i+1]+state[j+1:]
            arr.append(state[i])
            arr.append(state[j])
            arr.append(tmp_state)
            neighbors.append(arr)

    return neighbors

def subset_swap(state):
    neighbors = []

    for i in range(number_of_cities):
        for j in range(i+1, number_of_cities):
            arr=[]
            tmp = state[i:j+1]
            tmp_state = state[:i] + tmp[::-1] +state[j+1:]
            arr.append(state[i])
            arr.append(state[j])
            arr.append(tmp_state)
            neighbors.append(arr)

    return neighbors

def AssassindCreed(state):
    neighbors = []

    for i in range(number_of_cities):
        for j in range(i+1, number_of_cities):
            arr=[]
            tmp_state = state[:i] + state[i+1:j+1] +state[i:i+1]+state[j+1:]
            arr.append(state[i])
            arr.append(state[j])
            arr.append(tmp_state)
            neighbors.append(arr)

    return neighbors

def tabu_search(file_path,current_sol, Stop):
    global max_fitness
    global tabu_tenure
    Stopping_Condition=Stop
    TabuList=[]
    num_nei=3
    Ad_TabList=[0]*num_nei
    consecutive=[0]*num_nei
    listn, number_of_cities, tabu_tenure = read_path(file_path)

    for m in range(num_nei):
        array1 = []
        for n in range(number_of_cities):
            arr2 = [tabu_tenure * (-1)] * number_of_cities
            array1.append(arr2)
        TabuList.append(array1)


    matrix=calculate_matrix(listn)
    #Tạo lời giải ban đầu

#    current_sol=[]
#    for i in range(number_of_cities):
#        current_sol.append(i)
#    random.shuffle(current_sol)

    min_fitness=fitness(current_sol,matrix)
    best_sol=current_sol

    for i in range(Stopping_Condition):
        current_neighbourhood=[]
        current_neighbourhood.append(subset_swap(current_sol))
        current_neighbourhood.append(two_swap(current_sol))
        current_neighbourhood.append(AssassindCreed(current_sol))
        index=[]
        min=[]

        for k in range(len(current_neighbourhood)):
            index.append(-1)
            min.append(INT_MAX)
            for j in range(len(current_neighbourhood[k])):
                cfnode = fitness(current_neighbourhood[k][j][2],matrix)
                if cfnode < min_fitness:
                    min[k] = cfnode
                    index[k] = j
                    min_fitness = cfnode
                    best_sol = current_neighbourhood[k][j][2]
                if (cfnode < min[k] and TabuList[k][current_neighbourhood[k][j][0]][current_neighbourhood[k][j][1]] == 0):
                    min[k] = cfnode
                    index[k] = j


        sor=min.copy()
        sor.sort()
        SN=-1
        for n in range(len(Ad_TabList)):
            if Ad_TabList[n] == 0 or sor[n] < min_fitness :
                SN=min.index(sor[n])
                # SN = Selection Index
                if consecutive[n] != 0:
                    consecutive[n] += 1
                else:
                    consecutive = [0]*num_nei
                    consecutive[n] += 1
                for p in range(len(Ad_TabList)):
                    if Ad_TabList[p] != 0:
                        Ad_TabList[p] -=1
                if consecutive[n] >= 7:
                    Ad_TabList[n] = 5

                break
            else:
                continue


        current_sol = current_neighbourhood[SN][index[SN]][2]
        for m in range(number_of_cities):
            for n in range(number_of_cities):
                if TabuList[SN][m][n] != 0:
                    TabuList[SN][m][n] -= 1
                else:
                    continue

        TabuList[SN][current_neighbourhood[SN][index[SN]][1]][current_neighbourhood[SN][index[SN]][0]] = tabu_tenure
        TabuList[SN][current_neighbourhood[SN][index[SN]][0]][current_neighbourhood[SN][index[SN]][1]] = tabu_tenure


    return min_fitness, best_sol

def main(file_path, Stopping_Condition):
    return tabu_search(file_path, Stopping_Condition)





