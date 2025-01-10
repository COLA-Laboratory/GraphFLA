from collections import Counter
import networkx as nx
from tqdm import tqdm
from networkx.algorithms.isomorphism import is_isomorphic

def find_squares(graph):
    squares = []
    for node in graph.nodes:
        one_mutant_neighbors = list(graph.successors(node))
        for i in range(len(one_mutant_neighbors)):
            for j in range(i + 1, len(one_mutant_neighbors)):
                single_mutant_1 = one_mutant_neighbors[i]
                single_mutant_2 = one_mutant_neighbors[j]
                double_mutants = set(graph.successors(single_mutant_1)).intersection(
                    set(graph.successors(single_mutant_2))
                )
                for double_mutant in double_mutants:
                    squares.append((node, single_mutant_1, single_mutant_2, double_mutant))
    return squares

def assign_roles_with_fitness(graph, squares):
    role_assigned_squares = []
    for square in squares:
        fitness_values = {node: graph.nodes[node].get("fitness", 0) for node in square}
        double_mutant = max(fitness_values, key=fitness_values.get)  

        neighbors = set(nx.subgraph(graph, square).predecessors(double_mutant))

        single_mutants = [node for node in square if node in neighbors]

        if len(single_mutants) != 2:
            raise ValueError(f"Incorrect number of single mutants identified {square}.")
        
        wild_type = next(node for node in square if node not in single_mutants and node != double_mutant)

        role_assigned_squares.append({
            "wild_type": wild_type,
            "single_mutant_1": single_mutants[0],
            "single_mutant_2": single_mutants[1],
            "double_mutant": double_mutant,
            "fitness_values": fitness_values,
        })
    return role_assigned_squares

def classify_positive_negative_epistasis(square):
    wild_type = square['fitness_values'][square['wild_type']]
    single_mutant_1 = square['fitness_values'][square['single_mutant_1']]
    single_mutant_2 = square['fitness_values'][square['single_mutant_2']]
    double_mutant = square['fitness_values'][square['double_mutant']]
    epsilon = double_mutant + wild_type - single_mutant_1 - single_mutant_2
    epistasis_sign = "positive" if epsilon > 0 else "negative"
    return {"epistasis_type": epistasis_sign, "epsilon": epsilon}

def classify_epistasis_types(square):
    """
    Classify epistasis types: magnitude, simple sign, reciprocal sign, or non-epistatic.
    """
    fitness_values = square['fitness_values']
    wild_type = square['wild_type']
    single_mutant_1 = square['single_mutant_1']
    single_mutant_2 = square['single_mutant_2']
    double_mutant = square['double_mutant']

    f_ab = fitness_values[wild_type]
    f_Ab = fitness_values[single_mutant_1]
    f_aB = fitness_values[single_mutant_2]
    f_AB = fitness_values[double_mutant]

    epsilon = f_AB + f_ab - f_Ab - f_aB
    
    condition_1 = (f_Ab > f_ab) and (f_aB > f_ab) 
    condition_2 = ((f_Ab > f_ab) and (f_aB < f_ab)) or ((f_Ab < f_ab) and (f_aB > f_ab))
    condition_3 = (f_Ab < f_ab) and (f_aB < f_ab)

    if condition_1:
        return {"epistasis_type": "magnitude epistasis", "epsilon": epsilon}
    elif condition_2:
        return {"epistasis_type": "single sign epistasis", "epsilon": epsilon}
    elif condition_3:
        return {"epistasis_type": "reciprocal sign epistasis", "epsilon": epsilon} 
    else:
        return {"epistasis_type": "non-epistatic", "epsilon": epsilon}

def analyze_epistasis(graph):
    squares = find_squares(graph)

    ref_graph = nx.Graph()
    ref_graph.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4)])

    valid_squares = []
    for square in tqdm(squares[:10000], total=10000):
        square_graph = nx.subgraph(graph, square)
        square_graph = nx.to_undirected(square_graph)
        if is_isomorphic(ref_graph, square_graph):
            valid_squares.append(square)

    role_assigned_squares = assign_roles_with_fitness(graph, valid_squares)
    results = [classify_epistasis_types(square) for square in role_assigned_squares]
    epistasis_signs = [result['epistasis_type'] for result in results]
    sign_counts = Counter(epistasis_signs)
    total_squares = len(results)
    sign_fractions = {sign: count / total_squares for sign, count in sign_counts.items()}
    return sign_fractions