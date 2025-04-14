from itertools import combinations, product


def dna_neighbor_generator(config, config_dict, n_edit=1):
    bases = ["A", "T", "C", "G"]

    def get_neighbors(index, value):
        return [b for b in bases if b != value]

    def k_edit_combinations():
        for indices in combinations(range(len(config)), n_edit):
            current = list(config)
            possible_values = [get_neighbors(i, current[i]) for i in indices]
            for changes in product(*possible_values):
                neighbor = list(current)
                for idx, new_val in zip(indices, changes):
                    neighbor[idx] = new_val
                yield tuple(neighbor)

    return list(k_edit_combinations())


def rna_neighbor_generator(config, config_dict, n_edit=1):
    bases = ["A", "U", "C", "G"]

    def get_neighbors(index, value):
        return [b for b in bases if b != value]

    def k_edit_combinations():
        for indices in combinations(range(len(config)), n_edit):
            current = list(config)
            possible_values = [get_neighbors(i, current[i]) for i in indices]
            for changes in product(*possible_values):
                neighbor = list(current)
                for idx, new_val in zip(indices, changes):
                    neighbor[idx] = new_val
                yield tuple(neighbor)

    return list(k_edit_combinations())


def protein_neighbor_generator(config, config_dict, n_edit=1):
    amino_acids = [
        "A",
        "R",
        "N",
        "D",
        "C",
        "Q",
        "E",
        "G",
        "H",
        "I",
        "L",
        "K",
        "M",
        "F",
        "P",
        "S",
        "T",
        "W",
        "Y",
        "V",
    ]

    def get_neighbors(index, value):
        return [aa for aa in amino_acids if aa != value]

    def k_edit_combinations():
        for indices in combinations(range(len(config)), n_edit):
            current = list(config)
            possible_values = [get_neighbors(i, current[i]) for i in indices]
            for changes in product(*possible_values):
                neighbor = list(current)
                for idx, new_val in zip(indices, changes):
                    neighbor[idx] = new_val
                yield tuple(neighbor)

    return list(k_edit_combinations())


def boolean_neighbor_generator(config, config_dict, n_edit=1):
    def get_neighbors(index, value):
        return [1 - int(value)]

    def k_edit_combinations():
        for indices in combinations(range(len(config)), n_edit):
            current = list(config)
            possible_values = [get_neighbors(i, current[i]) for i in indices]
            for changes in product(*possible_values):
                neighbor = list(current)
                for idx, new_val in zip(indices, changes):
                    neighbor[idx] = new_val
                yield tuple(neighbor)

    return list(k_edit_combinations())
