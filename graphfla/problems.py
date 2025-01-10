import random
import itertools
import math
import pandas as pd

class OptimizationProblem:
    """
    Base class for defining optimization problems.

    This class provides a framework for representing optimization problems
    and includes methods for evaluating solutions and generating data for analysis.
    Subclasses should implement specific optimization problem behavior.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    """

    def __init__(self, n):
        """
        Initialize the optimization problem with a given number of variables.
        """
        self.n = n
        self.variables = range(n)
    
    def evaluate(self, config):
        """
        Evaluate the fitness of a given configuration.

        This method should be implemented by subclasses to define the 
        specific evaluation criteria for the optimization problem.

        Parameters
        ----------
        config : tuple
            A configuration representing a potential solution.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_all_configs(self):
        """
        Generate all possible configurations for the problem.

        This method should be implemented by subclasses to provide the
        complete set of possible configurations for the problem.

        Returns
        -------
        iterator
            An iterator over all possible configurations.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """

        raise NotImplementedError("Subclasses should define this method to generate all configurations.")
    
    def get_data(self):
        """
        Generate a DataFrame containing configurations and their fitness values.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with columns `config` (list of variables) and `fitness`.
        """

        all_configs = self.get_all_configs()
        config_values = {config: self.evaluate(config) for config in all_configs}
        
        data = pd.DataFrame(list(config_values.items()), columns=["config", "fitness"])
        data['config'] = data['config'].apply(list)
        return data

class NK(OptimizationProblem):
    """
    NK model for fitness landscapes.

    This class represents an NK landscape, a model used to study the complexity of 
    adaptive landscapes based on interactions among components.

    Parameters
    ----------
    n : int
        The number of variables in the problem.

    k : int
        The number of interacting components for each variable.

    exponent : int, default=1
        An exponent used to transform the fitness values.
    """

    def __init__(self, n, k, exponent=1):
        """
        Initialize the NK model with given parameters.
        """

        super().__init__(n)
        self.k = k
        self.exponent = exponent
        self.dependence = [
            tuple(sorted([e] + random.sample(set(self.variables) - set([e]), k)))
            for e in self.variables
        ]
        self.values = {}
    
    def get_all_configs(self):
        """
        Generate all possible binary configurations for the NK model.

        Returns
        -------
        iterator
            An iterator over all binary configurations of length `n`.
        """

        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the NK model.

        Parameters
        ----------
        config : tuple
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """

        total_value = 0.0
        config = tuple(config)  
        for e in self.variables:
            key = (e,) + tuple(config[i] for i in self.dependence[e])
            if key not in self.values:
                self.values[key] = random.random()
            total_value += self.values[key]
        
        total_value /= self.n
        if self.exponent != 1:
            total_value = math.pow(total_value, self.exponent)
        
        return total_value

class RoughMountFuji(OptimizationProblem):
    """
    Rough Mount Fuji (RMF) model for fitness landscapes.

    This model combines a smooth fitness component with a rugged (random) component,
    creating a landscape with controlled ruggedness.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    alpha : float, default=0.5
        The ruggedness parameter that determines the balance between the smooth
        and rugged components. Must be between 0 and 1.
    """

    def __init__(self, n, alpha=0.5):
        """
        Initialize the RMF model with the given number of variables and ruggedness parameter.
        """
        super().__init__(n)
        self.alpha = alpha
        self.smooth_contribution = [random.uniform(-1, 1) for _ in range(n)]
        self.random_values = {}

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the RMF model.

        Returns
        -------
        iterator
            An iterator over all binary configurations of length `n`.
        """
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the RMF model.

        The fitness is computed as a weighted sum of the smooth and rugged components.

        Parameters
        ----------
        config : tuple
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """
        config = tuple(config)

        # Smooth component
        smooth_value = sum(self.smooth_contribution[i] * config[i] for i in self.variables)

        # Rugged component
        if config not in self.random_values:
            self.random_values[config] = random.random()
        
        rugged_value = self.random_values[config]

        # Combine the smooth and rugged components using alpha
        fitness = (1 - self.alpha) * smooth_value + self.alpha * rugged_value

        return fitness

class HoC(RoughMountFuji):
    """
    House of Cards (HoC) model for fitness landscapes.

    This model represents a purely random fitness landscape where each configuration
    is assigned a random fitness value independently of others. It is a special case
    of the Rough Mount Fuji (RMF) model where the ruggedness parameter alpha is 1.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    """

    def __init__(self, n):
        """
        Initialize the HoC model with the given number of variables.
        """
        # Initialize the RMF model with alpha = 1 (purely random)
        super().__init__(n, alpha=1.0)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the HoC model.

        The fitness is purely random for each configuration.

        Parameters
        ----------
        config : tuple
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The random fitness value of the configuration.
        """
        config = tuple(config)

        # Purely random fitness, no smooth component
        if config not in self.random_values:
            self.random_values[config] = random.random()

        return self.random_values[config]
    
class Additive(OptimizationProblem):
    """
    Additive model for fitness landscapes.

    This model represents a fitness landscape where the fitness of a configuration
    is the sum of independent contributions from each variable.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.
    """

    def __init__(self, n):
        """
        Initialize the Additive model with the given number of variables.
        """
        super().__init__(n)
        # Assign a random fitness contribution for each variable set to 0 and 1
        self.contributions = {i: (random.random(), random.random()) for i in self.variables}

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the Additive model.

        Returns
        -------
        iterator
            An iterator over all binary configurations of length `n`.
        """
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the Additive model.

        The fitness is computed as the sum of independent contributions of each variable.

        Parameters
        ----------
        config : tuple
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """
        config = tuple(config)
        fitness = sum(self.contributions[i][config[i]] for i in self.variables)
        return fitness

class Eggbox(OptimizationProblem):
    """
    Eggbox model for fitness landscapes.

    This model represents a fitness landscape with regularly spaced peaks and valleys,
    resembling the structure of an egg carton.

    Parameters
    ----------
    n : int
        The number of variables in the optimization problem.

    frequency : float, default=1.0
        The frequency of the peaks and troughs in the landscape.
    """

    def __init__(self, n, frequency=1.0):
        """
        Initialize the Eggbox model with the given number of variables and frequency.
        """
        super().__init__(n)
        self.frequency = frequency

    def get_all_configs(self):
        """
        Generate all possible binary configurations for the Eggbox model.

        Returns
        -------
        iterator
            An iterator over all binary configurations of length `n`.
        """
        return itertools.product((0, 1), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the Eggbox model.

        The fitness is determined based on the sum of the variables and a sine function,
        creating a periodic landscape with alternating peaks and valleys.

        Parameters
        ----------
        config : tuple
            A binary configuration representing a potential solution.

        Returns
        -------
        float
            The fitness value of the configuration.
        """
        config = tuple(config)
        # Sum of the binary configuration elements
        sum_of_elements = sum(config)
        # Fitness value using a sine function to create periodic peaks
        fitness = math.sin(self.frequency * sum_of_elements * math.pi) ** 2
        return fitness

    
class Max3Sat(OptimizationProblem):
    """
    Max-3-SAT optimization problem.

    This class represents the Max-3-SAT problem, where the goal is to maximize
    the number of satisfied clauses in a Boolean formula with exactly three literals per clause.

    Parameters
    ----------
    n : int
        The number of Boolean variables.

    alpha : float
        The clause-to-variable ratio, determining the number of clauses.
    """

    def __init__(self, n, alpha):
        """
        Initialize the Max-3-SAT problem with given parameters.
        """

        super().__init__(n)
        self.m = int(alpha * n)
        self.clauses = self._generate_clauses()
        
    def _generate_clauses(self):
        """
        Generate a set of 3-literal clauses.

        Returns
        -------
        list
            A list of randomly generated 3-literal clauses.
        """

        clauses = set()
        while len(clauses) < self.m:
            vars = random.sample(self.variables, 3)
            clause = tuple((var, random.choice([True, False])) for var in vars)
            clauses.add(clause)
        return list(clauses)
    
    def get_all_configs(self):
        """
        Generate all possible configurations for the Max-3-SAT problem.

        Returns
        -------
        iterator
            An iterator over all Boolean configurations of length `n`.
        """

        return itertools.product((True, False), repeat=self.n)

    def evaluate(self, config):
        """
        Evaluate the fitness of a configuration in the Max-3-SAT problem.

        Parameters
        ----------
        config : tuple
            A Boolean configuration representing a potential solution.

        Returns
        -------
        int
            The number of satisfied clauses.
        """

        num_satisfied = 0
        for clause in self.clauses:
            if any((config[var] == is_positive) for var, is_positive in clause):
                num_satisfied += 1
        return num_satisfied
    
