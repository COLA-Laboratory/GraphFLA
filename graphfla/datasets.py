import pandas as pd
import os

class DataLoader:
    """
    Class implementing a data loader object for automatically loading datasets.

    Parameters
    ----------
    path : str
        The file path to the dataset. Supports CSV and Excel file formats.

    cache : bool, default=True
        Indicates whether to cache the loaded data for reuse to avoid redundant loading.

    Attributes
    ----------
    path : str
        The path to the data file.

    cache : bool
        Whether caching is enabled.

    _data : pd.DataFrame or None
        The cached dataset stored as a pandas DataFrame if caching is enabled.
    """

    def __init__(self, path: str, cache: bool = True) -> None:
        """
        Initialize the DataLoader with the given path and caching option.

        Parameters
        ----------
        path : str
            The path to the data file.

        cache : bool, default=True
            Whether to enable caching of the loaded dataset.
        """
        self.path = path
        self.cache = cache
        self._data = None  

    def load_data(self) -> pd.DataFrame:
        """
        Load the dataset from the specified path. Supports CSV and Excel files.

        Returns
        -------
        pd.DataFrame
            The loaded dataset as a pandas DataFrame.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.

        ValueError
            If the file format is not supported.
        """
        if self.cache and self._data is not None:
            print("Returning cached data.")
            return self._data

        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"The file '{self.path}' does not exist.")

        file_extension = os.path.splitext(self.path)[1].lower()

        if file_extension == '.csv':
            self._data = pd.read_csv(self.path)
        elif file_extension in ['.xls', '.xlsx']:
            self._data = pd.read_excel(self.path)
        else:
            raise ValueError(f"Unsupported file format: '{file_extension}'")

        print(f"Data loaded successfully from '{self.path}'.")
        return self._data
    
def GB1():
    """
    The GB1 binding protein landscape from [1]. It is a 4-site combinatorially complete 
    landscape, containing all 4^20 = 160,000 possible variants of the B1 domain of protein 
    G through saturation mutagenesis at four carefully chosen residue sites (V39, D40, G41, 
    and V54). Among these, fitness was measured experimentally for 149,361 variants. It is 
    quantified the protein's ability to bind to the fragment crystallizable domain of 
    immunoglobulins. 

    - Sequence: protein
    - Type: combinatorially complete
    - Size: 4^20 = 160,000
    - Fitness: binding affinity

    This landscape is fairly smooth, with only 30 fitness peaks. 

    [1] N. C. Wu, L. Dai, C. A. Olson, J. O. Lloyd-Smith, R. Sun, Adaptation in protein 
    fitness landscapes is facilitated by indirect paths. eLife 5, e16965 (2016).
    """
    data = pd.read_csv("GB1.csv")
    
    return data

def DNA():
    """
    The DNA fitness landscape dataset from [1]. It represents a combinatorially complete 
    mutational landscape of 9 nucleotides encoding three successive amino acids in the 
    dihydrofolate reductase (DHFR) protein of Escherichia coli. The dataset contains 
    over 260,000 possible genotypes (262,144 variants) created via CRISPR-Cas9 gene 
    editing and measured for fitness in the presence of the antibiotic trimethoprim.

    - Sequence: DNA (nucleotide)
    - Type: combinatorially complete
    - Size: 4^9 = 262,144
    - Fitness: resistance to trimethoprim (relative fitness of DHFR variants)

    The fitness landscape is rugged, harboring 514 fitness peaks, yet the highest peaks 
    are accessible via abundant fitness-increasing paths.

    [1] A. Papkou, L. Garcia-Pastor, J. A. Escudero, A. Wagner, 
    "A rugged yet easily navigable fitness landscape," *Science*, 382(6673), eadh3860 (2023).
    DOI: 10.1126/science.adh3860
    """
    data = pd.read_csv("DNA.csv")

    return data


