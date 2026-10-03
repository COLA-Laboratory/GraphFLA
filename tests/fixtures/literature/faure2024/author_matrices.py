"""Unmodified functions extracted from Faure et al. author notebook.

lehner-lab/whmatrixextms at daabe62d0a8256e2333be8818324413daf723486.
See PROVENANCE.md and LICENSE.author. Used only for bounded validation.
"""

import numpy as np

def H_matrix_recursive(
    sequence_length = 1,
    num_states = 2,
    invert = False):
    """
    Construct multiallelic extension of the Walsh-Hadamard (WH) transform using recursive definition.

    :param sequence_length: sequence length (default:1).
    :param num_states: integer number of states (default:2).
    :param invert: invert the matrix (default:False).
    :returns: WH matrix as Numpy array.
    """
    mat = np.asarray([[1]])
    for i in range(sequence_length):
        mat_list = []
        for j in range(num_states):
            if j==0:
                mat_list += [np.concatenate([mat]*num_states, axis = 1)]
            else:
                if invert:
                    mat_list += [[mat]*num_states]
                    mat_list[-1][j] = mat_list[-1][j]*(1-num_states)
                else:
                    mat_list += [[mat*0]*num_states]
                    mat_list[-1][0] = mat
                    mat_list[-1][j] = -mat
                mat_list[-1] = np.concatenate(mat_list[-1], axis = 1)
        mat = np.concatenate(mat_list, axis = 0)
        if invert:
            mat = (1/num_states)*mat
    return mat

def H_matrix(
    str_geno,
    str_coef,
    num_states = 2,
    invert = False):
    """
    Construct multiallelic extension of the Walsh-Hadamard (WH) transform using the formula to get elements.

    :param str_geno: list of genotype strings where '0' indicates WT state.
    :param str_coef: list of coefficient strings where '0' indicates WT state.
    :param num_states: integer number of states (identical per position) or list of integers with length matching that of sequences.
    :param invert: invert the matrix.
    :returns: WH matrix as Numpy array.
    """
    #Genotype string length
    string_length = len(str_geno[0])
    #Number of states per position in genotype string (float)
    if type(num_states) == int:
        num_states = [float(num_states) for i in range(string_length)]
    else:
        num_states = [float(i) for i in num_states]
    #Convert reference characters to "." and binary encode
    str_coef = [[ord(j) for j in i.replace("0", ".")] for i in str_coef]
    str_geno = [[ord(j) for j in i] for i in str_geno]
    #Matrix representations
    num_statesi = np.repeat([num_states], len(str_geno)*len(str_coef), axis = 0)
    str_genobi = np.repeat(str_geno, len(str_coef), axis = 0)
    str_coefbi = np.transpose(np.tile(np.transpose(np.asarray(str_coef)), len(str_geno)))
    str_genobi_eq_str_coefbi = (str_genobi == str_coefbi)
    #Factors
    row_factor2 = str_genobi_eq_str_coefbi.sum(axis = 1)
    if invert:
        row_factor1 = np.prod(str_genobi_eq_str_coefbi * (num_statesi-2) + 1, axis = 1)       
        return ((row_factor1 * np.power(-1, row_factor2))/np.prod(num_states)).reshape((len(str_geno),-1))
#         row_factor1 = np.prod(np.power(1-num_statesi, str_genobi_eq_str_coefbi), axis = 1)       
#         return (row_factor1/np.prod(num_states)).reshape((len(str_geno),-1))
    else:
        row_factor1 = (np.logical_or(np.logical_or(str_genobi_eq_str_coefbi, str_genobi==ord('0')), str_coefbi==ord('.')).sum(axis = 1) == string_length).astype(float)            
        return ((row_factor1 * np.power(-1, row_factor2))).reshape((len(str_geno),-1))

def V_matrix(
    str_coef,
    num_states = 2,
    invert = False):
    """
    Construct multiallelic extension of the diagonal weighting matrix using the formula to get elements.

    :param str_geno: list of genotype strings where '0' indicates WT state.
    :param str_coef: list of coefficient strings where '0' indicates WT state.
    :param num_states: integer number of states (identical per position) or list of integers with length matching that of sequences (default:2).
    :param invert: invert the matrix (default:False).
    :returns: diagonal weighting matrix as a numpy matrix.
    """

    #Genotype subset
    str_geno = str_coef
    #Genotype string length
    string_length = len(str_geno[0])
    #Number of states per position in genotype string
    if type(num_states) == int:
        num_states = [float(num_states) for i in range(string_length)]
    else:
        num_states = [float(i) for i in num_states]
    #Convert reference characters to "."
    str_coef_ = [i.replace("0", ".") for i in str_coef]
    #initialize V matrix
    V = np.array([[0.0]*len(str_coef)]*len(str_geno))
    #Fill matrix
    for i in range(len(str_geno)):
        factor1 = int(np.prod([c for a,b,c in zip(str_coef_[i], str_geno[i], num_states) if ord(a) != ord(b)]))
        factor2 = sum([1 for a,b in zip(str_coef_[i], str_geno[i]) if ord(a) == ord(b)])
        if invert:
            V[i,i] = factor1 * np.power(-1, factor2)
        else:
            V[i,i] = 1/(factor1 * np.power(-1, factor2))
    return V
