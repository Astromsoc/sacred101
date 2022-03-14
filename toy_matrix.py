import numpy as np
import pickle
import os

M = 200
N = 100
DATA_PATH = 'data'

if __name__ == '__main__':

    # fix the seed
    np.random.seed(416)

    # generate the matrix
    mat = np.random.random((M, N))
    pickle.dump(mat, open(os.path.join(DATA_PATH, 'toy_mat.pickle'),'wb'))
