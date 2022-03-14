# the simplest toy SVD problem
# only to track RMSE

import pickle
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('simple_config')
ex.observers.append(FileStorageObserver('just_svd'))

@ex.config
def first_config():
    LATENT_DIM = 1
    message = "Parameters passed from Sacred configuration."

def get_rmse(mat1, mat2):
    return np.sqrt(np.square(mat1 - mat2).mean())

class ssvvdd:
    def __init__(self, num_latent):
        self.lat_dim = num_latent
        self.u = None
        self.sv = None
        self.vh = None

    def fit(self, matrix):
        # full factorization
        self.u, self.sv, self.vh = np.linalg.svd(matrix)
        # truncate to the size of the latent vectors
        self.u = self.u[:, :self.lat_dim]
        self.sv = self.sv[:self.lat_dim]
        self.vh = self.vh[:self.lat_dim, :]

    def reconstruct(self):
        diag = np.multiply(self.sv, np.eye(self.lat_dim))
        return self.u.dot(diag).dot(self.vh)

@ex.automain
def main(message, LATENT_DIM):
    # load the raw data matrix
    DATA_PATH = 'data/toy_mat.pickle'
    matrix = pickle.load(open(DATA_PATH, 'rb'))

    # build svd for matrix factorization
    new_svd = ssvvdd(LATENT_DIM)
    new_svd.fit(matrix)
    print(f"\nFinished matrix factorization.")

    # reconstruct the matrix
    rebuilt = new_svd.reconstruct()
    print("Finished matrix reconstruction.")

    # compute rmse
    rmse = get_rmse(rebuilt, matrix)
    print(f"| RMSE = {rmse:.6f} |")
    ex.log_scalar('rmse', rmse)

    print("\n%s\n\n" % message)

if __name__ == '__main__':
    for ld in list(range(2, 101)):
        ex.run(config_updates={'LATENT_DIM': ld})
