import os
import json
import pickle
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
from sacred import Experiment
from sacred.observers import FileStorageObserver

ex = Experiment('cf_with_svd')
ex.observers.append(FileStorageObserver('cf_svd'))

@ex.config
def config():
    LATENT_DIM = 500
    NEIGHBOR_K = 100
    message = "Parameters passed from Sacred configuration."

def get_map(pds, gts):
    gts = set(gts)
    aps = list()
    hits = 0
    for i, pd in enumerate(pds):
        if pd in gts:
            hits += 1
            aps.append(hits / (i + 1))
    return np.mean(aps) if len(aps) != 0 else 0

def load_umr(json_path):
    return json.load(open(json_path, 'r'))

def transform_sparse(umr):
    uids, mids = list(), list()
    uset, mset = set(), set()
    ratings = list()
    for uid, ud in umr.items():
        uset.add(uid)
        for mid, r in ud.items():
            mset.add(mid)
            uids.append(uid)
            mids.append(mid)
            ratings.append(r)
    u2i = dict(zip(uset, range(len(uset))))
    m2i = dict(zip(mset, range(len(mset))))
    uidx = np.array([u2i[uid] for uid in uids])
    midx = np.array([m2i[mid] for mid in mids])
    return coo_matrix((np.array(ratings), (uidx, midx)), dtype=np.float32), u2i, m2i

def transform_dense(umr):
    uset, mset = set(), set()
    for uid, ud in umr.items():
        uset.add(uid)
        for mid, r in ud.items():
            mset.add(mid)
    u2i = dict(zip(uset, range(len(uset))))
    m2i = dict(zip(mset, range(len(mset))))
    dense = np.zeros((len(u2i), len(m2i)))
    for uid, ud in umr.items():
        for mid, r in ud.items():
            dense[u2i[uid]][m2i[mid]] = r
    return dense, u2i, m2i

class svd:
    def __init__(self, latent_dim, neighbor_count):
        self.lat_dim = latent_dim
        self.k = neighbor_count
        self.ulat = None
        self.mlat = None
        self.sv = None
        self.m_neighbors = dict()
        self.existed = dict()
        # self.predicted = dict()

    def fit(self, matrix):
        self.ulat, self.sv, self.mlat = svds(matrix, self.lat_dim)

    def add_existed(self, umr):
        self.existed = umr

    def find_neighbors_movies(self, m2i):
        global_simi = dict()
        global_normed = dict()
        mids = list(m2i.keys())
        # use movie similarity
        for mid in mids:
            if mid not in global_normed:
                global_normed[mid] = self.mlat[:, m2i[mid]] / np.linalg.norm(self.mlat[:, m2i[mid]])
            candidates = list()
            for mmid in mids:
                if mid == mmid:
                    continue
                pair = tuple(sorted([mmid, mid]))
                if pair not in global_simi:
                    if mmid not in global_normed:
                        global_normed[mmid] =  self.mlat[:, m2i[mmid]] / np.linalg.norm(self.mlat[:, m2i[mmid]])
                    global_simi[pair] = global_normed[mmid].dot(global_normed[mid])
                candidates.append((mmid, global_simi[pair]))
            self.m_neighbors[mid] = sorted(candidates, key=lambda x: -x[1])[:self.k]

    def test(self, umr, u2i, m2i):
        preds = dict()
        error, count = 0, 0
        for uid, ud in umr.items():
            if uid not in preds:
                preds[uid] = dict()
            # ulat = np.multiply(self.ulat[u2i[uid], :], self.sv)
            ulat = np.multiply(self.ulat[u2i[uid], :], self.sv)
            for mid, r in ud.items():
                if mid not in m2i:
                    preds[uid][mid] = float(3) # neutral rating for unseen movies
                else:
                    preds[uid][mid] = float(ulat.dot(self.mlat[:, m2i[mid]]))
                error += (r - preds[uid][mid]) ** 2
                count += 1
        return preds, np.sqrt(error / count)

    def predict(self, umr, u2i, m2i):
        aps = list()
        for uid, ud in umr.items():
            candidates = dict()
            r_sum = 0
            for mid, r in self.existed[uid].items():
                for n, s in self.m_neighbors[mid]:
                    if n in self.existed[uid]:
                        continue
                    if n not in candidates:
                        candidates[n] = 0
                    candidates[n] += r * s
            candidates = sorted(list(candidates.items()), key=lambda x: -x[1])[:self.k]
            preds = [t[0] for t in candidates]
            gts = set([mid for mid, r in ud.items() if r >= 1])
            aps.append(get_map(preds, gts))
        return np.mean(aps)

    def save_model(self, folder, annotation):
        pickle.dump(self.ulat, open(os.path.join(folder, annotation + '-u.pickle'), 'wb'), protocol=4)
        pickle.dump(self.mlat, open(os.path.join(folder, annotation + '-m.pickle'), 'wb'), protocol=4)
        pickle.dump(self.m_neighbors, open(os.path.join(folder, annotation + '-n.pickle'), 'wb'), protocol=4)
        pickle.dump(self.existed, open(os.path.join(folder, annotation + '-e.pickle'), 'wb'), protocol=4)
        # pickle.dump(self.predicted, open(os.path.join(folder, annotation + '-p.pickle'), 'wb'), protocol=4)

@ex.automain
def main(LATENT_DIM, NEIGHBOR_K, message):
    TRAIN_PATH = 'data/toy_train.json'
    # TEST_PATH = 'data/toy_test.json'
    VAL_PATH = 'data/toy_val.json'
    OUTPUT_PATH = 'pred_output'
    MODEL_PATH = 'best_model'
    ANNOTATION = 'baseline'

    for folder in (OUTPUT_PATH, MODEL_PATH):
        if not os.path.exists(folder):
            os.mkdir(folder)

    # load training set
    train_umr = load_umr(TRAIN_PATH)
    # test_umr = load_umr(TEST_PATH)
    val_umr = load_umr(VAL_PATH)
    print("Finished loading data.")
    train_coo, uid_to_idx, mid_to_idx = transform_sparse(train_umr)
    print("Finised converting to sparse matrix.")

    # build svd model
    new_svd = svd(LATENT_DIM, NEIGHBOR_K)
    new_svd.fit(train_coo)
    new_svd.add_existed(train_umr)
    print("Finished adding existing ratings--to avoid repeated recommendations.")
    new_svd.find_neighbors_movies(mid_to_idx)
    print("Finished finding neighbors.")
    new_svd.save_model(MODEL_PATH, ANNOTATION)
    print("Finished training (and saving) the model.")
    _, train_error = new_svd.test(train_umr, uid_to_idx, mid_to_idx)
    print(f"\tTrain RMSE:{train_error:.6f}\n")

    # test on the validation set
    val_map = new_svd.predict(val_umr, uid_to_idx, mid_to_idx)
    print("Finished testing on validation set.")
    print(f"\tVal MAP:{val_map:.6f}")

    # add the result
    ex.log_scalar('train_rmse', train_error)
    ex.log_scalar('val_map', val_map)

    print("\n%s\n\n" % message)


if __name__ == '__main__':
    lat_dims = [10, 20, 50, 100, 200, 500]
    num_neighbors = [5, 10, 20, 50, 100]
    for lat_dim in lat_dims:
        for num_neighbor in num_neighbors:
            if lat_dim == 500 and num_neighbor == 100:
                continue
            ex.run(config_updates={'LATENT_DIM': lat_dim,
                                   'NEIGHBOR_K': num_neighbor})
