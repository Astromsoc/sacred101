import os
import json
import pickle
import numpy as np
from preprocess import train_test_split

if __name__ == '__main__':
    EXTRACT_REAL_DATA = False
    USER_COUNT = 5000
    MOVIE_COUNT = 1000
    ORIGINAL_PATH = 'pickles/movies_rated_by_users.pickle'
    OUTPUT_PATH = 'processed'

    if EXTRACT_REAL_DATA:
        original_umr = pickle.load(open(ORIGINAL_PATH, 'rb'))
        toy_umr = dict(list(original_umr.items())[:USER_COUNT])
        toy_train, toy_test = train_test_split(toy_umr, 0.9)
        toy_train, toy_val = train_test_split(toy_train, 0.9)
        json.dump(toy_train, open(os.path.join(OUTPUT_PATH, 'toy_train.json'), 'w'), indent=2)
        json.dump(toy_val, open(os.path.join(OUTPUT_PATH, 'toy_val.json'), 'w'), indent=2)
        json.dump(toy_test, open(os.path.join(OUTPUT_PATH, 'toy_test.json'), 'w'), indent=2)

    else:
        uids = list(np.random.permutation(range(1, 10000))[:USER_COUNT])
        raw_mids = list(np.random.permutation(range(100000, 200000))[:MOVIE_COUNT])
        mids = list()
        for i, mid in enumerate(raw_mids):
            mid = str(mid)
            digits = np.random.choice(range(6), 2, replace=False)
            for d in digits:
                mid = mid[:d] + chr(ord('a') + int(mid[d])) + mid[d + 1:]
            mids.append(mid)
        umr = dict()
        for i, uid in enumerate(uids):
            if (i + 1) % 100 == 0:
                print(f"{(i + 1)} users finished.")
            uid = str(uid)
            umr[uid] = dict()
            valids = np.random.choice(range(10,101), replace=False)
            now_mids = np.random.permutation(mids)[:valids]
            for mid in now_mids:
                umr[uid][mid] = float(np.random.choice(range(1, 6)))

        toy_train, toy_test = train_test_split(umr, 0.6)
        toy_train, toy_val = train_test_split(toy_train, 0.6)
        json.dump(toy_train, open(os.path.join(OUTPUT_PATH, 'toy_train_fake.json'), 'w'), indent=2)
        json.dump(toy_val, open(os.path.join(OUTPUT_PATH, 'toy_val_fake.json'), 'w'), indent=2)
        json.dump(toy_test, open(os.path.join(OUTPUT_PATH, 'toy_test_fake.json'), 'w'), indent=2)
