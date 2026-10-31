 # -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio

def Sampling(groundtruth):              #divide dataset into train and test datasets
    labeled = {}
    test    = {}
    valid   = {}
    all     = {}
    m = max(groundtruth)
    labeled_indices = []
    test_indices    = []
    valid_indices   = []
    all_indices     = []
    for i in range(m+1):
        indices = [j for j, x in enumerate(groundtruth.ravel().tolist()) if x == i]
        if i != 0:
            np.random.shuffle(indices)
            n = len(indices)
            # Author protocol = 100 train / 100 valid / rest test (designed for
            # large-class datasets such as Pavia).  Indian Pines has several
            # classes with < 200 samples, which would leave those classes with
            # ZERO test pixels (-> AA = nan).  So for large classes we keep the
            # original 100/100/rest; for small classes we scale down while
            # guaranteeing at least one test pixel per class.
            if n >= 300:
                n_lab, n_val = 100, 100
            else:
                n_lab = max(3, int(round(0.15 * n)))
                n_val = max(1, int(round(0.15 * n)))
                if n_lab + n_val >= n:
                    n_lab = max(1, int(0.40 * n))
                    n_val = max(1, int(0.20 * n))
                    if n_lab + n_val >= n:
                        n_val = max(1, n - n_lab - 1)
            all[i]     = indices
            labeled[i] = indices[:n_lab]
            valid[i]   = indices[n_lab:n_lab + n_val]
            test[i]    = indices[n_lab + n_val:]
            labeled_indices += labeled[i]
            valid_indices   += valid[i]
            test_indices    += test[i]
            all_indices     += all[i]

    np.random.shuffle(labeled_indices)
    np.random.shuffle(valid_indices)
    np.random.shuffle(test_indices)
    np.random.shuffle(all_indices)
    return labeled_indices, test_indices, valid_indices,all_indices

def _find_ip_gt():
    import os, glob
    for c in ["../../data/Indian_pines_gt.mat",
              "/kaggle/input/dataset-mp/Indian_pines_gt.mat"]:
        if os.path.exists(c):
            return c
    hits = glob.glob("/kaggle/input/**/Indian_pines_gt.mat", recursive=True)
    if hits:
        return hits[0]
    return "../../data/Indian_pines_gt.mat"

mat_gt = sio.loadmat(_find_ip_gt())
label  = mat_gt['indian_pines_gt']
GT     = label.reshape(np.prod(label.shape[:2]),)

labeled_indices, test_indices, valid_indices,all_indices= Sampling(GT)

np.save('labeled_index.npy', labeled_indices)
np.save('valid_index.npy', valid_indices)
np.save('test_index.npy', test_indices)
np.save('all_index.npy', all_indices)




 


