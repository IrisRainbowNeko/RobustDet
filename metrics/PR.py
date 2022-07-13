from matplotlib import pyplot as plt
import pickle
import numpy as np

def to_PR(path):
    with open(path, 'rb') as f:
        data_dict=pickle.load(f)
    tp=data_dict['tp']
    fp=data_dict['fp']
    npos=data_dict['npos']

    #data_len=len(tp)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    recall = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return recall, prec

