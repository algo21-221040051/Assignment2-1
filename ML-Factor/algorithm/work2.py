import os
import _pickle as cPickle
# import pickle
from six import StringIO
import pandas as pd
import addpath


pkl_path = os.path.join(addpath.data_path, 'cn_data', 'factors', 'factor_solve_data.pkl')
# pkl_file_read = pd.read_pickle(pkl_path)
# F = open(pkl_path, 'rb')
# factor_data = cPickle.load(StringIO(pkl_file_read))

with open(pkl_path, 'r') as f:
    data = cPickle.load(f)

print('done')