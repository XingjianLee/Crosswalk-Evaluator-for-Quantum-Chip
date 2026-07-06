import pickle
import torch
import os
# import torch.nn.functional as F
import numpy as np

def load_pkl(input_file):
    f = open(input_file, 'rb')
    output_file = pickle.load(f)
    f.close()
    return output_file



def save_pkl(data, loc):
    f = open(loc, 'wb')
    pickle.dump(data, file=f)
    f.close()
    return 0



# C_matrix_list = load_pkl('Data/Random_0/C_matrix_list.pkl')
# state_list = load_pkl('Data/Random_0/state.pkl')
# print('a')