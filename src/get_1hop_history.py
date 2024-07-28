import os
import torch
torch.backends.cudnn.enabled = False
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from tqdm import tqdm
import  argparse

args = argparse.ArgumentParser()
args.add_argument('--dataset', type=str, default='ICEWS14s')
args = args.parse_args()
print(args)

def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def load_all_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    quadrupleList = []
    times = set()
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName2), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    with open(os.path.join(inPath, fileName3), 'r') as fr:
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

all_data, all_times = load_all_quadruples('../data/{}'.format(args.dataset), 'train.txt', 'valid.txt', "test.txt")
num_e, num_r = get_total_number('../data/{}'.format(args.dataset), 'stat.txt')

save_dir_obj = '../data/{}/history_1hop_10/'.format(args.dataset)

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

mkdirs(save_dir_obj)
raw_num_r = num_r
num_r = num_r * 2
for tim in tqdm(all_times):
    # if i != 0:
    #     if i < 10:
    #         input_data = all_data[0:i]
    #     else:
    #         input_data = all_data[i-10:i]
    #     train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in input_data])
    train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in all_data if quad[3] < tim])
    if tim != all_times[0]:
        train_new_data = torch.from_numpy(train_new_data)
        train_new_data = train_new_data[:, [0, 2]]
        inverse_train_data = train_new_data[:, [1, 0]]
        # inverse_train_data[:, 1] = inverse_train_data[:, 1] + raw_num_r
        train_new_data = torch.cat([train_new_data, inverse_train_data])

        # entity history
        train_new_data = torch.unique(train_new_data, sorted=False, dim=0)
        train_new_data = train_new_data.numpy()
        row = train_new_data[:, 0]
        col = train_new_data[:, 1]
        d = np.ones(len(row))
        tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e, num_e))
        #
        # # relation history
        # rel_row = train_new_data[:, 0] * num_e + train_new_data[:, 2]
        # rel_col = train_new_data[:, 1]
        # rel_d = np.ones(len(rel_row))
        # rel_seq = sp.csr_matrix((rel_d, (rel_row, rel_col)), shape=(num_e * num_e, num_r))
    else:
        tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_e, num_e))
        # rel_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * num_e, num_r))
    sp.save_npz('../data/{}/history_1hop_10/tail_history_{}.npz'.format(args.dataset, tim), tail_seq)
    # sp.save_npz('../data/{}/history/rel_history_{}.npz'.format(args.dataset, tim), rel_seq)








