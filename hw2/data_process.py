import numpy as np
import csv
from Config import Config
import math
import pickle
import os
from pathlib import Path


class Sample:
    def __init__(self, dim, name):
        self.name = name
        self.dim = dim
        self.time_list = []
        self.dim_list = []
        self.G_matrix = None

    def add_point(self, t, u):
        self.time_list.append(t)
        self.dim_list.append(u)

    def get_point(self, idx):
        return self.time_list[idx], self.dim_list[idx]

    def get_size(self):
        return len(self.time_list)

    def shift_time(self):
        init = self.time_list[0]
        for i in range(self.get_size()):
            self.time_list[i] -= init

    def pre_calculate(self):
        self.G_matrix = np.zeros((self.get_size(), self.dim))

        def g_func(x):
            return math.exp(-Config.beta*x)

        for i in range(self.get_size()):
            sam_i = self.get_point(i)
            for j in range(i):
                sam_j = self.get_point(j)
                self.G_matrix[i,self.get_point(j)[1]] += g_func(sam_i[0]-sam_j[0])


def build_data(datapath, save_name):
    dir = Config.cache_path + "/" + 'beta_{0}'.format(Config.beta) + '/' + save_name+"/"
    pathdir = Path(dir)
    if pathdir.exists() is False:
        pathdir.mkdir(parents=True)
    sample_dict = dict()
    names = list()
    lengths = list()
    with open(datapath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader][1:]
    for row in rows:
        if row[0] not in sample_dict.keys():
            sample_dict[row[0]] = Sample(dim=Config.dim, name=row[0])
            names.append(row[0])
        sample_dict[row[0]].add_point(float(row[1]), int(row[2]))
    for sam in sample_dict.values():
        lengths.append(sam.get_size())
        name = dir + sam.name + ".dat"
        train_skip = ["g4005", "g1515"]
        test_skip = ["g2314", "g3758"]
        if os.path.exists(name) or (sam.name in train_skip and save_name == 'train') or (sam.name in test_skip and save_name == 'test'):
            continue
        sam.shift_time()
        print("calculating " + sam.name)
        sam.pre_calculate()
        # save sample file
        file = open(dir + sam.name + ".dat", "wb")
        pickle.dump(sam, file)
        file.close()
    namefile = open(Config.cache_path+'/'+save_name+'.txt', 'w')
    for i in range(len(names)):
        print(names[i], lengths[i], file=namefile)


class Dataset:
    def __init__(self, save_name):
        self.sample_list = []
        self.name_path = Config.cache_path+"/"+save_name+".txt"
        self.data_dir = Config.cache_path+'/'+save_name+"/"
        self.names = []
        with open(self.name_path) as f:
            names = f.readlines()
        train_skip = ["g4005", "g1515"]
        test_skip = ["g2314", "g3758"]
        for line in names:
            ll = line.strip().split(' ')
            if not ((ll[0] in train_skip and save_name == 'train') or (ll[0] in test_skip and save_name == 'test')):
                self.names.append(ll)
        self._get_all_num()

    def get_sample(self, idx):
        sample = pickle.load(open(self.data_dir+self.names[idx][0]+".dat", "rb"))
        return sample

    def get_size(self):
        return len(self.names)

    def get_next_batch(self):
        num = 0
        batch_data = []
        while num < self.ave_batch_len:
            sam = self.get_sample(self.start)
            batch_data.append(sam)
            num += sam.get_size()
            self.start += 1
            self.start %= self.get_size()
        return batch_data

    def _get_all_num(self):
        num = []
        for ll in self.names:
            num.append(int(ll[1]))
        ave_len = sum(num) / len(num)
        self.ave_batch_len = ave_len * Config.batch_size
        self.start = 0
        print("overall len:", sum(num))
        print("ave batch len:", self.ave_batch_len)
        print("num batches:", sum(num)/self.ave_batch_len)


if __name__ == '__main__':
    build_data(Config.train_data_path, 'train')
    build_data(Config.test_data_path, 'test')