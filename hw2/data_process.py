import numpy as np
import csv
from Config import Config


class Sample:
    def __init__(self, dim, name):
        self.name = name
        self.dim = dim
        self.time_list = []
        self.dim_list = []

    def add_point(self, t, u):
        self.time_list.append(t)
        self.dim_list.append(u)

    def get_point(self, idx):
        return self.time_list[idx], self.dim_list[idx]

    def get_size(self):
        return len(self.time_list)

    def shift_time(self):
        for i in range(self.get_size()):
            self.time_list[i] -= self.time_list[0]


class Dataset:
    def __init__(self, datapath):
        with open(datapath, 'r') as csvfile:
            reader = csv.reader(csvfile)
            rows = [row for row in reader][1:]
        self.sample_dict = dict()
        self.names = list()
        for row in rows:
            if row[0] not in self.sample_dict.keys():
                self.sample_dict[row[0]] = Sample(dim=Config.dim, name=row[0])
                self.names.append(row[0])
            self.sample_dict[row[0]].add_point(float(row[1]), int(row[2]))
        for sam in self.sample_dict.values():
            sam.shift_time()
        self._get_all_num()
        self.start = 0

    def get_sample(self, idx):
        return self.sample_dict[self.names[idx]]

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
        for sam in self.sample_dict.values():
            num.append(sam.get_size())
        ave_len = sum(num) / len(num)
        self.ave_batch_len = ave_len * Config.batch_size


if __name__ == '__main__':
    sample_dict = Dataset(Config.train_data_path)
    sample_dict.get_next_batch()