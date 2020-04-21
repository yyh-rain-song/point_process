import numpy as np
import csv
from Config import Config


class sample:
    def __init__(self, dim, name):
        self.name = name
        self.dim = dim
        self.time_list = []
        self.dim_list = []

    def add_point(self, t, u):
        self.time_list.append(t)
        self.dim_list.append(u)


def load_data(datapath):
    with open(datapath, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader][1:]
    print(rows[:8])
    sample_dict = dict()
    for row in rows:
        if row[0] not in sample_dict.keys():
            sample_dict[row[0]] = sample(dim=Config.dim, name=row[0])
        sample_dict[row[0]].add_point(float(row[1]), int(row[2]))
    return sample_dict


if __name__ == '__main__':
    sample_dict = load_data(Config.train_data_path)
    print(sample_dict)