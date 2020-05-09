import numpy as np
import math
from Config import Config
from data_process import Dataset, Sample
# to check whether my calculation is right.


class Checker:
    def __init__(self, A, mu, g_func):
        self.A = A
        self.mu = mu
        self.g_func = g_func

    # do calculationg in force
    def Calc_C(self, u, u_2, data):
        c = 0
        for sample in data:
            for i in range(sample.get_size()):
                if sample.get_point(i)[1] != u:
                    continue
                for j in range(i):
                    if sample.get_point(j)[1] != u_2:
                        continue
                    c += self.Calc_pij(i, j, sample)
        return c

    def Calc_pij(self, i, j, sample):
        low = self.mu[sample.get_point(i)[1]]
        for idx in range(i):
            low += self.A[sample.get_point(i)[1]][sample.get_point(idx)[1]] * self.g_func(sample.get_point(i)[0] - sample.get_point(idx)[0])
        high = self.A[sample.get_point(i)[1]][sample.get_point(j)[1]] * self.g_func(sample.get_point(i)[0] - sample.get_point(j)[0])
        return high/low

    def pij_func(self, batch_data, u, u_2):
        # given u,u', calc C
        C = 0
        for sample in batch_data:
            dim_idx = np.array(sample.dim_list)
            wanted_dim_idx = np.where(dim_idx==u)[0]
            for i in wanted_dim_idx:
                sum_func = np.dot(sample.G_matrix[i], self.A[u])
                sum_pij = self.A[u,u_2]/(self.mu[u]+sum_func)*sample.G_matrix[i,u_2]
                C += sum_pij
        C2 = self.Calc_C(u, u_2, batch_data)
        if math.fabs(C-C2) > 1e-9:
            print("ERROR!")
            print('expected '+ str(C2) + ' actual '+ str(C))
        return C

    def pii_func(self, batch_data, u):
        sum_pii = 0
        for sample in batch_data:
            dim_idx = np.array(sample.dim_list)
            wanted_dim_idx = np.where(dim_idx == u)[0]
            for i in wanted_dim_idx:
                sum_func = np.dot(sample.G_matrix[i], self.A[u])
                pii = self.mu[u] / (self.mu[u] + sum_func)
                sum_pii += pii
        return sum_pii

    def force_sum_pii(self, batch_data, u):
        sum_pii = 0
        for sample in batch_data:
            for i in range(sample.get_size()):
                if sample.get_point(i)[1] != u:
                    continue
                low = self.mu[sample.get_point(i)[1]]
                for j in range(i):
                    low += self.A[sample.get_point(i)[1]][sample.get_point(j)[1]]*self.g_func(sample.get_point(i)[0]-sample.get_point(j)[0])
                sum_pii += self.mu[u] / low
        return sum_pii


def g_func(x):
    return math.exp(-Config.beta*x)


if __name__ == '__main__':
    dim = Config.dim
    A = np.random.rand(dim, dim)
    mu = np.random.rand(dim)
    dataset = Dataset('train')
    checker = Checker(A, mu, g_func)
    for i in range(100):
        batch_data = dataset.get_next_batch()
        for u in range(dim):
            print(i, u)
            p1 = checker.pii_func(batch_data, u)
            p2 = checker.force_sum_pii(batch_data, u)
            print(p1-p2)
            if math.fabs(p1-p2) > 1e-7:
                print('ERROR')
                print("expected "+str(p2)+" get "+str(p1))
