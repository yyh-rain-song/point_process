import numpy as np


def init_variable(shape):
    return np.zeros(shape)


class HawkesProcessLearner:
    def __init__(self, lam, row, train_data, test_data, dim):
        self.train = train_data
        self.test = test_data
        self.lam1 = lam[0]
        self.lam2 = lam[1]
        self.dim = dim
        self.row = row
        self.A = init_variable((self.dim, self.dim))
        self.mu = init_variable(self.dim)
        self.Z1 = init_variable((self.dim, self.dim))
        self.Z2 = init_variable((self.dim, self.dim))
        self.U1 = init_variable((self.dim, self.dim))
        self.U2 = init_variable((self.dim, self.dim))

    def S_func(self, alpha, X):
        u, s, v = np.linalg.svd(X)
        s = s - alpha
        s[s<0] = 0
        return u*np.diag(s)*v

    def renew_Z1(self):
        self.Z1 = self.S_func(self.lam1/self.row, self.A+self.U1)

    def renew_Z2(self):
        A1 = self.A + self.U2 - self.lam2/self.row
        A1[A1<0] = 0
        A2 = self.A + self.U2 + self.lam2/self.row
        A2[A2>0] = 0
        self.Z2 = A1 + A2

    def renew_U(self):
        self.U1 = self.U1 + (self.A - self.Z1)
        self.U2 = self.U2 + (self.A - self.Z2)

    def p_func(self, i, j, c):
        pass