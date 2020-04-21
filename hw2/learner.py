import numpy as np
import math


class HawkesProcessLearner:
    def __init__(self, lam, row, beta, train_data, test_data, dim):
        self.train_data = train_data
        self.test_data = test_data
        self.lam1 = lam[0]
        self.lam2 = lam[1]
        self.dim = dim
        self.row = row
        self.beta = beta
        self.A = np.random.rand(self.dim, self.dim)
        self.mu = np.random.rand(self.dim)
        self.Z1 = self.A - 1e-3
        self.Z2 = self.A - 1e-3
        self.U1 = np.zeros((self.dim, self.dim))
        self.U2 = np.zeros((self.dim, self.dim))

    def g_func(self, x):
        return math.exp(-self.beta*x)

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

    def p_func(self, i, j, c, batch_data, sum=None):
        sample = batch_data[c]

        val = self.mu[sample.get_point(i)[1]]
        if i == j:
            for k in range(0, i):
                val += self.A[sample.get_point(i)[1]][sample.get_point(k)[1]] * self.g_func(
                    sample.get_point(i)[0] - sample.get_point(k)[0])
            val = self.mu[sample.get_point(i)[1]] / val
        else:
            sum += self.A[sample.get_point(i)[1]][sample.get_point(j)[1]] * \
                   self.g_func(sample.get_point(i)[0]-sample.get_point(j)[0])
            val += sum
            val = self.A[sample.get_point(i)[1]][sample.get_point(j)[1]] * \
                  self.g_func(sample.get_point(i)[0]-sample.get_point(j)[0]) / val
        return val, sum

    def renew_mu(self, batch_data):
        sum_T = 0
        mu = np.zeros(self.dim)
        print("calculating mu")
        for i in range(len(batch_data)):
            sample = batch_data[i]
            sum_T += sample.get_point(sample.get_size()-1)[0]
            for idx in range(sample.get_size()):
                mu[sample.get_point(idx)[1]] += self.p_func(idx, idx, i, batch_data)[0]
        self.mu = mu / sum_T

    def B_func(self, batch_data):
        B = -self.Z1 + self.U1 - self.Z2 + self.U2
        B = B*self.row
        for idx in range(len(batch_data)):
            sample = batch_data[idx]
            last_T = sample.get_point(sample.get_size()-1)[0]
            for j in range(sample.get_size()):
                B[:,sample.get_point(j)[1]] += self.G_func(last_T-sample.get_point(j)[0])
        return B

    def C_func(self, batch_data):
        C = np.zeros((self.dim, self.dim))
        for idx in range(len(batch_data)):
            sample = batch_data[idx]
            print(idx)
            for i in range(sample.get_size()):
                sum = 0
                for j in range(i):
                    delta, sum = self.p_func(i, j, idx, batch_data, sum)
                    C[sample.get_point(i)[1]][sample.get_point(j)[1]] += delta
        return C

    def G_func(self, x):
        return (math.exp(-self.beta*x) - 1)/(-self.beta)

    def renew_A(self, batch_data):
        print("calculating B")
        B = self.B_func(batch_data)
        print("calculating C")
        C = self.C_func(batch_data)
        A = -B + np.sqrt(B.dot(B) + 8*self.row*C)
        A = A / (4*self.row)
        self.A = A

    def train(self, epoc, verbose=False):
        if verbose:
            print("begin training")
        L_history = []
        for k in range(epoc):
            batch_data = self.train_data.get_next_batch()
            old_A = self.A
            old_mu = self.mu
            self.renew_A(batch_data)
            self.renew_mu(batch_data)
            while np.linalg.norm(old_A-self.A) > 1e-7 or np.linalg.norm(old_mu - self.mu) > 1e-7:
                print(np.linalg.norm(old_A-self.A), np.linalg.norm(old_mu - self.mu))
                old_A = self.A
                old_mu = self.mu
                self.renew_A(batch_data)
                self.renew_mu(batch_data)
            self.renew_Z1()
            self.renew_Z2()
            self.renew_U()
            L = self.log_likelyhood()
            if verbose:
                print("epoc: "+str(k)+" loss: "+str(L))
            L_history.append(L)
        return L_history

    def log_likelyhood(self):
        L = 0
        for idx in range(self.test_data.get_size()):
            sample = self.test_data.get_sample(idx)
            log = 0
            for i in range(sample.get_size()):
                tmp = self.mu[sample.get_point(i)[1]]
                for j in range(i):
                    tmp += self.A[sample.get_point(i)[1]][sample.get_point(j)[1]]*\
                           self.g_func(sample.get_point(i)[0]-sample.get_point(j)[0])
                log += math.log(tmp)
            Tc = sample.get_point(sample.get_size()-1)[0]
            item2 = -Tc * self.mu.sum()
            item3 = 0
            for i in range(self.dim):
                for j in range(sample.get_size()):
                    item3 += self.A[i][sample.get_point(j)[1]] * self.G_func(Tc-sample.get_point(j)[1])
            L += (log + item2 + item3)
        return L