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
        self.Z1 = self.A - 1e-7
        self.Z2 = self.A - 1e-7
        self.U1 = np.zeros((self.dim, self.dim))
        self.U2 = np.zeros((self.dim, self.dim))
        self.eps = 1e-3

    def g_func(self, x):
        return math.exp(-self.beta*x)

    def S_func(self, alpha, X):
        u, s, v = np.linalg.svd(X)
        s = s - alpha
        s[s<0] = 0
        return np.matmul(np.matmul(u,np.diag(s)), v)

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

    def p_func(self, batch_data, u, u_2):
        # given u,u', calc \sum p_ii and C
        sum_pii = 0
        C = 0
        for sample in batch_data:
            dim_idx = np.array(sample.dim_list)
            wanted_dim_idx = np.where(dim_idx==u)[0]
            for i in wanted_dim_idx:
                sum_func = np.dot(sample.G_matrix[i], self.A[u])
                pii = self.mu[u]/(self.mu[u]+sum_func)
                sum_pii += pii
                sum_pij = self.A[u,u_2]/(self.mu[u]+sum_func)*sample.G_matrix[i,u]
                C += sum_pij
        return sum_pii, C


    def B_func(self, batch_data):
        B = -self.Z1 + self.U1 - self.Z2 + self.U2
        B = B*self.row
        for idx in range(len(batch_data)):
            sample = batch_data[idx]
            last_T = sample.get_point(sample.get_size()-1)[0]
            for j in range(sample.get_size()):
                B[:,sample.get_point(j)[1]] += self.G_func(last_T-sample.get_point(j)[0])
        return B

    def G_func(self, x):
        return (math.exp(-self.beta*x) - 1)/(-self.beta)

    def renew_A_mu(self, batch_data):
        B = self.B_func(batch_data)

        sum_T = 0
        for i in range(len(batch_data)):
            sample = batch_data[i]
            sum_T += sample.get_point(sample.get_size() - 1)[0]

        mu = np.zeros(self.dim)
        C = np.zeros((self.dim, self.dim))
        for u1 in range(self.dim):
            for u2 in range(self.dim):
                sum_pii, c = self.p_func(batch_data, u1, u2)
                mu[u1] = sum_pii / sum_T
                C[u1,u2] = c
        self.mu = mu
        A = -B + np.sqrt(B*B + 8*self.row*C)
        A = A / (4*self.row)
        self.A = A

    def train(self, epoc, verbose=False):
        if verbose:
            print("begin training")
        L_history = []
        for k in range(epoc):
            batch_data = self.train_data.get_next_batch()
            old_A = self.A.copy()
            old_mu = self.mu.copy()
            self.renew_A_mu(batch_data)
            while np.linalg.norm(old_A-self.A) > self.eps*np.linalg.norm(self.A) or np.linalg.norm(old_mu - self.mu) > self.eps*np.linalg.norm(self.mu):
                print(np.linalg.norm(old_A-self.A), np.linalg.norm(old_mu - self.mu))
                old_A = self.A.copy()
                old_mu = self.mu.copy()
                self.renew_A_mu(batch_data)
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
        print('calculating L')
        for idx in range(self.test_data.get_size()):
            sample = self.test_data.get_sample(idx)
            log = 0
            for i in range(sample.get_size()):
                tmp = self.mu[sample.get_point(i)[1]]
                tmp += np.dot(sample.G_matrix[i], self.A[sample.get_point(i)[1]])
                log += math.log(tmp)
            Tc = sample.get_point(sample.get_size()-1)[0]
            item2 = -Tc * self.mu.sum()
            item3 = 0
            for u in range(self.dim):
                for i in range(sample.get_size()):
                    item3 += self.A[u,sample.get_point(i)[1]]*self.G_func(Tc-sample.get_point(i)[0])
            item3 = (item3 - 1)/(-self.beta)
            L += (log + item2 - item3)
        return L