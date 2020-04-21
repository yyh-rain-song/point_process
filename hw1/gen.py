import numpy as np
import math
import random
import matplotlib.pyplot as pyplot

random.seed(3)


class Point:
    def __init__(self, timestep=0.0, dim=-1):
        self.timestep = timestep
        self.dim = dim

    def get_time(self):
        return self.timestep


def calculus(func, s, t, interval):
    delta = (t-s)/interval
    ans = 0
    for i in range(0, interval):
        ans += func(s + i*delta) * delta
    return ans


def calculus2(func, k, s, t, interval):
    delta = (t-s)/interval
    ans = 0
    for i in range(0, interval):
        ans += func(k, s + i*delta) * delta
    return ans


class OneDimHaw:
    def __init__(self):
        self.a = 0.6
        self.b = 0.8
        self.u = 1.2
        self.points = []

    def lam(self, s):
        ans = 0
        for point in self.points:
            if point > s:
                break
            ans = ans + self.a * math.exp(-self.b*(s-point))
        ans = ans + self.u
        return ans

    def gen_points(self):
        s = 0
        n = 0
        T = 100
        while s < T:
            lam = self.lam(s)
            u = random.uniform(0,1)
            w = -math.log(u)/lam
            s = s + w
            D = random.uniform(0,1)
            if D*lam <= self.lam(s):
                n = n + 1
                self.points.append(s)
                print(s)

    def draw_qq_pic(self):
        seq = []
        for i in range(len(self.points)-1):
            interval = calculus(self.lam, self.points[i], self.points[i+1], 100)
            seq.append(interval)
        seq.sort() # this is true value
        n = len(seq)
        lam = n/sum(seq)
        y = [math.log((n+1)/i)/lam for i in range(1, n+1)]
        y.sort()
        pyplot.plot(seq, y)
        pyplot.plot(seq, seq)
        pyplot.show()


def find_suitible_dim(I, I_, D):
    dim = 0
    while D >= I[dim]/I_:
        dim = dim + 1
    return dim


class MultiDimHaw:
    def __init__(self):
        A = [0.1, 0.07, 0.004, 0, 0.003, 0, 0.09, 0, 0.07, 0.025,
             0, 0.05, 0.028, 0, 0.027, 0.065, 0, 0, 0.097, 0,
             0.09, 0, 0.006, 0.045, 0, 0, 0.053, 0.01, 0, 0.083,
             0.02, 0.03, 0, 0.073, 0.058, 0, 0.026, 0, 0, 0,
             0.05, 0.09, 0, 0, 0.066, 0, 0, 0.033, 0.006, 0,
             0.07, 0, 0, 0, 0, 0.075, 0.063, 0.078, 0.085, 0.095,
             0, 0.02, 0.001, 0, 0.057, 0.091, 0.009, 0.065, 0, 0.073,
             0, 0.09, 0, 0.088, 0, 0.078, 0, 0.09, 0.068, 0,
             0, 0, 0.093, 0, 0.033, 0, 0.069, 0, 0.082, 0.033,
             0.001, 0, 0.089, 0, 0.008, 0, 0.007, 0, 0, 0.052]
        u = [0, 0.001, 0.05, 0.1, 0.025, 0.01, 0.007, 0.03, 0.008, 0.004]
        self.w = 0.01
        self.Z = 10
        self.A = np.array(A).reshape((self.Z, self.Z))
        self.u = np.array(u)
        self.point_list = []

    def lam(self, d, t):
        tmp = 0
        for point in self.point_list:
            if point.timestep > t:
                break
            tmp = tmp + self.A[d][point.dim] * math.exp(-self.w * (t - point.timestep))
        tmp += self.u[d]
        return tmp

    def I(self, lam):
        I = [lam[0]]
        for dim in range(1, self.Z):
            I.append(I[dim-1] + lam[dim])
        return I

    def gen_event(self):
        T = 100
        I = [1]*self.Z
        I_ = sum(self.u)
        V = random.uniform(0,1)
        t = -math.log(V) / I_
        if t > T:
            return
        D = random.uniform(0,1)
        dim = find_suitible_dim(I, I_, D)
        self.point_list.append(Point(t, dim))

        while True:
            if len(self.point_list) >= 2000:
                break
            if len(self.point_list)%100 == 0:
                print(len(self.point_list))
            I_ = I[self.Z - 1]
            V = random.uniform(0,1)
            s = -math.log(V) / I_
            t = t + s
            U = random.uniform(0,1)
            if t > T:
                break
            lam = [self.lam(i, t) for i in range(self.Z)]
            I = self.I(lam)
            if U < I[self.Z - 1] / I_:
                dim = find_suitible_dim(I, I_, U)
                self.point_list.append(Point(t, dim))

    def interval_calc(self):
        A = np.zeros((self.Z, self.Z, len(self.point_list)))
        for m in range(self.Z):
            for n in range(self.Z):
                seq_start = seq_end = 0
                i = 2
                while True:
                    while seq_start < len(self.point_list) and self.point_list[seq_start].dim != m:
                        seq_start = seq_start + 1
                    if seq_start >= len(self.point_list):
                        break
                    seq_end = seq_start + 1
                    while seq_end < len(self.point_list) and self.point_list[seq_end].dim != m:
                        seq_end = seq_end + 1
                    if seq_end >= len(self.point_list):
                        break
                    end_time = self.point_list[seq_end].timestep
                    A[m][n][i-1] = math.exp(-self.w*(end_time - self.point_list[seq_start].timestep))
                    for idx in range(seq_start, seq_end):
                        point = self.point_list[idx]
                        A[m][n][i - 1] += math.exp(-self.w*(end_time-point.timestep))
                    i = i + 1
                    seq_start = seq_end

        gamma = np.zeros((self.Z, len(self.point_list)))
        for m in range(self.Z):
            seq_start = seq_end = 0
            i = 1
            while True:
                while seq_start < len(self.point_list) and self.point_list[seq_start].dim != m:
                    seq_start = seq_start + 1
                if seq_start >= len(self.point_list):
                    break
                seq_end = seq_start + 1
                while seq_end < len(self.point_list) and self.point_list[seq_end].dim != m:
                    seq_end = seq_end + 1
                if seq_end >= len(self.point_list):
                    break
                s_time = self.point_list[seq_start].timestep
                e_time = self.point_list[seq_end].timestep
                gamma[m][i] = self.u[m] * (e_time - s_time)
                for n in range(self.Z):
                    tmp = 0
                    for idx in range(seq_start, seq_end):
                        tmp += 1-math.exp(-self.w*(e_time-self.point_list[idx].timestep))
                    gamma[m][i] += self.A[m][n]/self.w*((1-math.exp(-self.w*(e_time-s_time)))*A[m][n][i-1]+tmp)
                i += 1
                seq_start = seq_end
        return gamma

    def draw_qq_pic(self, k):
        seq = []
        points = [[] for i in range(self.Z)]
        for point in self.point_list:
            points[point.dim].append(point.timestep)

        for i in range(len(points[k]) - 1):
            interval = calculus2(self.lam, k, points[k][i], points[k][i+1], 3)
            seq.append(interval)
        seq.sort()  # this is true value
        # gamma = self.interval_calc()
        # seq = gamma[k][1:]
        # seq = seq[np.nonzero(seq)]
        # seq = list(seq)
        n = len(seq)
        lam = n / sum(seq)
        y = [math.log((n + 1) / i) / lam for i in range(1, n + 1)]
        y.sort()
        pyplot.plot(seq, y)
        pyplot.plot(seq, seq)
        pyplot.show()

    def arrange_event(self):
        self.event_dict = dict()
        for i in range(self.Z):
            self.event_dict[i] = []
        for point in self.point_list:
            self.event_dict[point.dim].append(point.timestep)
        for i in range(self.Z):
            print(len(self.event_dict[i]))
#
# gen_points()
# draw_picture()


generat = MultiDimHaw()
generat.gen_event()
generat.arrange_event()
generat.draw_qq_pic(0)