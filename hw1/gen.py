import numpy as np
import math
import random
import matplotlib.pyplot as pyplot

random.seed(3)




class point:
    def __init__(self, timestep=0, dim=-1):
        self.timestep = timestep
        self.dim = dim

    def get_time(self):
        return self.timestep


def gen_a_point(s):
    tmp_point_list = []
    for d in range(Z):
        lam = lambda_function(s, d)
        if math.fabs(lam) < 1e-9:
            continue
        v = random.uniform(0.0,1.0)
        w = -math.log(v)/lam
        new_s = s + w
        D = random.uniform(0.0,1.0)
        if D*lam <= lambda_function(new_s, d):
            tmp_point_list.append(point(new_s, d))
    return tmp_point_list


def gen_points():
    s, n = 0, 0
    while s < stopping_time:
        tmp_point = gen_a_point(s)
        if len(tmp_point) != 0:
            tmp_point.sort(key=point.get_time)
            new_point = tmp_point[0]
            point_list.append(new_point)
            s = new_point.get_time()
            n = n + 1
            print('dim='+str(new_point.dim) + ' time='+str(new_point.get_time()))


def draw_picture():
    x = []
    y = []
    for point in point_list:
        x.append(point.timestep)
        y.append(point.dim)
    print(y)

    times = []
    value1 = []
    value2 = []
    for i in range(0, 5000):
        t = i / 1000
        times.append(t)
        value1.append(lambda_function(t, 5))
        value2.append(lambda_function(t, 0))
    pyplot.plot(times,value1)
    pyplot.plot(times,value2)
    pyplot.show()


def calculus(func, s, t, interval):
    delta = (t-s)/interval
    ans = 0
    for i in range(0, interval):
        ans += func(s + i*delta) * delta
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
        T = 900
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

    def lam(self, d):
        tmp = 0
        t = 100
        for point in self.point_list:
            if point.timestep > t:
                break
            tmp = tmp + self.A[d][point.dim] * math.exp(-self.w * (t - point.timestep))
        tmp += self.u[d]
        return tmp

#
# gen_points()
# draw_picture()


generat = OneDimHaw()
generat.gen_points()
generat.draw_qq_pic()