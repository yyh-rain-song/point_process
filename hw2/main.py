from learner import HawkesProcessLearner
from Config import Config
from data_process import Dataset
from matplotlib import pyplot as plt
from data_process import Sample

train_data = Dataset("train")
test_data = Dataset("test")
print("==== dataset read ====")
dim = Config.dim

lam = [0.6, 0.02]
row = 0.3
beta = Config.beta

learner = HawkesProcessLearner(lam, row, beta, train_data, test_data, dim)
L_history = learner.train(epoc=1000, verbose=True)
plt.plot(L_history)
plt.show()