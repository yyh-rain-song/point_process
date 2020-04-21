from learner import HawkesProcessLearner
from Config import Config
from data_process import Dataset
from matplotlib import pyplot as plt

train_data = Dataset(Config.train_data_path)
test_data = Dataset(Config.test_data_path)
print("==== dataset read ====")
dim = Config.dim

lam = [0.6, 0.02]
row = 0.3
beta = 1

learner = HawkesProcessLearner(lam, row, beta, train_data, test_data, dim)
L_history = learner.train(epoc=1000, verbose=True)
plt.plot(L_history)
plt.show()