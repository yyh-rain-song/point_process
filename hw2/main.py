from learner import HawkesProcessLearner
from Config import Config
from data_process import Dataset
import pickle
from data_process import Sample

train_data = Dataset("train")
test_data = Dataset("test")
print("==== dataset read ====")
dim = Config.dim

lam = Config.lam
row = Config.row
beta = Config.beta

learner = HawkesProcessLearner(lam, row, beta, train_data, test_data, dim)
A, mu = learner.train(epoc=100, verbose=True)
file = open("A.dat", "wb")
pickle.dump(A, file)
file2 = open("mu.dat", "wb")
pickle.dump(mu, file2)