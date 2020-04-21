class Config:
    dim = 1
    train_data_path = "data/atm_train.csv"
    test_data_path = "data/atm_test.csv"
    output_path = ""

    def __init__(self, d=0):
        self.d = d