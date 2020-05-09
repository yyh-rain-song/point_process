class Config:
    dim = 7
    train_data_path = "data/atm_train.csv"
    test_data_path = "data/atm_test.csv"
    cache_path = "cache"
    output_path = "output"
    batch_size = 64
    beta = 0.2
    lam = [0.6, 0.02]
    row = 0.3
