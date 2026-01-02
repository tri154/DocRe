from preprocessing import Preprocessing
from config import Config

if __name__ == '__main__':
    cfg = Config()
    pre = Preprocessing(cfg)
    train_set, dev_set, test_set = pre.train_set, pre.dev_set, pre.test_set
