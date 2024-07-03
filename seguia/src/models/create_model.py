from src.models.split_data import split_data
from src.models.train_model import training


def process():
    split_data()
    training()


if __name__ == '__main__':
    process()
