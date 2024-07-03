from src.features.process_utils import (
    joining_meteorological_data,
    identify_neighbours,
    process_drought_data,
)


def process():
    joining_meteorological_data()
    identify_neighbours()
    process_drought_data()


if __name__ == '__main__':
    process()
