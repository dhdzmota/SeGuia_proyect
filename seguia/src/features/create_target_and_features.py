from src.features.build_features import (
    create_target,
    create_initial_features,
    create_meteorological_features,
    merge_initial_and_meteorological_features,
    create_neighbour_features,
)


def process():
    create_target()
    create_initial_features()
    create_meteorological_features()
    merge_initial_and_meteorological_features()
    create_neighbour_features()


if __name__ == '__main__':
    process()
