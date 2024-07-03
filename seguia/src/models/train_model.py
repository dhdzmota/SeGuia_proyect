import xgboost

from src.data.utils import (
    get_general_path, join_paths, save_as_pickle
)
from src.models.model_utils import (
    RANDOM_SEED, read_xy_set

)

MODEL_PATH = 'models/'
MODEL_FILE = 'SeGuia.pkl'

PROCESSED_DATA = 'data/processed/'


# municipal_information_path = join_paths(general_path, INTERIM_DATA, MUNICIPAL_DATA)
# municipal_information = pd.read_pickle(municipal_information_path)
def training():
    general_path = get_general_path()
    x_train, y_train = read_xy_set('train')
    x_dev, y_dev = read_xy_set('dev')
    # x_oot, y_oot = read_xy_set('oot')
    # x_test, y_test = read_xy_set('test')

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    params = {
        'n_estimators': 10000,
        'max_depth': 6,
        'gamma': 0.4,
        'learning_rate': 0.001,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'colsample_bylevel': 0.8,
        'colsample_bynode': 0.8,
        'reg_alpha': 10,
        'reg_lambda': 10,
        'scale_pos_weight': scale_pos_weight,
        'base_score': 0.5,
        'random_state': RANDOM_SEED,
        'eval_metric': ['aucpr', 'auc', 'logloss'],
        'early_stopping_rounds': 3,
    }
    model = xgboost.XGBClassifier(**params)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_train, y_train), (x_dev, y_dev)],
    )
    model_path_file = join_paths(general_path, MODEL_PATH, MODEL_FILE)
    print('Saving Model...')
    save_as_pickle(what=model, where=model_path_file)
