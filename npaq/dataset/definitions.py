import os

DATA_FOLDER = '/home/shiqi/project/test-nn/real_data'

HAM_FOLDER = os.path.join(DATA_FOLDER, 'hmnist')
HAM_RGBDATA_PATH = os.path.join(HAM_FOLDER, 'hmnist_8_8_RGB.csv')
HAM_LDATA_PATH = os.path.join(HAM_FOLDER, 'hmnist_8_8_L.csv')
HAM_METADATA_PATH = os.path.join(HAM_FOLDER, 'HAM10000_metadata.csv')

DIAMONDS_FOLDER = os.path.join(DATA_FOLDER, 'diamonds')
DIAMONDS_DATA_PATH = os.path.join(DIAMONDS_FOLDER, 'diamonds.csv')

BEER_FOLDER = os.path.join(DATA_FOLDER, 'beer')
BEER_LABEL_PATH = os.path.join(BEER_FOLDER, 'styleData.csv')
BEER_DATA_PATH = os.path.join(BEER_FOLDER, 'recipeData.csv')

SAC_FOLDER = '/home/shiqi/project/test-nn/real_data/SAC'
SAC_MATH_DATA_PATH = os.path.join(SAC_FOLDER, 'student-mat.csv')
SAC_POR_DATA_PATH = os.path.join(SAC_FOLDER, 'student-por.csv')

UCI_FOLDER = '/home/shiqi/project/test-nn/real_data/uci_adult'
UCI_TEST_DATA_PATH = os.path.join(UCI_FOLDER, 'adult.data')
UCI_TRAIN_DATA_PATH = os.path.join(UCI_FOLDER, 'adult.test')