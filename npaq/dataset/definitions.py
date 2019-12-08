import os

DATASET_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FOLDER = os.path.join(DATASET_DIR, 'real_data')

UCI_FOLDER = os.path.join(DATA_FOLDER, 'uci_adult'
UCI_TEST_DATA_PATH = os.path.join(UCI_FOLDER, 'adult.data')
UCI_TRAIN_DATA_PATH = os.path.join(UCI_FOLDER, 'adult.test')
