import os

COUNTER='scalmc'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.abspath(os.path.join(ROOT_DIR, '..', 'data'))

BNN2CNF_PATH = os.path.join(ROOT_DIR, '../mlp2cnf/bin/bnn2cnf')
EXAMPLE_PATH = os.path.join(ROOT_DIR, '../mlp2cnf/bin/example')

# Tests paths
TEST_SAMPLES_DIR = os.path.join(ROOT_DIR, '..', 'test_samples')
TEST_FORMULAS_DIR = os.path.join(ROOT_DIR, '..', 'test_formulas')
TESTS_INPUT = os.path.join(ROOT_DIR, 'tests', 'tests_input')
ROOT_BNN = os.path.join(TESTS_INPUT, 'bnn_tests')
BNN_TEST_CFG = os.path.join(TESTS_INPUT, 'bnn_tests_cfg')

# XXX hack should put it as option but too many options
# UNCOMMENT THIS TO RUN ON THE SHARED STORAGE
RESULTS_DIR = os.path.join(ROOT_DIR, '..')
#RESULTS_DIR = '/home/teo/test-nn/experiments'
TRAINED_MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
TRAINED_MODELS_CP_DIR = os.path.join(RESULTS_DIR, 'models_checkpoint')
#FORMULAS_DIR = os.path.join(RESULTS_DIR, 'formulas')
#FORMULAS_DIR = os.path.join(RESULTS_DIR, 'formulas_card')
FORMULAS_DIR = os.path.join(RESULTS_DIR, 'formulas_cam_ready')
COUNT_OUT_DIR = os.path.join(RESULTS_DIR, 'output_' + COUNTER)
MNIST_SAMPLES = os.path.join(RESULTS_DIR, 'mnist_samples')
CONCRETE_IN_DIR = os.path.join(RESULTS_DIR, 'concrete_inputs')

# For canaries insertion
CANARY_DATASET_DIR = os.path.join(DATA_PATH, 'canary')

# For trojan attack
# TROJAN_RETRAIN_DATASET_DIR = ''
TROJAN_RETRAIN_DATASET_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'trojan'))
TROJAN_ORIGIN_DATA_DIR = ''
TROJAN_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'trojan'))
TROJAN_VERBOSE_DIR = os.path.join(TROJAN_DIR, 'verbose')
TROJAN_PREFC1_PATH = ''
ORIGIN_TROJAN_DATA_DIR = os.path.abspath(os.path.join(os.path.join(ROOT_DIR, '..', 'data'),
                                                      'trojan_data'))

# For differential privacy
GRADIENT_NORM_BOUND = 1.0
NOISE_SCALE = 1.0
GROUP_SIZE = 256

# For adversarial training
ADV_TRAIN_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..', 'adv_train'))
ADV_TRAIN_DATA_DIR = os.path.join(DATA_PATH, 'adv_train_data')

# For hmnist dataset
HMNIST_DATA_FOLDER = '/home/shiqi/project/test-nn/real_data/hmnistd/data'
# For diamond dataset
DIAMONDS_DATA_FOLDER = '/home/shiqi/project/test-nn/real_data/diamonds/data'

# constraints for dataset
UCI_CONSTRAINTS = '/mnt/storage/teo/npaq/ccs-submission/experiments/dataset_constraints/uci_adult-constraints.txt'

# Hardcoded values
TROJAN_IMGS='/mnt/storage/teo/npaq/ccs-submission/experiments/trojan_imgs/'
TROJAN_MASK='/mnt/storage/teo/npaq/ccs-submission/experiments/trojan_mask/'

TROJAN_TARGETS=[0,1,4,5,9]
TROJAN_EPOCHS=[1,10,30]
