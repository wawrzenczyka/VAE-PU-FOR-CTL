# %%
from occ_all_tests_lib import *
from occ_test_base_general import run_general_tests
import occ_datasets

DATASET_TYPE = 'ECODdata'

datasets = occ_datasets.ECOD_dataset_list
# datasets = [
#     ('Arrhythmia', 'mat'),
# ]

def get_all_dataset_configs(train_ratio=0.6):
    all_configs = []

    for (dataset, format) in datasets:
        test_case_name = f'({format}) {dataset}'

        def get_dataset_function(dataset=dataset, format=format):
            X, y = occ_datasets.load_dataset(dataset, format)
            X_train_orig, X_test_orig, y_test_orig = occ_datasets.split_occ_dataset(X, y, train_ratio=train_ratio)
            return X_train_orig, X_test_orig, y_test_orig
        
        all_configs.append((test_case_name, get_dataset_function))
    
    return all_configs

run_general_tests(DATASET_TYPE, get_all_dataset_configs)

# %%
