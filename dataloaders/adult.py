from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

def load_adult_data():
    dataset_orig = load_preproc_data_adult()

    privileged_groups = [{'sex': 1, 'race': 1}]
    unprivileged_groups = [{'sex': 0, 'race': 0}]

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    return dataset_orig_train, dataset_orig_test, privileged_groups, unprivileged_groups