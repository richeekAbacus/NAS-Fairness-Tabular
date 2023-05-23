from sklearn.preprocessing import MaxAbsScaler
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

from .dataloader import get_dataloaders


def get_adult_dataloaders(train_bs=64, test_bs=64):
    privileged_groups = [{'sex': 1, 'race': 1}]
    unprivileged_groups = [{'sex': 0, 'race': 0}]
    dataset_orig = load_preproc_data_adult()
    data_orig_train, data_orig_test = dataset_orig.split([0.7], shuffle=True)

    preproc = MaxAbsScaler()
    data_orig_train.features = preproc.fit_transform(data_orig_train.features)
    data_orig_test.features = preproc.fit_transform(data_orig_test.features)
    train_loader, test_loader = get_dataloaders(data_orig_train, data_orig_test,
                                                train_bs=train_bs, test_bs=test_bs)
    data = {
        'task_type': 'bin-class',
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_dataset': data_orig_train,
        'test_dataset': data_orig_test,
        'privileged_groups': privileged_groups,
        'unprivileged_groups': unprivileged_groups
    }
    return data
