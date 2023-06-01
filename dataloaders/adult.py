from sklearn.preprocessing import MaxAbsScaler
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult

from .dataloader import get_dataloaders


def get_adult_dataloaders(train_bs=64, test_bs=64, privilege_mode='sex'):
    if privilege_mode == 'sex':
        privileged_groups, unprivileged_groups = [{'sex': 1}], [{'sex': 0}]
    elif privilege_mode == 'race':
        privileged_groups, unprivileged_groups = [{'race': 1}], [{'race': 0}]

    dataset_orig = load_preproc_data_adult([privilege_mode])
    data_orig_train, data_orig_val = dataset_orig.split([0.65], shuffle=True)
    data_orig_val, data_orig_test = data_orig_val.split([0.43], shuffle=True) # 0.43 * 0.35 = 0.15

    preproc = MaxAbsScaler()
    data_orig_train.features = preproc.fit_transform(data_orig_train.features)
    data_orig_val.features = preproc.fit_transform(data_orig_val.features)
    data_orig_test.features = preproc.fit_transform(data_orig_test.features)
    train_loader, val_loader, test_loader = get_dataloaders(data_orig_train, data_orig_val,
                                                            data_orig_test, train_bs=train_bs,
                                                            test_bs=test_bs)
    data = {
        'task_type': 'bin-class',
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': data_orig_train,
        'val_dataset': data_orig_val,
        'test_dataset': data_orig_test,
        'privileged_groups': privileged_groups,
        'unprivileged_groups': unprivileged_groups
    }
    return data
