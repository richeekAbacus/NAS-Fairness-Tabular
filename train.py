import rtdl
import argparse

import torch
import torch.nn.functional as F

from sklearn.preprocessing import MaxAbsScaler

from aif360.metrics import BinaryLabelDatasetMetric

from dataloaders import load_adult_data, get_dataloaders
from utils import train, test

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult')
parser.add_argument('--train_bs', type=int, default=64)
parser.add_argument('--test_bs', type=int, default=64)
parser.add_argument('--model', type=str, default='FTTransformer')
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()

#! TODO: Write a new dataloader for the adult dataset. This one drops a lot of columns!

# print out some labels, names, etc.
# print(dataset_orig_train.features.shape)
# print(dataset_orig_train.favorable_label, dataset_orig_train.unfavorable_label)
# print(dataset_orig_train.protected_attribute_names)
# print(dataset_orig_train.privileged_protected_attributes, 
#       dataset_orig_train.unprivileged_protected_attributes)
# print(dataset_orig_train.feature_names)
# print(dataset_orig_train.features[:5])


def data_adult(train_bs=64, test_bs=64):
    TRAIN_BS = train_bs
    TEST_BS = test_bs
    data_orig_train, data_orig_test, privileged_groups, unprivileged_groups = load_adult_data()
    preproc = MaxAbsScaler()
    data_orig_train.features = preproc.fit_transform(data_orig_train.features)
    data_orig_test.features = preproc.fit_transform(data_orig_test.features)
    train_loader, test_loader = get_dataloaders(data_orig_train, data_orig_test,
                                                train_bs=TRAIN_BS, test_bs=TEST_BS)
    return train_loader, test_loader, privileged_groups, unprivileged_groups

DATA_FN_MAP = {
    'adult': data_adult
}
BIN_CLASS = {'adult'} # datasets which are binary classification tasks

if __name__ == '__main__':
    train_bs, test_bs = args.train_bs, args.test_bs
    train_loader, test_loader, privileged_groups, unprivileged_groups = \
        DATA_FN_MAP[args.dataset](train_bs, test_bs)

    num_features = train_loader.dataset[0][0].shape[0]

    if args.model == 'FTTransformer':
        model = rtdl.FTTransformer.make_default(
            n_num_features=num_features,
            cat_cardinalities=None,
            last_layer_query_idx=[-1],
            d_out=1
        )
    else:
        raise NotImplementedError
    model.to('cuda')

    if args.model == 'FTTransformer':
        optimizer = model.make_default_optimizer()
    else:
        raise NotImplementedError

    if args.dataset in BIN_CLASS:
        loss_fn = F.binary_cross_entropy_with_logits

    for epoch in range(args.epochs):
        model = train(model, optimizer, train_loader, loss_fn, epoch, 100)
        _, _, pred_y = test(model, test_loader, loss_fn, epoch)


# # Metric for the original dataset
# metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)
# print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
# metric_orig_test = BinaryLabelDatasetMetric(dataset_orig_test, 
#                                              unprivileged_groups=unprivileged_groups,
#                                              privileged_groups=privileged_groups)
# print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())


# min_max_scaler = MaxAbsScaler()
# dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
# dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)
# metric_scaled_train = BinaryLabelDatasetMetric(dataset_orig_train, 
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_train.mean_difference())
# metric_scaled_test = BinaryLabelDatasetMetric(dataset_orig_test, 
#                              unprivileged_groups=unprivileged_groups,
#                              privileged_groups=privileged_groups)
# print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_scaled_test.mean_difference())
