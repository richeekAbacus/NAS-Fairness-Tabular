import rtdl
import argparse

import torch
import torch.nn.functional as F

from sklearn.preprocessing import MaxAbsScaler

from aif360.metrics import BinaryLabelDatasetMetric

from dataloaders import load_adult_data, get_dataloaders

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult')
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

def preprocess(dataset, preproc=MaxAbsScaler()):
    dataset.features = preproc.fit_transform(dataset.features)
    return dataset



data_orig_train, data_orig_test, privileged_groups, unprivileged_groups = load_adult_data()
data_orig_train = preprocess(data_orig_train)
data_orig_test = preprocess(data_orig_test)
train_loader, test_loader = get_dataloaders(data_orig_train, data_orig_test, train_bs=64, test_bs=64)



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

# model = rtdl.FTTransformer.make_default(
#     n_num_features=dataset_orig_train.features.shape[1],
#     cat_cardinalities=None,
#     last_layer_query_idx=[-1],
#     d_out=1
# )

# model.to('cuda')

# optimizer = model.make_default_optimizer()

# loss_fn = F.binary_cross_entropy_with_logits

# print(model(torch.Tensor(dataset_orig_train.features[0]).to('cuda'), None))

