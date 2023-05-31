from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from dataloaders import get_adult_dataloaders


privilege_mode = 'sex'

data_dict = get_adult_dataloaders(privilege_mode=privilege_mode)

train_dataset = data_dict['train_dataset']
val_dataset = data_dict['val_dataset']
test_dataset = data_dict['test_dataset']

privileged_groups = data_dict['privileged_groups']
unprivileged_groups = data_dict['unprivileged_groups']

# Metric for the original dataset
metric_orig_train = BinaryLabelDatasetMetric(train_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
metric_orig_val = BinaryLabelDatasetMetric(val_dataset,
                                           unprivileged_groups=unprivileged_groups,
                                           privileged_groups=privileged_groups)
print("Val set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_val.mean_difference())
metric_orig_test = BinaryLabelDatasetMetric(test_dataset, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())

