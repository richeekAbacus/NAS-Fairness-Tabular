from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def get_fairness_metrics(dataset, pred_y, unprivileged_groups, privileged_groups,
                         dataset_metric=True, classification_metric=True):
    modified_dataset = dataset.copy(deepcopy=True)
    modified_dataset.labels = pred_y.cpu().numpy()
    dataset_m, classification_m = None, None
    if dataset_metric:
        dataset_m = BinaryLabelDatasetMetric(modified_dataset,
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
    if classification_metric:
        classification_m = ClassificationMetric(dataset,
                                                modified_dataset,
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    return dataset_m, classification_m
