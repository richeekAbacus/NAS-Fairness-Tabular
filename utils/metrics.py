import math
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


def print_all_metrics(classification_metric):
    print("Disparate Impact = {}".format(classification_metric.disparate_impact()))
    print("Statistical Parity Difference = {}".format(classification_metric.statistical_parity_difference()))
    print("Average Odds Difference = {}".format(classification_metric.average_odds_difference()))
    print("Average Absolute Odds Difference = {}".format(classification_metric.average_abs_odds_difference()))
    print("Equal Opportunity Difference = {}".format(classification_metric.equal_opportunity_difference()))        


def get_fairness_obj(classification_metric, metric_name):
    if metric_name == 'disparate_impact':
        return abs(1 - classification_metric.disparate_impact())
    elif metric_name == 'statistical_parity_difference':
        return abs(classification_metric.statistical_parity_difference())
    elif metric_name == 'average_odds_difference':
        return abs(classification_metric.average_odds_difference())
    elif metric_name == 'average_abs_odds_difference':
        return abs(classification_metric.average_abs_odds_difference())
    elif metric_name == 'equal_opportunity_difference':
        return abs(classification_metric.equal_opportunity_difference())
    else:
        raise NotImplementedError