from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

def get_fairness_metrics(dataset, pred_y, scores_y, unprivileged_groups, privileged_groups,
                         dataset_metric=True, classification_metric=True):
    modified_dataset = dataset.copy(deepcopy=True)
    modified_dataset.scores = scores_y.cpu().numpy()
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

def log_fairness_metrics(classification_metric, asdict=False):
    TPR = classification_metric.true_positive_rate()
    TNR = classification_metric.true_negative_rate()
    bal_acc_debiasing_test = 0.5*(TPR+TNR)
    if asdict:
        return {'accuracy': classification_metric.accuracy(),
                'bal_acc': bal_acc_debiasing_test,
                'statistical_parity_difference': classification_metric.statistical_parity_difference(),
                'disparate_impact': classification_metric.disparate_impact(),
                'equal_opportunity_difference': classification_metric.equal_opportunity_difference(),
                'average_odds_difference': classification_metric.average_odds_difference(),
                'theil_index': classification_metric.theil_index()}
    else:
        return (classification_metric.accuracy(),
                bal_acc_debiasing_test,
                classification_metric.statistical_parity_difference(),
                classification_metric.disparate_impact(),
                classification_metric.equal_opportunity_difference(),
                classification_metric.average_odds_difference(),
                classification_metric.theil_index())

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