from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing, LFR, OptimPreproc
from aif360.algorithms.inprocessing import AdversarialDebiasing

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.set_random_seed(42)

from dataloaders import get_adult_dataloaders


parser = argparse.ArgumentParser(description='Run debiasing on Adult dataset')
parser.add_argument("--dataset", type=str, default="adult",
                    help="dataset to use")
parser.add_argument("--debiaser", type=str, default="adversarial_debiasing",
                    choices=["adversarial_debiasing",
                             "disparate_impact_remover",
                             "reweighing",
                             "lfr",
                             "optim_proc"],
                    help="debiasing algorithm to use")
args = parser.parse_args()


preprocessing = {"disparate_impact_remover", "reweighing", "lfr", "optim_proc"}
inprocessing = {"adversarial_debiasing"}

def main():
    privilege_mode = 'sex' # hardcoded for now

    data_dict = get_adult_dataloaders(privilege_mode=privilege_mode)

    train_dataset = data_dict['train_dataset']
    val_dataset = data_dict['val_dataset']
    test_dataset = data_dict['test_dataset']

    privileged_groups = data_dict['privileged_groups']
    unprivileged_groups = data_dict['unprivileged_groups']

    index = train_dataset.feature_names.index(privilege_mode)

    # Metric for the original dataset #################################################################
    print("\n\nStats of Initial Dataset: ")
    metric_orig_train = BinaryLabelDatasetMetric(train_dataset, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
    print("Train set: Initial disparate impact in source dataset = %f" % metric_orig_train.disparate_impact())
    metric_orig_val = BinaryLabelDatasetMetric(val_dataset,
                                            unprivileged_groups=unprivileged_groups,
                                            privileged_groups=privileged_groups)
    print("Val set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_val.mean_difference())
    print("Val set: Initial disparate impact in source dataset = %f" % metric_orig_val.disparate_impact())
    metric_orig_test = BinaryLabelDatasetMetric(test_dataset, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())
    print("Test set: Initial disparate impact in source dataset = %f" % metric_orig_test.disparate_impact())


    if args.debiaser in preprocessing:
        if args.debiaser == "reweighing":
            debias_model = Reweighing(unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)
            debias_model.fit(train_dataset)
            train_repd = debias_model.transform(train_dataset)
            test_repd = debias_model.transform(test_dataset)
            metrics_train_dataset = BinaryLabelDatasetMetric(train_repd,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
            print("\nTRANSFORMED Train set: Difference in mean outcomes between unprivileged and privileged groups = %f\n" % metrics_train_dataset.mean_difference())

            X_tr = train_repd.features
            y_tr = train_repd.labels.ravel()
            w_tr = train_repd.instance_weights.ravel()
            X_te = test_dataset.features

            lmod = LogisticRegression()
            lmod.fit(X_tr, y_tr, sample_weight=w_tr)

            dataset_debiasing_test = test_dataset.copy()
            dataset_debiasing_test.labels = lmod.predict(X_te).reshape(-1,1)

        elif args.debiaser == "disparate_impact_remover":
            debias_model = DisparateImpactRemover(repair_level=1.0)
            train_repd = debias_model.fit_transform(train_dataset)
            val_repd = debias_model.fit_transform(val_dataset)
            test_repd = debias_model.fit_transform(test_dataset)

            metrics_train_dataset = BinaryLabelDatasetMetric(train_repd,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
            print("\nTRANSFORMED Train set: Difference in mean outcomes between unprivileged and privileged groups = %f\n" % metrics_train_dataset.mean_difference())
            
            X_tr = np.delete(train_repd.features, index, axis=1)
            y_tr = train_repd.labels.ravel()
            X_te = np.delete(test_repd.features, index, axis=1)
            
            lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
            lmod.fit(X_tr, y_tr)

            dataset_debiasing_test = test_dataset.copy()
            dataset_debiasing_test.labels = lmod.predict(X_te).reshape(-1,1)
        
        elif args.debiaser == "lfr":
            debias_model = LFR(unprivileged_groups=unprivileged_groups,
                               privileged_groups=privileged_groups,
                               k=10, Ax=0.1, Ay=0.9, Az=2.0,
                               verbose=1)
            debias_model = debias_model.fit(train_dataset, maxiter=5000, maxfun=5000)

            train_repd = debias_model.transform(train_dataset)
            test_repd = debias_model.transform(test_dataset)

            metrics_train_dataset = BinaryLabelDatasetMetric(train_repd,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
            print("\nTRANSFORMED Train set: Difference in mean outcomes between unprivileged and privileged groups = %f\n" % metrics_train_dataset.mean_difference())
            
            dataset_debiasing_test = test_dataset.copy()
            dataset_debiasing_test.labels = test_repd.labels
        

    elif args.debiaser in inprocessing:
        if args.debiaser == "adversarial_debiasing":
            sess = tf.Session()
            debias_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                                    unprivileged_groups = unprivileged_groups,
                                    scope_name='debiased_classifier',
                                    debias=True, sess=sess)
            debias_model.fit(train_dataset)
            dataset_debiasing_train = debias_model.predict(train_dataset)
            dataset_debiasing_val = debias_model.predict(val_dataset)
            dataset_debiasing_test = debias_model.predict(test_dataset)


    # Metric for the debiased dataset #################################################################
    print("\n\nDebiased Test Results: ")
    metric_dataset_debiasing_test = BinaryLabelDatasetMetric(dataset_debiasing_test, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_dataset_debiasing_test.mean_difference())


    classified_metric_debiasing_test = ClassificationMetric(test_dataset, 
                                                    dataset_debiasing_test,
                                                    unprivileged_groups=unprivileged_groups,
                                                    privileged_groups=privileged_groups)
    print("Test set: Classification accuracy = %f" % classified_metric_debiasing_test.accuracy())
    TPR = classified_metric_debiasing_test.true_positive_rate()
    TNR = classified_metric_debiasing_test.true_negative_rate()
    bal_acc_debiasing_test = 0.5*(TPR+TNR)
    print("Test set: Balanced classification accuracy = %f" % bal_acc_debiasing_test)
    print("Test set: Disparate impact = %f" % classified_metric_debiasing_test.disparate_impact())
    print("Test set: Equal opportunity difference = %f" % classified_metric_debiasing_test.equal_opportunity_difference())
    print("Test set: Average odds difference = %f" % classified_metric_debiasing_test.average_odds_difference())
    print("Test set: Theil_index = %f" % classified_metric_debiasing_test.theil_index())


if __name__ == '__main__':
    main()