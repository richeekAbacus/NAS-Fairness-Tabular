from aif360.algorithms.preprocessing import DisparateImpactRemover, Reweighing, LFR, OptimPreproc
from aif360.algorithms.inprocessing import (AdversarialDebiasing, GerryFairClassifier,
                                            MetaFairClassifier, PrejudiceRemover,
                                            ExponentiatedGradientReduction, GridSearchReduction)
from aif360.algorithms.postprocessing import (CalibratedEqOddsPostprocessing, EqOddsPostprocessing,
                                              RejectOptionClassification)

from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

from aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools import OptTools
from aif360.algorithms.preprocessing.optim_preproc_helpers.distortion_functions import get_distortion_adult, get_distortion_compas
from aif360.algorithms.preprocessing.optim_preproc_helpers.data_preproc_functions import load_preproc_data_adult, load_preproc_data_compas

from aif360.datasets import AdultDataset, CompasDataset

from dataloaders import ACSIncomeFolktablesDataset, get_distortion_acs_income

import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
tf.set_random_seed(1234)
np.random.seed(1234)


parser = argparse.ArgumentParser(description='Run debiasing on Adult dataset')
parser.add_argument("--dataset", type=str, default="adult",
                    choices=["adult", "compas", "acs-income"],
                    help="dataset to use")
parser.add_argument("--debiaser", type=str, default="adversarial_debiasing",
                    choices=["disparate_impact_remover",
                             "reweighing",
                             "lfr",
                             "optim_proc",
                             "adversarial_debiasing",
                             "gerryfair",
                             "metafair",
                             "prejudice_remover",
                             "exponentiated_gradient_reduction",
                             "grid_search_reduction",
                             "calibrated_eq_odds",
                             "eq_odds",
                             "reject_option_classification",
                             "logistic-regression"],
                    help="debiasing algorithm to use")
parser.add_argument("--privilege_mode", type=str, default='sex',
                    help="privileged group for the dataset")
parser.add_argument("--log", type=str, default="",
                    help="log file to write the test results to")
args = parser.parse_args()


preprocessing = {"disparate_impact_remover", "reweighing", "lfr", "optim_proc"}
inprocessing = {"adversarial_debiasing", "gerryfair", "metafair", "prejudice_remover",
                "exponentiated_gradient_reduction", "grid_search_reduction"}
postprocessing = {"calibrated_eq_odds", "eq_odds", "reject_option_classification"}

def compas_preproc(df):
    df['c_charge_degree'] = df['c_charge_degree'].replace(['F', 'M'], [0, 1])
    return df

def main():
    if args.dataset == "adult":
        dd = load_preproc_data_adult([args.privilege_mode])
        # dd = AdultDataset(protected_attribute_names=[args.privilege_mode],
        #                   privileged_classes=[['Male']], categorical_features=[],
        #                   features_to_keep=['age', 'education-num', 'capital-gain',
        #                                     'capital-loss', 'hours-per-week'])
    elif args.dataset == "compas":
        dd = load_preproc_data_compas([args.privilege_mode])
        # dd = CompasDataset(protected_attribute_names=[args.privilege_mode], favorable_classes=[0],
        #                    privileged_classes=[['Female']], categorical_features=[],
        #                    features_to_keep=['age', 'priors_count', 'c_charge_degree'],
        #                    custom_preprocessing=compas_preproc)
    elif args.dataset == "acs-income":
        dd = ACSIncomeFolktablesDataset(protected_attr_name=args.privilege_mode)

    train_dataset, val_dataset = dd.split([0.65], shuffle=True)
    val_dataset, test_dataset = val_dataset.split([0.43], shuffle=True) # 0.43 * 0.35 = 0.15

    privileged_groups = [{args.privilege_mode: 1}]
    unprivileged_groups = [{args.privilege_mode: 0}]

    index = train_dataset.feature_names.index(args.privilege_mode)

    # Metric for the original dataset #################################################################
    print("\n\nStats of Initial Dataset: ")
    metric_orig_train = BinaryLabelDatasetMetric(train_dataset, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
    print("Train set: Initial disparate impact in source dataset = %f" % metric_orig_train.disparate_impact())
    metric_orig_test = BinaryLabelDatasetMetric(test_dataset, 
                                                unprivileged_groups=unprivileged_groups,
                                                privileged_groups=privileged_groups)
    print("Test set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_test.mean_difference())
    print("Test set: Initial disparate impact in source dataset = %f" % metric_orig_test.disparate_impact())

    # Debiasing techniques ############################################################################
    if args.debiaser in preprocessing:
        lmod = None
        X_tr, y_tr, X_te = None, None, None
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

        elif args.debiaser == "disparate_impact_remover":
            debias_model = DisparateImpactRemover(repair_level=1.0)
            train_repd = debias_model.fit_transform(train_dataset)
            test_repd = debias_model.fit_transform(test_dataset)

            metrics_train_dataset = BinaryLabelDatasetMetric(train_repd,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
            print("\nTRANSFORMED Train set: Difference in mean outcomes between unprivileged and privileged groups = %f\n" % metrics_train_dataset.mean_difference())
            
            X_tr = np.delete(train_repd.features, index, axis=1)
            y_tr = train_repd.labels.ravel()
            X_te = np.delete(test_repd.features, index, axis=1)

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

        elif args.debiaser == "optim_proc":
            distort_fn = {
                "adult": get_distortion_adult,
                "compas": get_distortion_compas,
                "acs-income": get_distortion_acs_income
            }
            optim_options = {
                "distortion_fun": distort_fn[args.dataset],
                "epsilon": 0.05,
                "clist": [0.99, 1.99, 2.99],
                "dlist": [.1, 0.05, 0]
            }
                
            debias_model = OptimPreproc(OptTools, optim_options)
            debias_model = debias_model.fit(train_dataset)

            train_repd = debias_model.transform(train_dataset, transform_Y=True)
            test_repd = debias_model.transform(test_dataset, transform_Y=True)

            metrics_train_dataset = BinaryLabelDatasetMetric(train_repd,
                                                            unprivileged_groups=unprivileged_groups,
                                                            privileged_groups=privileged_groups)
            print("\nTRANSFORMED Train set: Difference in mean outcomes between unprivileged and privileged groups = %f" % metrics_train_dataset.mean_difference())

        if lmod is None:
            X_tr = train_repd.features if X_tr is None else X_tr
            y_tr = train_repd.labels.ravel() if y_tr is None else y_tr
            X_te = test_repd.features if X_te is None else X_te

            lmod = LogisticRegression(class_weight='balanced', solver='liblinear')
            lmod.fit(X_tr, y_tr)

        dataset_debiasing_test = test_dataset.copy()
        dataset_debiasing_test.labels = lmod.predict(X_te).reshape(-1,1)

    elif args.debiaser in inprocessing:
        if args.debiaser == "adversarial_debiasing":
            sess = tf.Session()
            debias_model = AdversarialDebiasing(privileged_groups = privileged_groups,
                                    unprivileged_groups = unprivileged_groups,
                                    scope_name='debiased_classifier',
                                    debias=True, sess=sess, num_epochs=200)
            debias_model.fit(train_dataset)
            dataset_debiasing_test = debias_model.predict(test_dataset)

        elif args.debiaser == "gerryfair":
            C, print_flag, gamma, max_iterations = 1000, True, 0.005, 500
            #! Need to use LogisticRegression as the base classifier, instead its LinearRegression now
            debias_model = GerryFairClassifier(C=C, printflag=print_flag, gamma=gamma,
                                               max_iters=max_iterations, heatmapflag=False)
            debias_model.fit(train_dataset, early_termination=True)
            dataset_debiasing_test = debias_model.predict(test_dataset)
        
        elif args.debiaser == "metafair":
            #! Fails for different versions of the same dataset for reasons I dont fully understand
            debias_model = MetaFairClassifier(tau=0.9, sensitive_attr=args.privilege_mode,
                                              type='fdr').fit(train_dataset)
            dataset_debiasing_test = debias_model.predict(test_dataset)

        elif args.debiaser == "prejudice_remover":
            debias_model = PrejudiceRemover(eta=1.0, sensitive_attr=args.privilege_mode)
            debias_model.fit(train_dataset)
            dataset_debiasing_test = debias_model.predict(test_dataset)
        
        elif args.debiaser == "exponentiated_gradient_reduction":
            estimator = LogisticRegression(solver='lbfgs', max_iter=1000)
            debias_model = ExponentiatedGradientReduction(estimator=estimator, 
                                                          constraints="EqualizedOdds",
                                                          drop_prot_attr=False)
            debias_model.fit(train_dataset)
            dataset_debiasing_test = debias_model.predict(test_dataset)

        elif args.debiaser == "grid_search_reduction":
            estimator = LogisticRegression(solver='liblinear', random_state=1234)
            debias_model = GridSearchReduction(estimator=estimator, constraints="EqualizedOdds",
                                               prot_attr=args.privilege_mode, grid_size=20,
                                               grid_limit=30, drop_prot_attr=False)
            debias_model.fit(train_dataset)
            dataset_debiasing_test = debias_model.predict(test_dataset)

    elif args.debiaser in postprocessing or args.debiaser == "logistic-regression":
        scale_orig = StandardScaler()
        X_train = scale_orig.fit_transform(train_dataset.features)
        y_train = train_dataset.labels.ravel()
        lmod = LogisticRegression()
        lmod.fit(X_train, y_train)

        fav_idx = np.where(lmod.classes_ == train_dataset.favorable_label)[0][0]

        # Prediction probs for validation and testing data
        X_valid = scale_orig.transform(val_dataset.features)
        y_valid_pred_prob = lmod.predict_proba(X_valid)[:,fav_idx]

        X_test = scale_orig.transform(test_dataset.features)
        y_test_pred_prob = lmod.predict_proba(X_test)[:,fav_idx]

        # Placeholder for predicted and transformed datasets
        val_pred = val_dataset.copy(deepcopy=True)
        test_pred = test_dataset.copy(deepcopy=True)

        class_thresh = 0.5
        val_pred.scores = y_valid_pred_prob.reshape(-1,1)
        test_pred.scores = y_test_pred_prob.reshape(-1,1)

        y_valid_pred = np.zeros_like(val_pred.labels)
        y_valid_pred[y_valid_pred_prob >= class_thresh] = val_pred.favorable_label
        y_valid_pred[~(y_valid_pred_prob >= class_thresh)] = val_pred.unfavorable_label
        val_pred.labels = y_valid_pred
            
        y_test_pred = np.zeros_like(test_pred.labels)
        y_test_pred[y_test_pred_prob >= class_thresh] = test_pred.favorable_label
        y_test_pred[~(y_test_pred_prob >= class_thresh)] = test_pred.unfavorable_label
        test_pred.labels = y_test_pred

        if args.debiaser == "calibrated_eq_odds":
            debias_model = CalibratedEqOddsPostprocessing(privileged_groups=privileged_groups,
                                                 unprivileged_groups=unprivileged_groups,
                                                 cost_constraint='fnr',
                                                 seed=1234)
            debias_model = debias_model.fit(val_dataset, val_pred)
            dataset_debiasing_test = debias_model.predict(test_pred)
        
        elif args.debiaser == "eq_odds":
            debias_model = EqOddsPostprocessing(privileged_groups=privileged_groups,
                                                unprivileged_groups=unprivileged_groups,
                                                seed=1234)
            debias_model = debias_model.fit(val_dataset, val_pred)
            dataset_debiasing_test = debias_model.predict(test_pred)

        elif args.debiaser == "reject_option_classification":
            debias_model = RejectOptionClassification(privileged_groups=privileged_groups,
                                                      unprivileged_groups=unprivileged_groups,
                                                      low_class_thresh=0.01, high_class_thresh=0.99,
                                                      num_class_thresh=100, num_ROC_margin=50,
                                                      metric_name="Statistical parity difference",
                                                      metric_lb=-0.05, metric_ub=0.05)
            debias_model = debias_model.fit(val_dataset, val_pred)
            dataset_debiasing_test = debias_model.predict(test_pred)
        
        elif args.debiaser == "logistic-regression": # No debiasing baseline
            dataset_debiasing_test = test_pred


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

    if args.log != "":
        with open(args.log, 'a') as file:
            print("Dataset: ", args.dataset, ", Debiaser: ", args.debiaser,
                  ", Privilege Mode: ", args.privilege_mode, file=file)
            print("Classification accuracy", "Balanced classification accuracy",
                  "Mean Difference", "Disparate impact", "Equal opportunity difference",
                  "Average odds difference", "Theil_index", file=file, sep=",")
            print(classified_metric_debiasing_test.accuracy(),
                bal_acc_debiasing_test,
                metric_dataset_debiasing_test.mean_difference(),
                classified_metric_debiasing_test.disparate_impact(),
                classified_metric_debiasing_test.equal_opportunity_difference(),
                classified_metric_debiasing_test.average_odds_difference(),
                classified_metric_debiasing_test.theil_index(),
                sep=',', end='\n\n', file=file)

if __name__ == '__main__':
    main()