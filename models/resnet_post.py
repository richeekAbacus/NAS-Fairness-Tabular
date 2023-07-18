import rtdl
import numpy as np

import torch
import torch.nn.functional as F

from .nas import NAS
from utils import train, test, get_fairness_metrics, print_all_metrics

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
)

from aif360.metrics import ClassificationMetric
from aif360.algorithms.postprocessing import (CalibratedEqOddsPostprocessing, EqOddsPostprocessing,
                                              RejectOptionClassification)


class ResNetPostSearch(NAS):
    def __init__(self, args, data_fn, fairness_metric=None, postprocessing_method="eq_odds") -> None:
        super(ResNetPostSearch, self).__init__(args, data_fn, fairness_metric)
        self.postprocessing_method = postprocessing_method

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        n_blocks = Integer("n_blocks", (4, 10), default=6)
        d_main = Categorical("d_main", [32, 64, 128, 256, 512], default=64)
        d_hidden_multiplier = Categorical("d_hidden_multiplier", [1, 1.5, 2, 4, 8], default=4)
        dropout_first = Float("dropout_first", (0.0, 0.5), default=0.25)
        dropout_second = Float("dropout_second", (0.0, 0.5), default=0.001)
        
        lr = Float("lr", (1e-5, 1e-3), default=1e-4, log=True)
        weight_decay = Float("weight_decay", (1e-6, 1e-3), default=1e-5, log=True)
        train_batch_size = Constant("train_batch_size", value=self.args.train_bs)
        test_batch_size = Constant("test_batch_size", value=self.args.test_bs)

        cs.add_hyperparameters([n_blocks, d_main, d_hidden_multiplier, dropout_first, dropout_second,
                                lr, weight_decay, train_batch_size, test_batch_size])
        
        return cs

    def create_and_train_model_from_config(self, config: Configuration, budget: int) -> tuple[float, float, float]:
        print(config)
        print("Budget:", budget)
        print("#"*50)

        data_dict = self.data_fn(config["train_batch_size"], config["test_batch_size"],
                                 privilege_mode=self.args.privilege_mode)
        
        train_dataset, val_dataset, test_dataset = data_dict['train_dataset'], data_dict['val_dataset'], data_dict['test_dataset']
        train_loader, val_loader, test_loader = data_dict['train_loader'], data_dict['val_loader'], data_dict['test_loader']
        num_features = train_loader.dataset[0][0].shape[0]

        print(config)

        model = rtdl.ResNet.make_baseline(
            d_in=num_features,
            n_blocks=config["n_blocks"],
            d_main=config["d_main"],
            d_hidden=int(config["d_hidden_multiplier"]*config["d_main"]),
            dropout_first=config["dropout_first"],
            dropout_second=config["dropout_second"],
            d_out=1
        )
        model.to('cuda')

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        if data_dict['task_type'] == 'bin-class':
            loss_fn = F.binary_cross_entropy_with_logits
        else:
            raise NotImplementedError

        for epoch in range(int(budget)):
            train(model, optimizer, train_loader, loss_fn, epoch, 100)

            loss, train_acc, _, _ = test(model, train_loader, loss_fn, epoch)
            print('Epoch: {}, Train Loss: {}, Train Accuracy: {}'.format(epoch, loss, train_acc))

            # validation metrics ##########################################################
            print("#"*30)
            loss, val_acc, pred_y_val, scores_y_val = test(model, val_loader, loss_fn, epoch)
            print('Epoch: {}, Val Loss: {}, Val Accuracy: {}'.format(epoch, loss, val_acc))

            val_data_metric, val_class_metric = get_fairness_metrics(val_dataset, 
                                                                     pred_y_val,
                                                                     scores_y_val,
                                                                     data_dict['unprivileged_groups'],
                                                                     data_dict['privileged_groups'])
            print("Val set: Difference in mean outcomes between unprivileged and privileged groups = {}".\
                  format(val_data_metric.mean_difference()))
            print("VALIDATION SET METRICS: ")
            print_all_metrics(val_class_metric)
            print("#"*30)

            # test metrics ################################################################
            print("#"*30)
            loss, test_acc, pred_y_test, scores_y_test = test(model, test_loader, loss_fn, epoch)
            print('Epoch: {}, Test Loss: {}, Test Accuracy: {}'.format(epoch, loss, test_acc))

            test_data_metric, test_class_metric = get_fairness_metrics(test_dataset,
                                                                       pred_y_test,
                                                                       scores_y_test,
                                                                       data_dict['unprivileged_groups'],
                                                                       data_dict['privileged_groups'])
            print("Test set: Difference in mean outcomes between unprivileged and privileged groups = {}".\
                  format(test_data_metric.mean_difference()))
            print("TEST SET METRICS: ")
            print_all_metrics(test_class_metric)
            print("#"*30)
        
        val_pred = val_dataset.copy(deepcopy=True)
        val_pred.labels = pred_y_val.cpu().numpy()
        val_pred.scores = scores_y_val.cpu().numpy()
        test_pred = test_dataset.copy(deepcopy=True)
        test_pred.labels = pred_y_test.cpu().numpy()
        test_pred.scores = scores_y_test.cpu().numpy()
        
        if self.postprocessing_method == "calibrated_eq_odds":
            debias_model = CalibratedEqOddsPostprocessing(privileged_groups=data_dict['privileged_groups'],
                                                 unprivileged_groups=data_dict['unprivileged_groups'],
                                                 cost_constraint='fnr',
                                                 seed=1234)
            debias_model = debias_model.fit(val_dataset, val_pred)
            dataset_debiasing_test = debias_model.predict(test_pred)
        
        elif self.postprocessing_method == "eq_odds":
            debias_model = EqOddsPostprocessing(privileged_groups=data_dict['privileged_groups'],
                                                unprivileged_groups=data_dict['unprivileged_groups'],
                                                seed=1234)
            debias_model = debias_model.fit(val_dataset, val_pred)
            dataset_debiasing_test = debias_model.predict(test_pred)

        elif self.postprocessing_method == "reject_option_classification":
            debias_model = RejectOptionClassification(privileged_groups=data_dict['privileged_groups'],
                                                      unprivileged_groups=data_dict['unprivileged_groups'],
                                                      low_class_thresh=0.01, high_class_thresh=0.99,
                                                      num_class_thresh=100, num_ROC_margin=50,
                                                      metric_name="Statistical parity difference",
                                                      metric_lb=-0.05, metric_ub=0.05)
            debias_model = debias_model.fit(val_dataset, val_pred)
            dataset_debiasing_test = debias_model.predict(test_pred)
        
        else:
            raise NotImplementedError

        
        classified_metric_debiasing_test = ClassificationMetric(test_dataset, 
                                                                dataset_debiasing_test,
                                                                unprivileged_groups=data_dict['unprivileged_groups'],
                                                                privileged_groups=data_dict['privileged_groups'])
        retacc = classified_metric_debiasing_test.accuracy()

        return retacc, retacc, retacc, classified_metric_debiasing_test, classified_metric_debiasing_test
