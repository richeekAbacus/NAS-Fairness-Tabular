import os
import json
import torch

from ConfigSpace import (
    Configuration,
    ConfigurationSpace
)

from utils import train, test, get_fairness_metrics, print_all_metrics, get_fairness_obj, log_fairness_metrics

class NAS:
    def __init__(self, args, data_fn, fairness_metric=None) -> None:
        self.args = args
        self.data_fn = data_fn
        self.fairness_metric = fairness_metric
        self.fairness_obj_name = self.fairness_metric + "_obj"

    @property
    def get_objectives(self)-> list[str]:
        if self.fairness_metric is not None:
            return ["rev_acc", self.fairness_obj_name] # both need to be minimized
        else:
            return ["rev_acc"]

    @property
    def configspace(self) -> ConfigurationSpace:
        raise NotImplementedError

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
        _, val_acc, _, val_class_metric, _ = \
                self.create_and_train_model_from_config(config, budget) # fm = fairness metric
        val_fobj = get_fairness_obj(val_class_metric, self.fairness_metric)
        # objectives = {'rev_acc': 1 - val_acc} # replace with balanced classification accuracy and check
        objectives = {'rev_acc': get_fairness_obj(val_class_metric, 'balanced_class_acc')}
        if self.fairness_metric is not None:
            objectives[self.fairness_obj_name] = val_fobj

        with open(os.path.join(self.args.output_dir, self.args.run_name, str(self.args.seed), 'stats.csv'), 'a') as f:
            f.write(','.join(str(x) for x in log_fairness_metrics(val_class_metric)) + '\n')
        
        with open(os.path.join(self.args.output_dir, self.args.run_name, str(self.args.seed), 'stats.json'), 'r') as f:
            existing = json.load(f)
        existing["data"].append(log_fairness_metrics(val_class_metric, asdict=True))
        with open(os.path.join(self.args.output_dir, self.args.run_name, str(self.args.seed), 'stats.json'), 'w') as f:
            json.dump(existing, f)

        return objectives


# external function to train a model for budget.
def budget_trainer(args, model, optimizer, data_dict, loss_fn, budget):
    best_val_metric = 2.0 # metric = (1-val_acc)^2 + (1-val_fairness)^2. Worst case is 2.0
    for epoch in range(int(budget)):
        train(model, optimizer, data_dict['train_loader'], loss_fn, epoch, 100)

        loss, train_acc, _, _ = test(model, data_dict['train_loader'], loss_fn, epoch)
        print('Epoch: {}, Train Loss: {}, Train Accuracy: {}'.format(epoch, loss, train_acc))

        # validation metrics ##########################################################
        print("#"*30)
        loss, val_acc, pred_y_val, scores_y_val = test(model, data_dict['val_loader'], loss_fn, epoch)
        print('Epoch: {}, Val Loss: {}, Val Accuracy: {}'.format(epoch, loss, val_acc))

        val_data_metric, val_class_metric = get_fairness_metrics(data_dict['val_dataset'], 
                                                                    pred_y_val,
                                                                    scores_y_val,
                                                                    data_dict['unprivileged_groups'],
                                                                    data_dict['privileged_groups'])
        print("Val set: Difference in mean outcomes between unprivileged and privileged groups = {}".\
                format(val_data_metric.mean_difference()))
        print("VALIDATION SET METRICS: ")
        print_all_metrics(val_class_metric)
        print("#"*30)

        val_metric = get_fairness_obj(val_class_metric, args.fairness_metric)**2 + get_fairness_obj(val_class_metric, "balanced_class_acc")**2
        if val_metric <= best_val_metric:
            print("Found best model in budge optimization. Saving to...", "checkpoints/" + args.run_name + "_best_model.pt")
            best_val_metric = val_metric
            torch.save(model.state_dict(), "checkpoints/" + args.run_name + "_best_model.pt")

        # test metrics ################################################################
        print("#"*30)
        loss, test_acc, pred_y_test, scores_y_test = test(model, data_dict['test_loader'], loss_fn, epoch)
        print('Epoch: {}, Test Loss: {}, Test Accuracy: {}'.format(epoch, loss, test_acc))

        test_data_metric, test_class_metric = get_fairness_metrics(data_dict['test_dataset'],
                                                                    pred_y_test,
                                                                    scores_y_test,
                                                                    data_dict['unprivileged_groups'],
                                                                    data_dict['privileged_groups'])
        print("Test set: Difference in mean outcomes between unprivileged and privileged groups = {}".\
                format(test_data_metric.mean_difference()))
        print("TEST SET METRICS: ")
        print_all_metrics(test_class_metric)
        print("#"*30)

    # Load best model and get val and test metrics
    model.load_state_dict(torch.load("checkpoints/" + args.run_name + "_best_model.pt"))
    loss, val_acc, pred_y_val, scores_y_val = test(model, data_dict['val_loader'], loss_fn, epoch)
    val_data_metric, val_class_metric = get_fairness_metrics(data_dict['val_dataset'], 
                                                                pred_y_val,
                                                                scores_y_val,
                                                                data_dict['unprivileged_groups'],
                                                                data_dict['privileged_groups'])
    loss, test_acc, pred_y_test, scores_y_test = test(model, data_dict['test_loader'], loss_fn, epoch)
    test_data_metric, test_class_metric = get_fairness_metrics(data_dict['test_dataset'],
                                                                pred_y_test,
                                                                scores_y_test,
                                                                data_dict['unprivileged_groups'],
                                                                data_dict['privileged_groups'])

    return train_acc, val_acc, test_acc, val_class_metric, test_class_metric
