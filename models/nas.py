from ConfigSpace import (
    Configuration,
    ConfigurationSpace
)

from utils import get_fairness_obj

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
        objectives = {'rev_acc': 1 - val_acc}
        if self.fairness_metric is not None:
            objectives[self.fairness_obj_name] = val_fobj
        return objectives
