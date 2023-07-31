import rtdl
import torch
import torch.nn.functional as F

from .nas import NAS, budget_trainer

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
)


class MLPSearch(NAS):
    def __init__(self, args, data_fn, fairness_metric=None) -> None:
        super(MLPSearch, self).__init__(args, data_fn, fairness_metric)

    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace()

        n_layers = Integer("n_layers", (4, 10), default=5)
        layerfirst = Categorical("layerfirst", [32, 64, 128, 256, 512], default=32)
        layermid = Categorical("layermid", [32, 64, 128, 256, 512], default=256)
        layerlast = Categorical("layerlast", [32, 64, 128, 256, 512], default=32)
        dropout = Float("dropout", (0.0, 0.5), default=0.25)
        
        lr = Float("lr", (1e-5, 1e-3), default=1e-4, log=True)
        weight_decay = Float("weight_decay", (1e-6, 1e-3), default=1e-5, log=True)
        train_batch_size = Constant("train_batch_size", value=self.args.train_bs)
        test_batch_size = Constant("test_batch_size", value=self.args.test_bs)

        cs.add_hyperparameters([n_layers, layerfirst, layermid, layerlast, dropout,
                                lr, weight_decay, train_batch_size, test_batch_size])
        
        return cs

    def create_and_train_model_from_config(self, config: Configuration, budget: int) -> tuple[float, float, float]:
        print(config)
        print("Budget:", budget)
        print("#"*50)

        data_dict = self.data_fn(config["train_batch_size"], config["test_batch_size"],
                                 privilege_mode=self.args.privilege_mode)

        print(config)
        
        d_layers = [config["layerfirst"]] + [config["layermid"]]*(config["n_layers"]-2) + [config["layerlast"]]

        model = rtdl.MLP.make_baseline(
            d_in=data_dict['train_loader'].dataset[0][0].shape[0],
            d_layers=d_layers,
            dropout=config["dropout"],
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

        train_acc, val_acc, test_acc, val_class_metric, test_class_metric = \
            budget_trainer(self.args, model, optimizer, data_dict, loss_fn, budget)

        return train_acc, val_acc, test_acc, val_class_metric, test_class_metric
