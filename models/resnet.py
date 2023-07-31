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


class ResNetSearch(NAS):
    def __init__(self, args, data_fn, fairness_metric=None) -> None:
        super(ResNetSearch, self).__init__(args, data_fn, fairness_metric)

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

        print(config)

        model = rtdl.ResNet.make_baseline(
            d_in=data_dict['train_loader'].dataset[0][0].shape[0],
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

        train_acc, val_acc, test_acc, val_class_metric, test_class_metric = \
            budget_trainer(self.args, model, optimizer, data_dict, loss_fn, budget)

        return train_acc, val_acc, test_acc, val_class_metric, test_class_metric
