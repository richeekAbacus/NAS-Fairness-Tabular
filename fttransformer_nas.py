from __future__ import annotations

import rtdl
import torch
import torch.nn.functional as F

from utils import train, test, get_fairness_metrics, get_fairness_obj, print_all_metrics

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
    GreaterThanCondition,
    Constant,
)


class FTTransformerSearch:
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
        
        cs = ConfigurationSpace()

        #! add constraint that d_token should be divisible by n_head
        
        n_blocks = Integer("n_blocks", (1, 6), default=3)
        attention_n_heads = Constant("attention_n_heads", value=8)
        d_token_multiplier = Categorical("d_token_multiplier", [8, 12, 16, 24, 32, 40, 48, 56, 64], default=24)
        attention_dropout = Float("attention_dropout", (0.1, 0.5), default=0.2)
        ffn_dropout = Float("ffn_dropout", (0.0, 0.5), default=0.1)
        ffn_d_hidden_multiplier = Float("ffn_d_hidden_multiplier", (0.66, 2.66), default=1.33)
        
        if self.args.use_advanced_num_embeddings:
            embeddings = ["direct", "ple", "periodic"]
        else:
            embeddings = ["direct"]
            
        num_embedding_type = Categorical("num_embedding_type", embeddings, default="direct")
        
        if self.args.use_advanced_num_embeddings:
            num_embeddings_n_bins = Integer("num_embeddings_n_bins", (2, 6), default=2)
            sigma_periodic = Float("sigma_periodic", (0.0, 1.0), default=0.1)
        
        if self.args.use_mlp and self.args.use_intersample:
            layer_types = ["standard", "mlp", "intersample"]
        elif self.args.use_mlp:
            layer_types = ["standard", "mlp"]
        elif self.args.use_intersample:
            layer_types = ["standard", "intersample"]
        else:
            layer_types = ["standard"]
        
        layer1 = Categorical("layer1", layer_types, default="standard")
        layer2 = Categorical("layer2", layer_types, default="standard")
        layer3 = Categorical("layer3", layer_types, default="standard")
        layer4 = Categorical("layer4", layer_types, default="standard")
        layer5 = Categorical("layer5", layer_types, default="standard")
        layer6 = Categorical("layer6", layer_types, default="standard")
        
        if self.args.use_long_ffn:
            ffn_lengths = (1, 2)
            layer1_ffn_depth = Integer("layer1_ffn_depth", ffn_lengths, default=1)
            layer2_ffn_depth = Integer("layer2_ffn_depth", ffn_lengths, default=1)
            layer3_ffn_depth = Integer("layer3_ffn_depth", ffn_lengths, default=1)
            layer4_ffn_depth = Integer("layer4_ffn_depth", ffn_lengths, default=1)
            layer5_ffn_depth = Integer("layer5_ffn_depth", ffn_lengths, default=1)
            layer6_ffn_depth = Integer("layer6_ffn_depth", ffn_lengths, default=1)
        else:
            layer1_ffn_depth = Constant("layer1_ffn_depth", value=1)
            layer2_ffn_depth = Constant("layer2_ffn_depth", value=1)
            layer3_ffn_depth = Constant("layer3_ffn_depth", value=1)
            layer4_ffn_depth = Constant("layer4_ffn_depth", value=1)
            layer5_ffn_depth = Constant("layer5_ffn_depth", value=1)
            layer6_ffn_depth = Constant("layer6_ffn_depth", value=1)
            
        
        lr = Float("lr", (1e-5, 1e-3), default=1e-4, log=True)
        weight_decay = Float("weight_decay", (1e-6, 1e-3), default=1e-5, log=True)
        train_batch_size = Constant("train_batch_size", value=self.args.train_bs)
        test_batch_size = Constant("test_batch_size", value=self.args.test_bs)

        cs.add_hyperparameters([n_blocks, attention_n_heads, d_token_multiplier, attention_dropout,
                                ffn_dropout, ffn_d_hidden_multiplier, num_embedding_type,
                                layer1, layer2, layer3, layer4, layer5, layer6,
                                layer1_ffn_depth, layer2_ffn_depth, layer3_ffn_depth, layer4_ffn_depth,
                                layer5_ffn_depth, layer6_ffn_depth,
                                lr, weight_decay, train_batch_size, test_batch_size])
        
        if self.args.use_advanced_num_embeddings:
            cs.add_hyperparameters([num_embeddings_n_bins, sigma_periodic])
            use_num_embeddings_n_bins = EqualsCondition(child=num_embeddings_n_bins, parent=num_embedding_type, value="ple")
            use_sigma_periodic = EqualsCondition(child=sigma_periodic, parent=num_embedding_type, value="periodic")
            cs.add_conditions([use_num_embeddings_n_bins, use_sigma_periodic])
        
        use_layer2 = GreaterThanCondition(child=layer2, parent=n_blocks, value=1)
        use_layer3 = GreaterThanCondition(child=layer3, parent=n_blocks, value=2)
        use_layer4 = GreaterThanCondition(child=layer4, parent=n_blocks, value=3)
        use_layer5 = GreaterThanCondition(child=layer5, parent=n_blocks, value=4)
        use_layer6 = GreaterThanCondition(child=layer6, parent=n_blocks, value=5)
        
        cs.add_conditions([use_layer2, use_layer3, use_layer4, use_layer5, use_layer6])
        
        
        if self.args.use_intersample:
            use_conditions = ["standard", "intersample"]
        else:
            use_conditions = ["standard"]
        
        use_layer1_ffn_depth = InCondition(child=layer1_ffn_depth, parent=layer1, values=use_conditions)
        use_layer2_ffn_depth = InCondition(child=layer2_ffn_depth, parent=layer2, values=use_conditions)
        use_layer3_ffn_depth = InCondition(child=layer3_ffn_depth, parent=layer3, values=use_conditions)
        use_layer4_ffn_depth = InCondition(child=layer4_ffn_depth, parent=layer4, values=use_conditions)
        use_layer5_ffn_depth = InCondition(child=layer5_ffn_depth, parent=layer5, values=use_conditions)
        use_layer6_ffn_depth = InCondition(child=layer6_ffn_depth, parent=layer6, values=use_conditions)
        
        cs.add_conditions([use_layer1_ffn_depth, use_layer2_ffn_depth, use_layer3_ffn_depth,
                        use_layer4_ffn_depth, use_layer5_ffn_depth, use_layer6_ffn_depth])
        
        return cs

    def create_and_train_model_from_config(self, config: Configuration, budget: int) -> tuple[float, float, float]:
        print(config)
        print("Budget:", budget)
        print("#"*50)

        data_dict = self.data_fn(config["train_batch_size"], config["test_batch_size"],
                                 privilege_mode=self.args.privilege_mode)

        transformer_config = rtdl.FTTransformer.get_baseline_transformer_subconfig()
        transformer_config["n_blocks"] = config["n_blocks"]
        transformer_config["d_token"] = 8*config["d_token_multiplier"]
        transformer_config["attention_n_heads"] = config["attention_n_heads"]
        transformer_config["attention_dropout"] = config["attention_dropout"]
        transformer_config["ffn_d_hidden"] = int(8*config["d_token_multiplier"]*config["ffn_d_hidden_multiplier"])
        transformer_config["ffn_dropout"] = config["ffn_dropout"]
        transformer_config["residual_dropout"] = 0.0
        transformer_config["last_layer_query_idx"] = [-1]
        transformer_config["d_out"] = 1

        print(transformer_config)

        model = rtdl.FTTransformer._make(
            n_num_features=data_dict['train_loader'].dataset[0][0].shape[0],
            cat_cardinalities=None,
            transformer_config=transformer_config
        )
        model.to('cuda')

        optimizer = torch.optim.AdamW(
            model.optimization_param_groups(),
            lr=config["lr"],
            weight_decay=config["weight_decay"],
        )

        if data_dict['task_type'] == 'bin-class':
            loss_fn = F.binary_cross_entropy_with_logits
        else:
            raise NotImplementedError

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

        return train_acc, val_acc, test_acc, val_class_metric, test_class_metric


    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
        _, val_acc, test_acc, val_class_metric, test_class_metric = \
                self.create_and_train_model_from_config(config, budget) # fm = fairness metric
        val_fobj = get_fairness_obj(val_class_metric, self.fairness_metric)
        objectives = {'rev_acc': 1 - val_acc}
        if self.fairness_metric is not None:
            objectives[self.fairness_obj_name] = val_fobj
        return objectives
