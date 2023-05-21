import random
import numpy as np
import torch
import wandb

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
    GreaterThanCondition,
    LessThanCondition,
    Constant,
)

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband

from nas_utils import parse_args, get_dataset_openml, get_dataset_rtdl, create_and_train_model_from_config, plot_trajectory

args = parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

if args.use_openml:
    data = get_dataset_openml(data_id=args.data_id, task_type=args.task_type, seed=args.seed)
else:
    data = get_dataset_rtdl(name=args.dataset, seed=args.seed)

class FTTransformerSearch:
    @property
    def configspace(self) -> ConfigurationSpace:
        
        cs = ConfigurationSpace()
        
        n_blocks = Integer("n_blocks", (1, 6), default=3)
        attention_n_heads = Integer("attention_n_heads", (6, 12), default=8)
        d_token_multiplier = Categorical("d_token_multiplier", [8, 12, 16, 24, 32, 40, 48, 56, 64], default=24)
        attention_dropout = Float("attention_dropout", (0.1, 0.5), default=0.2)
        ffn_dropout = Float("ffn_dropout", (0.0, 0.5), default=0.1)
        ffn_d_hidden_multiplier = Float("ffn_d_hidden_multiplier", (0.66, 2.66), default=1.33)
        
        if args.use_advanced_num_embeddings:
            embeddings = ["direct", "ple", "periodic"]
        else:
            embeddings = ["direct"]
            
        num_embedding_type = Categorical("num_embedding_type", embeddings, default="direct")
        
        if args.use_advanced_num_embeddings:
            num_embeddings_n_bins = Integer("num_embeddings_n_bins", (2, 6), default=2)
            sigma_periodic = Float("sigma_periodic", (0.0, 1.0), default=0.1)
        
        if args.use_mlp and args.use_intersample:
            layer_types = ["standard", "mlp", "intersample"]
        elif args.use_mlp:
            layer_types = ["standard", "mlp"]
        elif args.use_intersample:
            layer_types = ["standard", "intersample"]
        else:
            layer_types = ["standard"]
        
        layer1 = Categorical("layer1", layer_types, default="standard")
        layer2 = Categorical("layer2", layer_types, default="standard")
        layer3 = Categorical("layer3", layer_types, default="standard")
        layer4 = Categorical("layer4", layer_types, default="standard")
        layer5 = Categorical("layer5", layer_types, default="standard")
        layer6 = Categorical("layer6", layer_types, default="standard")
        
        if args.use_long_ffn:
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
        train_batch_size = Constant("train_batch_size", value=args.train_batch_size)
        test_batch_size = Constant("test_batch_size", value=args.test_batch_size)

        cs.add_hyperparameters([n_blocks, attention_n_heads, d_token_multiplier, attention_dropout,
                                ffn_dropout, ffn_d_hidden_multiplier, num_embedding_type,
                                layer1, layer2, layer3, layer4, layer5, layer6,
                                layer1_ffn_depth, layer2_ffn_depth, layer3_ffn_depth, layer4_ffn_depth,
                                layer5_ffn_depth, layer6_ffn_depth,
                                lr, weight_decay, train_batch_size, test_batch_size])
        
        if args.use_advanced_num_embeddings:
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
        
        
        if args.use_intersample:
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

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
        
        train_score, val_score, test_score = create_and_train_model_from_config(config, data, budget)
        
        if args.use_wandb:
            wandb.log({"train score": train_score, "val score": val_score, "test score": test_score})
        
        if data['task_type'] == 'regression':
            return val_score

        return 1 - val_score


if __name__ == "__main__":
    
    if args.use_wandb:
        wandb.init(project="Tab Nas", name=args.name)
    
    ftts = FTTransformerSearch()

    facades: list[AbstractFacade] = []
    
    print('Running NAS on %d GPUs'%torch.cuda.device_count())
    
    scenario = Scenario(
        ftts.configspace,
        name=args.name,
        output_directory=args.output_directory,
        walltime_limit=args.wall_time_limit,  # After n seconds, we stop the hyperparameter optimization
        n_trials=args.n_trails,  # Evaluate max n different trials
        min_budget=args.min_budget,  # Train the model using a architecture configuration for at least n epochs
        max_budget=args.max_budget,  # Train the model using a architecture configuration for at most n epochs
        seed=args.seed,
        n_workers=torch.cuda.device_count(),
    )

    # We want to run n random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=args.initial_n_configs)

    intensifier = Hyperband(scenario, incumbent_selection="highest_budget") # SuccessiveHalving can be used

    smac = MFFacade(
        scenario,
        ftts.train,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
    )

    incumbent = smac.optimize()

    default_config = ftts.configspace.get_default_configuration()
    _, default_val_score, default_test_score = create_and_train_model_from_config(config=default_config, data=data, budget=args.eval_budget)
    print(f"\nDefault val score: {default_val_score}")
    print(f"Default test score: {default_test_score}")
    
    _, incumbent_val_score, incumbent_test_score = create_and_train_model_from_config(config=incumbent, data=data, budget=args.eval_budget)
    print(f"\nIncumbent val score: {incumbent_val_score}")
    print(f"Incumbent test score: {incumbent_test_score}")
    
    print("\nSelected Model")
    print(incumbent)

    facades.append(smac)

    plot_trajectory(facades, args.output_directory, args.seed, args.name)
    
    if args.use_wandb:
        wandb.finish()
