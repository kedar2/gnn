from torch_geometric.datasets import Planetoid
from examples.planetoid.experiment import PlanetoidExperiment
from attrdict import AttrDict
import numpy as np
import pandas as pd

cora = Planetoid(root="data", name="cora")
citeseer = Planetoid(root="data", name="citeseer")
pubmed = Planetoid(root="data", name="pubmed")

dataset_lookup = {"cora": cora, "citeseer": citeseer, "pubmed": pubmed}

default_args = AttrDict({
    "display": False,
    "learning_rate": 1e-3,
    "max_epochs": 10000,
    "batch_size": 64,
    "eval_every": 1,
    "num_trials": 5,
    "dropout": 0.5,
    "weight_decay": 1e-5,
    "num_hidden_layers": 1,
    "hidden_dim": 128,
    "learning_rate": 1e-3,
    "layer_type": "GCN",
    "rewiring": "fosr",
    "num_iterations": 50,
    "num_relations": 2,
    "patience": 20,
    "dataset": None,
    "last_layer_fa": False,
    "lower_loss_is_better": False # For classification, "loss" here refers to validation accuracy.
    })

def run(args):
    args = default_args + args
    results = []
    # Option to run only a single dataset in the benchmark.
    if args.dataset is None:
        datasets = dataset_lookup
    else:
        datasets = {args.dataset: dataset_lookup[args.dataset]}

    for key in datasets:
        dataset = datasets[key]
        train_accuracies = []
        validation_accuracies = []
        test_accuracies = []
        for _ in range(args.num_trials):
            experiment = PlanetoidExperiment(args=args, dataset=dataset)
            train_acc, validation_acc, test_acc = experiment.run()
            train_accuracies.append(train_acc)
            validation_accuracies.append(validation_acc)
            test_accuracies.append(test_acc)
        train_mean = 100 * np.mean(train_accuracies)
        validation_mean = 100 * np.mean(validation_accuracies)
        test_mean = 100 * np.mean(test_accuracies)
        train_ci = 100 * np.std(train_accuracies) / args.num_trials
        validation_ci = 100 * np.std(validation_accuracies) / args.num_trials
        test_ci = 100 * np.std(test_accuracies) / args.num_trials
        
        print(f"Results for {key} with {args.rewiring} rewiring:")
        print(f"Average accuracy: {test_mean} plus/minus {test_ci}")
        results.append({
            "dataset": key,
            "rewiring": args.rewiring,
            "layer_type": args.layer_type,
            "test_mean": test_mean,
            "test_ci": test_ci,
            "validation_mean": validation_mean,
            "validation_ci": validation_ci,
            "train_mean": train_mean,
            "train_ci": train_ci,
            "last_layer_fa": args.last_layer_fa}
        )

    df = pd.DataFrame(results)
    with open('results/planetoid.csv', 'a') as f:
        df.to_csv(f, mode='a', header=f.tell()==0)
