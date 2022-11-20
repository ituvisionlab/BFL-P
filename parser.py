import argparse

experiment_dict = {
    "noniid-labeldir" :     {"beta":0.5, "sample":1, "n_parties":10, "comm_round": 50},
    "noniid-labeldir-500" : {"beta":0.5, "sample":0.1, "n_parties":100, "comm_round": 500},
    "iid" :                 {"sample":1, "n_parties":10, "comm_round": 50},
    "iid-500" :             {"sample":0.1, "n_parties":100, "comm_round": 500},
    "noniid-label1" :       {"label":1, "sample":1, "n_parties":10, "comm_round": 50},
    "noniid-label2" :       {"label":2, "sample":1, "n_parties":10, "comm_round": 50},
    "noniid-label3" :       {"label":3, "sample":1, "n_parties":10, "comm_round": 50},
    "noniid-label4" :       {"label":4, "sample":1, "n_parties":10, "comm_round": 50},
    "iid-diff-quantity" :   {"beta":0.5, "sample":1, "n_parties":10, "comm_round": 50},
    "iid-diff-quantity-500":{"beta":0.5, "sample":0.1, "n_parties":100, "comm_round": 500}
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10', choices=["cifar10", "fmnist", "kmnist", "cifar100", "svhn", "covertype"], help='dataset used for training')
    parser.add_argument('--experiment', type=str, default='noniid-labeldir', help='the data partitioning strategy')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--epochs', type=int, default=10, help='number of local epochs')
    parser.add_argument('--alg', type=str, default='FedAVG', choices=["BFL", "BFLAVG", "Fed", "FedAVG", "FedProx", "FedNova", "Scaffold"], help='communication strategy')
    parser.add_argument('--is_same_initial', type=int, default=1, help='whether initial all the models with the same parameters')
    parser.add_argument('--init_seed', type=int, default=0, help="Random seed")
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--datadir', type=str, required=False, default="./data/", help="Data directory")
    parser.add_argument('--reg', type=float, default=1e-5, help="L2 regularization strength")
    parser.add_argument('--logdir', type=str, required=False, default="./logs", help='Log directory path')
    parser.add_argument('--device', type=str, default='cuda:0', help='The device to run the program')
    parser.add_argument('--optimizer', type=str, default='sgd', help='the optimizer')
    parser.add_argument('--mu', type=float, default=0.01, help='the mu parameter for fedprox')
    parser.add_argument('--rho', type=float, default=0.9, help='Parameter controlling the momentum SGD')
    parser.add_argument('--update_method', type=str, default="none", choices=["geometric", "arithmetic", "merging", "none", "conflation", "arithmetic_varscale", "population"], help='aggregation method for bayesian neural networks')
    parser.add_argument('--process', type=int, default=5, help='number of parallel process')
    parser.add_argument('--population_size', type=float, default=1000, help='population size in population pooling based aggregation training')
    parser.add_argument('--desc', type=str, default="", help='Description of run')
    
    args = parser.parse_args()
    
    if args.alg == "FedAVG" or args.alg == "Fed":
        args.arch = "cnn"
    elif args.alg == "BFL" or args.alg == "BFL_AVG":
        args.arch = "bcnn"
    
    if args.dataset == "covertype":
        if args.alg == "FedAVG" or args.alg == "Fed":
            args.arch = "fcnn"
        elif args.alg == "BFL" or args.alg == "BFL_AVG":
            args.arch = "bfcnn"

    if args.experiment not in experiment_dict:
        raise ValueError("Experiment not found")

    for key, value in experiment_dict[args.partition].items():
        setattr(args, key, value)
    return args
