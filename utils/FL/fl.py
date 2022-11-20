from architectures import CNN, BCNN, FcNN, BFcNN

def init_nets(args):
    nets = {net_i: None for net_i in range(args.n_parties)}
    if args.dataset == 'cifar10':
        input_channel = 3
        input_dim = (16 * 5 * 5)
        hidden_dims=[120, 84]
        output_dim = 10
    elif args.dataset == 'fmnist':
        input_channel = 1
        input_dim = (16 * 4 * 4)
        hidden_dims=[120, 84]
        output_dim = 10
    elif args.dataset == 'kmnist':
        input_channel = 1
        input_dim = (16 * 4 * 4)
        hidden_dims=[120, 84]
        output_dim = 10
    elif args.dataset == 'cifar100':
        input_channel = 3
        input_dim = (16 * 5 * 5)
        hidden_dims=[120, 84]
        output_dim = 100
    elif args.dataset == 'covertype':
        input_dim = 54
        hidden_dims=[100, 50]
        output_dim = 7
    elif args.dataset == 'svhn':
        input_channel = 3
        input_dim = (16 * 5 * 5)
        hidden_dims=[120, 84]
        output_dim = 10

    for net_i in range(args.n_parties):        
        if args.arch.lower() == "cnn":
            net = CNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
        elif args.arch == "bcnn":
            net = BCNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
        elif args.arch == "fcnn":
            net = FcNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        elif args.arch == "bfcnn":
            net = BFcNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
        else:
            raise ValueError("Unknown architecture: {}".format(args.arch))
        nets[net_i] = net


    if args.arch.lower() == "cnn":
        global_net = CNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
    elif args.arch == "bcnn":
        global_net = BCNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim, input_channel=input_channel)
    elif args.arch == "fcnn":
        global_net = FcNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    elif args.arch == "bfcnn":
        global_net = BFcNN(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=output_dim)
    else:
        raise ValueError("Unknown architecture: {}".format(args.arch))

    if args.is_same_initial:
        global_para = global_net.state_dict() 
        for net_id, net in nets.items():
            net.load_state_dict(global_para)

    return global_net, nets
