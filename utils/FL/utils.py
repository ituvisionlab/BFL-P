from utils.data.data import get_dataloader
import torch

def train_handler(args, net, net_id, dataidxs, reduction = "mean"):
    train_dataloader, _ = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg)
    elif args.optimizer == 'amsgrad':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, weight_decay=args.reg, amsgrad=True)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=args.lr, momentum=args.rho, weight_decay=args.reg)
    criterion = torch.nn.CrossEntropyLoss(reduction=reduction).to(args.device)
    return train_dataloader, optimizer, criterion