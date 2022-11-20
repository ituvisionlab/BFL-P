import torch
import os
import datetime 
import logging
import numpy as np
import time
import copy
import csv
import random
import torch.multiprocessing as mp
torch.multiprocessing.set_sharing_strategy('file_system')

from utils.data.data import get_dataloader, data_handler
from utils.FL.fl import init_nets
from parser import get_args
from algorithms import get_algorithm
from scores import compute_scores
# import wandb

def local_train(args, algo, networks, selected, net_dataidx_map):
    start = time.time()
    processes = []
    finished_process = 0

    for net_id, net in networks["nets"].items():
        if net_id not in selected:
            continue

        dataidxs = net_dataidx_map[net_id]
        net.dataset_size = len(dataidxs)
        process = mp.Process(target=algo.local_update, args=(args, networks["nets"][net_id], networks["global_model"], net_id, dataidxs))
        process.start()
        processes.append(process)

        if len(processes) == args.process:
            for p in processes:
                p.join()
                finished_process += 1
            processes = []
    for p in processes:
        p.join()
        finished_process += 1
    processes = []
    logger.info(f"{(time.time()-start):.2f} second.")


def main(args, logger, pre_selected):
    algo = get_algorithm(args)()
    test_dl_global, net_dataidx_map = data_handler(args, logger)
    global_model, nets = init_nets(args)
    networks = {"global_model": global_model, "nets": nets}

    results = []
    args.round = 0

    for args.round in range(args.comm_round):        
        logger.info(f"Communication round: {args.round} / {args.comm_round}")
        # selection of clients
        if args.partition == "noniid-labeldir-500" or args.partition == "iid-500":
            selected = pre_selected[args.round]
        else:
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            selected = arr[:int(args.n_parties * args.sample)]

        global_para = networks["global_model"].state_dict()
        if args.round == 0:
            if args.is_same_initial:
                for net_id in selected:
                    networks["nets"][net_id].load_state_dict(global_para)
        else:   
            for net_id in selected:
                networks["nets"][net_id].load_state_dict(global_para)

        # party executes
        local_train(args, algo, networks, selected, net_dataidx_map)

        # updating parameters of trained networks via reading files due to process operations
        for net_id in networks["nets"].keys():
            if net_id not in selected or (args.alg.lower() == "fednova"):
                continue
            networks["nets"][net_id].load_state_dict(torch.load(f"{args.logdir}/clients/client_{net_id}.pt"))
                
        # global update rule
        algo.global_update(args, networks, selected, net_dataidx_map)

        # if args.prior_update:
        #     for idx in range(len(selected)):
        #         networks["nets"][selected[idx]].update_priors(networks["global_model"])
        
        test_acc, test_ece, test_nll = compute_scores(networks["global_model"], test_dl_global, args, device=args.device, n_sample=[1,10]["bfl" in args.alg.lower()])
        networks["global_model"].cpu()
        logger.critical(f'>> Global Model Test accuracy: {test_acc}, ECE: {test_ece}, NLL: {test_nll}')
        
        results.append((args.round + 1, test_acc, test_ece, test_nll))

    networks["global_model"].to("cpu")
    torch.save(networks["global_model"].state_dict(), f"{args.logdir}/checkpoint.pt")

    with open(f"{args.logdir}/log.csv", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(("Round", "Acc", "ECE", "NLL"))
        writer.writerows(results)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.set_num_threads(1)
    args = get_args()

    log_time = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M-%S")
    args.logdir = os.path.join(f"{args.logdir}/{args.dataset}/{args.partition}/{args.alg}/{args.update_method}/{args.init_seed}/{log_time}")
    os.makedirs(f"{args.logdir}/clients", exist_ok=True)
    
    args.device = torch.device(args.device)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=f'{args.logdir}/log.log',
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%m-%d %H:%M-%S', level=logging.DEBUG, filemode='w')
    
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(args)
    logger.info(f"device: {args.device}")

    seed = args.init_seed
    logger.info("#" * 100)

    if args.desc != "":
        logger.critical(f"Description: {args.desc}")
    logger.critical(f"Update Method: {args.update_method}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    pre_selected = {}
    if args.partition == "noniid-labeldir-500" or args.partition == "iid-500":
        for i in range(args.comm_round):
            arr = np.arange(args.n_parties)
            np.random.shuffle(arr)
            pre_selected[i] = arr[:int(args.n_parties * args.sample)]
        with open(f"{args.logdir}/pre_selected.txt", 'w') as f:
            for i in range(args.comm_round):
                f.write(f"{i}: {pre_selected[i]}\n")
    main(args, logger, pre_selected)