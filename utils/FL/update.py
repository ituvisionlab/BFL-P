import torch
def update_global(args, networks, selected, freqs):
    if args.arch == "cnn" or args.arch == "fcnn":
        global_para = networks["global_model"].state_dict()
        for idx, net_id  in enumerate(selected):
            net_para = networks["nets"][net_id].cpu().state_dict()
            if idx == 0:
                for key in net_para:
                    global_para[key] = net_para[key] * freqs[idx]
            else:
                for key in net_para:
                    global_para[key] += net_para[key] * freqs[idx]
        networks["global_model"].load_state_dict(global_para)

    elif args.arch == "bcnn" or args.arch == "bfcnn":
        if args.update_method == "geometric":
            geometric_average_update(networks, selected, freqs)
        elif args.update_method == "arithmetic":
            arithmetic_average_update(networks, selected, freqs)
        elif args.update_method == "arithmetic_varscale":
            arithmetic_average_varscale_update(networks, selected, freqs)
        elif args.update_method == "merging":
            merging_update(networks, selected, freqs)
        elif args.update_method == "conflation":
            conflation_update(networks, selected, freqs)
        elif args.update_method == "population":
            population_update(networks, selected, freqs, args)
        else:
            raise ValueError("Wrong update method!")
    else:
        raise ValueError("Wrong arch!")


def geometric_average_update(networks, selected, freqs):
    global_para = networks["global_model"].state_dict()
    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                global_para[key] = net_para[key] * freqs[idx]
        else:
            for key in net_para:
                global_para[key] += net_para[key] * freqs[idx]
    networks["global_model"].load_state_dict(global_para)

def arithmetic_average_update(networks, selected, freqs):
    global_para = networks["global_model"].state_dict()
    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                if("sig" not in key):
                    global_para[key] = net_para[key] * freqs[idx]
        else:
            for key in net_para:
                if("sig" not in key):
                    global_para[key] += net_para[key] * freqs[idx]

    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                if("sig" in key):
                    global_para[key] = torch.log(net_para[key].clamp(-8, 8).exp() * freqs[idx])
        else:
            for key in net_para:
                if("sig" in key):
                    global_para[key] = torch.log(global_para[key].clamp(-8, 8).exp() + ((net_para[key].clamp(-8, 8).exp()) * freqs[idx]))
    networks["global_model"].load_state_dict(global_para)

def arithmetic_average_varscale_update(networks, selected, freqs):
    global_para = networks["global_model"].state_dict()
    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                if("sig" not in key):
                    global_para[key] = net_para[key] * freqs[idx]
        else:
            for key in net_para:
                if("sig" not in key):
                    global_para[key] += net_para[key] * freqs[idx]

    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                if("sig" in key):
                    global_para[key] = torch.log(net_para[key].clamp(-8, 8).exp() * freqs[idx]**2)
        else:
            for key in net_para:
                if("sig" in key):
                    global_para[key] = torch.log(global_para[key].clamp(-8, 8).exp() + ((net_para[key].clamp(-8, 8).exp()) * freqs[idx]**2))
    networks["global_model"].load_state_dict(global_para)

def merging_update(networks, selected, freqs):
    global_para = networks["global_model"].state_dict()
    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                if("sig" not in key):
                    global_para[key] = net_para[key] * freqs[idx]
        else:
            for key in net_para:
                if("sig" not in key):
                    global_para[key] += net_para[key] * freqs[idx]

    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                if("sig" in key):
                    layer = key.split(".")[0]
                    mu_global = global_para[f"{layer}.mu_w"]
                    mu_client = net_para[f"{layer}.mu_w"]
                    global_para[key] = torch.log(((mu_global-mu_client)**2 + net_para[key].clamp(-8, 8).exp()) * freqs[idx])
        else:
            for key in net_para:
                if("sig" in key):
                    layer = key.split(".")[0]
                    mu_global = global_para[f"{layer}.mu_w"]
                    mu_client = net_para[f"{layer}.mu_w"]
                    global_para[key] = torch.log(global_para[key].clamp(-8, 8).exp() + (((mu_global-mu_client)**2 + net_para[key].clamp(-8, 8).exp()) * freqs[idx]))
    networks["global_model"].load_state_dict(global_para)

def conflation_update(networks, selected, freqs):
    paras = {a: {b: {"mu":networks["nets"][b].state_dict()[f"{a}.mu_w"], "var":networks["nets"][b].state_dict()[f"{a}.logsig2_w"].clamp(-8,8).exp()} for b in selected} for a in ["fc1", "fc2", "fc3"] }
    global_para = networks["global_model"].state_dict()
    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                if("conv" in key):
                    global_para[key] = net_para[key] * freqs[idx]
        else:
            for key in net_para:
                if("conv" in key):
                    global_para[key] += net_para[key] * freqs[idx]

    for key in global_para:
        if("sig" in key):
            global_para[key] = torch.zeros_like(global_para[key])
            prod = torch.ones_like(global_para[key])
            for idx, net_id  in enumerate(selected):
                prod = prod*paras[key.split(".")[0]][net_id]["var"]
            alt = torch.zeros_like(global_para[key])
            for idx, net_id  in enumerate(selected):
                alt = alt+(prod/paras[key.split(".")[0]][net_id]["var"])*freqs[idx]
            global_para[key] = torch.log(max(freqs)*prod/alt)

    for key in global_para:
        if("mu_w" in key):
            global_para[key] = torch.zeros_like(global_para[key])
            prod = torch.ones_like(global_para[key])
            for idx, net_id  in enumerate(selected):
                prod = prod*paras[key.split(".")[0]][net_id]["var"]
            alt = torch.zeros_like(global_para[key])
            ust = torch.zeros_like(global_para[key])
            for idx, net_id  in enumerate(selected):
                alt = alt+(prod/paras[key.split(".")[0]][net_id]["var"])*freqs[idx]
                ust = ust+(paras[key.split(".")[0]][net_id]["mu"]*prod/paras[key.split(".")[0]][net_id]["var"])*freqs[idx]
            global_para[key] = ust/alt
    
    for key in global_para:
        if("bias" in key):
            global_para[key] = torch.zeros_like(global_para[key])
            for idx, net_id  in enumerate(selected):
                net_para = networks["nets"][net_id].cpu().state_dict()
                global_para[key] += net_para[key] * freqs[idx]

    networks["global_model"].load_state_dict(global_para)

def population_update(networks, selected, freqs, args):
    layers = ["fc1", "fc2", "fc3"]
    paras = {a: {b: {"mu":networks["nets"][b].state_dict()[f"{a}.mu_w"], "std":networks["nets"][b].state_dict()[f"{a}.logsig2_w"].clamp(-8,8).exp().sqrt()} for b in selected} for a in layers }
    global_para = networks["global_model"].state_dict()
    for idx, net_id  in enumerate(selected):
        net_para = networks["nets"][net_id].cpu().state_dict()
        if idx == 0:
            for key in net_para:
                if("conv" in key or "bias" in key):
                    global_para[key] = net_para[key] * freqs[idx]
        else:
            for key in net_para:
                if("conv" in key or "bias" in key):
                    global_para[key] += net_para[key] * freqs[idx]

    samples = {i:[] for i in layers}
    for idx, net_id  in enumerate(selected):
        for layer in layers:
            mu = paras[layer][net_id]["mu"]
            std = paras[layer][net_id]["std"]
            sample_size = int(freqs[idx] * args.population_size)
            samples[layer].append(torch.distributions.Normal(mu, std).sample(torch.Size([sample_size])))
    for layer in layers:
        layer_samples = torch.cat(samples[layer], dim=0)
        mu = layer_samples.mean(axis=0)
        std = layer_samples.std(axis=0)
        global_para[f"{layer}.mu_w"] = mu
        global_para[f"{layer}.logsig2_w"] = torch.log(std**2)
    networks["global_model"].load_state_dict(global_para)