import random
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import torch
import os
from PIL import Image

class CIFAR10_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if os.path.isfile(os.path.join(self.root, 'cifar-10/padded_train_data.pt')) == False:
            from utils.data.data_downloader import save_data_cifar10
            save_data_cifar10(self.root)
        if self.train:
            data = torch.load(os.path.join(self.root, 'cifar-10/padded_train_data.pt'))
            target = np.load(os.path.join(self.root, 'cifar-10/train_target.npy'))
        else:
            data = torch.load(os.path.join(self.root, 'cifar-10/test_data.pt'))
            target = np.load(os.path.join(self.root, 'cifar-10/test_target.npy'))
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class FashionMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if os.path.isfile(os.path.join(self.root, 'fmnist/train_data.pt')) == False:
            from utils.data.data_downloader import save_data_fmnist
            save_data_fmnist(self.root)
        if self.train:
            data = torch.load(os.path.join(self.root, 'fmnist/train_data.pt'))
            target = np.load(os.path.join(self.root, 'fmnist/train_target.npy'))
        else:
            data = torch.load(os.path.join(self.root, 'fmnist/test_data.pt'))
            target = np.load(os.path.join(self.root, 'fmnist/test_target.npy'))
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class KMNIST_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if os.path.isfile(os.path.join(self.root, 'kmnist/train_data.pt')) == False:
            from utils.data.data_downloader import save_data_kmnist
            save_data_kmnist(self.root)
        if self.train:
            data = torch.load(os.path.join(self.root, 'kmnist/train_data.pt'))
            target = np.load(os.path.join(self.root, 'kmnist/train_target.npy'))
        else:
            data = torch.load(os.path.join(self.root, 'kmnist/test_data.pt'))
            target = np.load(os.path.join(self.root, 'kmnist/test_target.npy'))
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class CIFAR100_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if os.path.isfile(os.path.join(self.root, 'cifar-100/padded_train_data.pt')) == False:
            from utils.data.data_downloader import save_data_cifar100
            save_data_cifar100(self.root)
        if self.train:
            data = torch.load(os.path.join(self.root, 'cifar-100/padded_train_data.pt'))
            target = np.load(os.path.join(self.root, 'cifar-100/train_target.npy'))
        else:
            data = torch.load(os.path.join(self.root, 'cifar-100/test_data.pt'))
            target = np.load(os.path.join(self.root, 'cifar-100/test_target.npy'))
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

class COVERTYPE_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if os.path.isfile(os.path.join(self.root, 'covertype/train_data.pt')) == False:
            from utils.data.data_downloader import save_data_covertype
            save_data_covertype(self.root)
        if self.train:
            data = np.load(os.path.join(self.root, 'covertype/train_data.npy'))
            target = np.load(os.path.join(self.root, 'covertype/train_target.npy'))
        else:
            data = np.load(os.path.join(self.root, 'covertype/test_data.npy'))
            target = np.load(os.path.join(self.root, 'covertype/test_target.npy'))
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        features, target = self.data[index], self.target[index]
        if self.transform is not None:
            features = self.transform(features)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return features.astype(np.float32), target

    def __len__(self):
        return len(self.data)

class SVHN_truncated(data.Dataset):
    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):
        if os.path.isfile(os.path.join(self.root, 'svhn/train_data.pt')) == False:
            from utils.data.data_downloader import save_data_svhn
            save_data_svhn(self.root)
        if self.train:
            data = torch.load(os.path.join(self.root, 'svhn/train_data.pt'))
            target = np.load(os.path.join(self.root, 'svhn/train_target.npy'))
        else:
            data = torch.load(os.path.join(self.root, 'svhn/test_data.pt'))
            target = np.load(os.path.join(self.root, 'svhn/test_target.npy'))
        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]
        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None):
    if dataset == 'cifar10':
        dl_obj = CIFAR10_truncated
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    elif dataset == 'fmnist':
        dl_obj = FashionMNIST_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    elif dataset == 'kmnist':
        dl_obj = KMNIST_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    elif dataset == 'cifar100':
        dl_obj = CIFAR100_truncated
        transform_train = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    elif dataset == 'svhn':
        dl_obj = SVHN_truncated
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    elif dataset == 'covertype':
        dl_obj = COVERTYPE_truncated
        transform_train = None
        transform_test = None

    train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
    test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

    if dataset == 'covertype':
        train_bs, test_bs = len(train_ds), 1024

    train_dl = data.DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False, num_workers=0)
    test_dl = data.DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False, num_workers=0)
    return train_dl, test_dl

def partition_data(args, logger):
    if args.dataset == 'cifar10':
        y_train = CIFAR10_truncated(args.datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])).target
        K = 10
    elif args.dataset == 'fmnist':
        y_train = FashionMNIST_truncated(args.datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])).target
        K = 10
    elif args.dataset == 'kmnist':
        y_train = KMNIST_truncated(args.datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])).target
        K = 10  
    elif args.dataset == 'cifar100':
        y_train = CIFAR100_truncated(args.datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])).target
        K = 100
    elif args.dataset == 'svhn':
        y_train = SVHN_truncated(args.datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])).target
        K = 10
    elif args.dataset == 'covertype':
        y_train = COVERTYPE_truncated(args.datadir, train=True, download=True, transform=transforms.Compose([transforms.ToTensor()])).target
        K = 7
    else:
        assert NotImplementedError(f"{args.dataset} is not available!")

    N = y_train.shape[0]

    if args.partition == "iid" or args.partition == "iid-500":
        idxs = np.random.permutation(N)
        batch_idxs = np.array_split(idxs, args.n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.n_parties)}

    elif args.partition == "noniid-labeldir" or args.partition == "noniid-labeldir-500":
        min_size = 0
        min_require_size = 10
        net_dataidx_map = {}
        
        while min_size < min_require_size:
            idx_batch = [[] for _ in range(args.n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(args.beta, args.n_parties))
                proportions = np.array([p * (len(idx_j) < N / args.n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(args.n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]
    
    elif args.partition > "noniid-label0" and args.partition <= "noniid-label9":
        num = int(args.partition[12:])

        times=[0 for i in range(K)]
        contain=[]
        for i in range(args.n_parties):
            current=[i%K]
            times[i%K]+=1
            j=1
            while (j<num):
                ind=random.randint(0,K-1)
                if (ind not in current):
                    j=j+1
                    current.append(ind)
                    times[ind]+=1
            contain.append(current)
        net_dataidx_map ={i:np.ndarray(0,dtype=np.int64) for i in range(args.n_parties)}
        
        for i in range(K):
            idx_k = np.where(y_train==i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k,times[i])
            ids=0
            for j in range(args.n_parties):
                if i in contain[j]:
                    net_dataidx_map[j]=np.append(net_dataidx_map[j],split[ids])
                    ids+=1

    elif args.partition in ["iid-diff-quantity", "iid-diff-quantity-500"]:
        idxs = np.random.permutation(N)
        min_size = 0
        while min_size < 10:
            proportions = np.random.dirichlet(np.repeat(args.beta, args.n_parties))
            proportions = proportions/proportions.sum()
            min_size = np.min(proportions*len(idxs))
        proportions = (np.cumsum(proportions)*len(idxs)).astype(int)[:-1]
        batch_idxs = np.split(idxs,proportions)
        net_dataidx_map = {i: batch_idxs[i] for i in range(args.n_parties)}
        
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    logger.info(f'Data statistics: {net_cls_counts}')
    return net_dataidx_map

def data_handler(args, logger):
    logger.info("Partitioning data")
    net_dataidx_map = partition_data(args, logger)
    _, test_dl_global = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)                                                   
    return test_dl_global, net_dataidx_map