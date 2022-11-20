import numpy as np
from torchvision.datasets import CIFAR10, FashionMNIST, CIFAR100, SVHN, KMNIST
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import os

def save_data_cifar10(datadir):
    os.makedirs(os.path.join(datadir, 'cifar-10'), exist_ok=True)
    transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4), mode='reflect').data.squeeze())
                ])
    cifar_train_dataobj = CIFAR10(datadir, True, None, None, True)
    train_data = cifar_train_dataobj.data
    train_target = np.array(cifar_train_dataobj.targets)
    padded_train_data = torch.zeros((train_data.shape[0], 3, 40, 40))
    for i, x in enumerate(train_data):
        padded_train_data[i] = transform_train(x)
    torch.save(padded_train_data, os.path.join(datadir, 'cifar-10/padded_train_data.pt'))
    np.save(os.path.join(datadir, 'cifar-10/train_target.npy'), train_target)

    cifar_test_dataobj = CIFAR10(datadir, False, None, None)
    test_data = cifar_test_dataobj.data
    test_target = np.array(cifar_test_dataobj.targets)
    torch.save(test_data, os.path.join(datadir, 'cifar-10/test_data.pt'))
    np.save(os.path.join(datadir, 'cifar-10/test_target.npy'), test_target)

def save_data_fmnist(datadir):
    os.makedirs(os.path.join(datadir, 'fmnist'), exist_ok=True)
    fmnist_train_dataobj = FashionMNIST(datadir, True, None, None, True)
    train_data = fmnist_train_dataobj.data
    train_target = np.array(fmnist_train_dataobj.targets)
    torch.save(train_data, os.path.join(datadir, 'fmnist/train_data.pt'))
    np.save(os.path.join(datadir, 'fmnist/train_target.npy'), train_target)

    fmnist_test_dataobj = FashionMNIST(datadir, False, None, None)
    test_data = fmnist_test_dataobj.data
    test_target = np.array(fmnist_test_dataobj.targets)
    torch.save(test_data, os.path.join(datadir, 'fmnist/test_data.pt'))
    np.save(os.path.join(datadir, 'fmnist/test_target.npy'), test_target)

def save_data_cifar100(datadir):
    os.makedirs(os.path.join(datadir, 'cifar-100'), exist_ok=True)
    transform_train = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(lambda x: F.pad(
                        Variable(x.unsqueeze(0), requires_grad=False),
                        (4, 4, 4, 4), mode='reflect').data.squeeze())
                ])
    cifar_train_dataobj = CIFAR100(datadir, True, None, None, True)
    train_data = cifar_train_dataobj.data
    train_target = np.array(cifar_train_dataobj.targets)
    padded_train_data = torch.zeros((train_data.shape[0], 3, 40, 40))
    for i, x in enumerate(train_data):
        padded_train_data[i] = transform_train(x)
    torch.save(padded_train_data, os.path.join(datadir, 'cifar-100/padded_train_data.pt'))
    np.save(os.path.join(datadir, 'cifar-100/train_target.npy'), train_target)

    cifar_test_dataobj = CIFAR100(datadir, False, None, None)
    test_data = cifar_test_dataobj.data
    test_target = np.array(cifar_test_dataobj.targets)
    torch.save(test_data, os.path.join(datadir, 'cifar-100/test_data.pt'))
    np.save(os.path.join(datadir, 'cifar-100/test_target.npy'), test_target)

def save_data_covertype(datadir):
    from sklearn.datasets import fetch_covtype
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    os.makedirs(os.path.join(datadir, 'covertype'), exist_ok=True)
    data, target = fetch_covtype(data_home=datadir, download_if_missing=True, return_X_y=True)
    data = preprocessing.MinMaxScaler().fit_transform(data)
    target = target - 1
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3, random_state=0)
    np.save(os.path.join(datadir, 'covertype/train_data.npy'), train_data)
    np.save(os.path.join(datadir, 'covertype/train_target.npy'), train_target)

    np.save(os.path.join(datadir, 'covertype/test_data.npy'), test_data)
    np.save(os.path.join(datadir, 'covertype/test_target.npy'), test_target)

def save_data_svhn(datadir):
    os.makedirs(os.path.join(datadir, 'svhn'), exist_ok=True)
    svhn_train_dataobj = SVHN(datadir, 'train', None, None, True)
    train_data = svhn_train_dataobj.data
    train_target = np.array(svhn_train_dataobj.labels)
    torch.save(train_data, os.path.join(datadir, 'svhn/train_data.pt'))
    np.save(os.path.join(datadir, 'svhn/train_target.npy'), train_target)

    svhn_test_dataobj = SVHN(datadir, 'test', None, None, True)
    test_data = svhn_test_dataobj.data
    test_target = np.array(svhn_test_dataobj.labels)
    torch.save(test_data, os.path.join(datadir, 'svhn/test_data.pt'))
    np.save(os.path.join(datadir, 'svhn/test_target.npy'), test_target)

def save_data_kmnist(datadir):
    os.makedirs(os.path.join(datadir, 'kmnist'), exist_ok=True)
    kmnist_train_dataobj = KMNIST(datadir, True, None, None, True)
    train_data = kmnist_train_dataobj.data
    train_target = np.array(kmnist_train_dataobj.targets)
    torch.save(train_data, os.path.join(datadir, 'kmnist/train_data.pt'))
    np.save(os.path.join(datadir, 'kmnist/train_target.npy'), train_target)

    kmnist_test_dataobj = KMNIST(datadir, False, None, None)
    test_data = kmnist_test_dataobj.data
    test_target = np.array(kmnist_test_dataobj.targets)
    torch.save(test_data, os.path.join(datadir, 'kmnist/test_data.pt'))
    np.save(os.path.join(datadir, 'kmnist/test_target.npy'), test_target)

if __name__ == "__main__":
    save_data_cifar10("./data")
    save_data_fmnist("./data")
    save_data_cifar100("./data")
    save_data_covertype("./data")
    save_data_svhn("./data")
    save_data_kmnist("./data")
    pass