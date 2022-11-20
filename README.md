# A Parallel Federated Learning Framework
This repository is created to experiment easily in parallel for the state of the art methods in Federated Learning. It also includes the implementation of [How to Combine Variational Bayesian Networks in Federated Learning](https://arxiv.org/abs/2206.10897) in Pytorch. Please cite the paper if you benefit from this framework. For further details, contact buldu19@itu.edu.tr or ozera17@itu.edu.tr.

# Requirements and Usage
## Requirements 
 You can install the requirements for this project by using requirements.txt

 ```sh
 $ conda install --file requirements.txt
```
## Data
### Data preperation
 The dataset is automatically downloaded and prepared by the code when first time running the experiment. For precaching the dataset, you can run the following command:

| $\textbf{Datasets}$ |     $\textbf{Image Size}$ | $\textbf{Number of Labels}$ | $\textbf{Train Size}$ | $\textbf{Test Size}$ |
|-------------------|------------------------:|:-----------------:|:-------------------:|:------------------:|
| FMNIST            | $1 \times 28 \times 28$ |        $10$       |       $60000$       |       $10000$      |
| Cifar-10          | $3 \times 32 \times 32$ |        $10$       |       $50000$       |       $10000$      |
| SVHN              | $3 \times 32 \times 32$ |        $10$       |       $73257$       |       $26032$      |

```sh
 $ python utils/data/data_downloader.py
```

### Non-IID Data Generation
We inherited the non-IID data generation methods from [Federated Learning on Non-IID Data Silos: An Experimental Study](https://arxiv.org/pdf/2102.02079). You can run the following experiments:

| Experiment | Description |
|------------|-------------|
| IID        | IID data generation for 10 clients |
| IID-500    | IID data generation for 100 clients |
| noniid-labeldir    | Non-IID data generation for 10 clients with dirichlet distribution |
| noniid-labeldir-500    | Non-IID data generation for 100 clients with dirichlet distribution |
| noniid-label[1:4]    | Non-IID data generation for 10 clients with selection of how many class each client have (choices: 1, 2, 3, 4) |
| iid-diff-quantity    | IID data generation for 10 clients with different quantity of data |
| iid-diff-quantity-500    | IID data generation for 100 clients with different quantity of data |

## Usage
 You can run the experiments by using the following command:

```sh
 python train.py \
    --dataset=cifar10 \
    --alg=BFLAVG \
    --experiment=noniid-labeldir \
    --device='cuda:0'\
    --process=5 \
    --datadir='./data/' \
    --logdir='./logs/' \
    --init_seed=0
```

| Parameter | Description |
| ------ | ------ |
| dataset | Dataset name: cifar10, fmnist, kmnist, cifar100, svhn, covertype|
| alg | Algorithm name: BFL, BFLAVG, Fed, FedAVG, FedProx, FedNova, Scaffold|
| experiment | Experiment name: noniid-labeldir[-500], iid[-500], noniid-label[1:4], iid-dif-quantity[-500]|
| device | Device name: cuda:0, cpu|
| process | Number of processes for multiprocessing |
| datadir | Data directory path |
| logdir | Log directory path |
| init_seed | Initial seed number for the experiment |
| desc | Description of the experiment |

# Example runtime comparison of multi-process pipeline
Runtime comparison results based on the number of processes of IID partitioned 100 clients experiment with means $\pm$ standard errors of Time Per Communication round (TPC) across five communication rounds for CIFAR-10 dataset.  Multi-processed pipeline with 10 processes is the fastest for all models.

$\textbf{\Large Time per Communication Round}$
| $\textbf{Model}$ | $\textbf{Agg.}$ |      $\textbf{1 process}$      |  $\textbf{5 processes}$  |  $\textbf{10 processes}$ |
|----------------|:-------------:|:----------------------------:|:----------------------:|:----------------------:|
| FED            |  $\texttt{N/A}$ |    $60.83$ <small>$\pm 0.26$<small>    | $15.55$ <small>$\pm 0.51$<small> |  $9.30$ <small>$\pm 0.07$<small> |
| FEDAVG         |  $\texttt{N/A}$ |    $60.90$ <small>$\pm 0.18$<small>    | $15.57$ <small>$\pm 0.39$<small> |  $9.22$ <small>$\pm 0.20$<small> |
| FVBA           |  $\texttt{EAA}$ |    $72.77$ <small>$\pm 0.18$<small>    | $16.22$ <small>$\pm 0.05$<small> |  $9.49$ <small>$\pm 0.06$<small> |
|                |  $\texttt{GAA}$ |    $71.23$ <small>$\pm 0.88$<small>    | $16.48$ <small>$\pm 0.10$<small> |  $9.41$ <small>$\pm 0.05$<small> |
|                | $\texttt{AALV}$ |    $72.10$ <small>$\pm 0.36$<small>    | $16.33$ <small>$\pm 0.10$<small> |  $9.51$ <small>$\pm 0.09$<small> |
|                |  $\texttt{PPA}$ |    $66.95$ <small>$\pm 0.31$<small>    | $18.06$ <small>$\pm 0.20$<small> | $11.23$ <small>$\pm 0.16$<small> |
|                |  $\texttt{CF}$  |    $72.53$ <small>$\pm 0.29$<small>    | $16.34$ <small>$\pm 0.10$<small> |  $9.36$ <small>$\pm 0.14$<small> |
| FVBWA          |  $\texttt{EAA}$ |    $72.38$ <small>$\pm 0.31$<small>    | $16.45$ <small>$\pm 0.06$<small> |  $9.44$ <small>$\pm 0.11$<small> |
|                |  $\texttt{GAA}$ |    $72.78$ <small>$\pm 0.15$<small>    | $15.88$ <small>$\pm 0.25$<small> |  $9.42$ <small>$\pm 0.11$<small> |
|                | $\texttt{AALV}$ |    $72.41$ <small>$\pm 0.24$<small>    | $16.19$ <small>$\pm 0.09$<small> |  $9.64$ <small>$\pm 0.13$<small> |
|                |  $\texttt{PPA}$ |    $67.51$ <small>$\pm 0.16$<small>    | $17.99$ <small>$\pm 0.42$<small> | $11.15$ <small>$\pm 0.12$<small> |
|                |  $\texttt{CF}$  |    $72.86$ <small>$\pm 0.40$<small>    | $17.22$ <small>$\pm 0.40$<small> | $10.56$ <small>$\pm 0.08$<small> |


# Citation
```latex
@article{ozer2022combine,
  title={How to Combine Variational Bayesian Networks in Federated Learning},
  author={Ozer, Atahan and Buldu, Kadir Burak and Akg{\"u}l, Abdullah and Unal, Gozde},
  journal={arXiv preprint arXiv:2206.10897},
  year={2022}
}
```

# Contributors

| Name | Email |  Github |
| ------ | ------ | ------ |
| Kadir Burak Buldu | buldu19@itu.edu.tr | [buldubu](https://github.com/buldubu) |
