import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal

class VBLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(VBLinear, self).__init__()
        self.n_in = in_features
        self.n_out = out_features

        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.mu_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.prior_mu_w = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.prior_mu_w.data.zero_()

        self.logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features))
        self.prior_logsig2_w = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=False)
        self.prior_logsig2_w.data.zero_()       
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.mu_w.size(1))
        self.mu_w.data.normal_(0, stdv)
        self.logsig2_w.data.zero_().normal_(-9, 0.001)  # var init via Louizos
        self.bias.data.zero_()

    def KL(self):
        return torch.distributions.kl.kl_divergence(Normal(self.mu_w, self.logsig2_w.clamp(-8, 8).exp().sqrt()), Normal(self.prior_mu_w, self.prior_logsig2_w.clamp(-8, 8).exp().sqrt()))

    def forward(self, input):
        mu_out = torch.nn.functional.linear(input, self.mu_w, self.bias)
        logsig2_w = self.logsig2_w.clamp(-8, 8)
        s2_w = logsig2_w.exp()
        var_out = torch.nn.functional.linear(input.pow(2), s2_w) + 1e-8
        return mu_out + var_out.sqrt() * torch.randn_like(mu_out)

    def __repr__(self):
        return self.__class__.__name__  + " (" + str(self.n_in) + " -> " + str(self.n_out)  + ")"

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10, input_channel=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BCNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=10, input_channel=3):
        super(BCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        self.fc1 = VBLinear(input_dim, hidden_dims[0])
        self.fc2 = VBLinear(hidden_dims[0], hidden_dims[1])
        self.fc3 = VBLinear(hidden_dims[1], output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.input_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def KL(self):
        return self.KL_with_normal()
        kl = 0
        for layer in self.children():
            if isinstance(layer, VBLinear):
                kl += layer.KL().sum()
        return kl
    
    def KL_with_target(self, target):
        kl = 0
        for layer, target_layer in zip(self.children(), target.children()):
            if isinstance(layer, VBLinear):
                kl += torch.distributions.kl.kl_divergence(Normal(layer.mu_w, layer.logsig2_w.clamp(-8, 8).exp().sqrt()), Normal(target_layer.mu_w, target_layer.logsig2_w.clamp(-8, 8).exp().sqrt())).sum()
        return kl
    
    def KL_with_normal(self):
        kl = 0
        for layer in self.children():
            if isinstance(layer, VBLinear):
                kl += torch.distributions.kl.kl_divergence(Normal(layer.mu_w, layer.logsig2_w.clamp(-8, 8).exp().sqrt()), Normal(0,1)).sum()
        return kl
    
    def update_priors(self, glob):
        for layer, glob_layer in zip(self.children(), glob.children()):
            if isinstance(layer, VBLinear):
                layer.prior_mu_w.data = glob_layer.mu_w.data.clone()
                layer.prior_logsig2_w.data = glob_layer.logsig2_w.data.clone()



class FcNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BFcNN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.fc1 = VBLinear(input_dim, hidden_dims[0])
        self.fc2 = VBLinear(hidden_dims[0], hidden_dims[1])
        self.fc3 = VBLinear(hidden_dims[1], output_dim)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def KL(self):
        kl = 0
        for layer in self.children():
            if isinstance(layer, VBLinear):
                kl += layer.KL().sum()
        return kl
    
    def KL_with_normal(self):
        kl = 0
        for layer in self.children():
            if isinstance(layer, VBLinear):
                kl += torch.distributions.kl.kl_divergence(Normal(layer.mu_w, layer.logsig2_w.clamp(-8, 8).exp()), Normal(0,1)).sum()
        return kl
