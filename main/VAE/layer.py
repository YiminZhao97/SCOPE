import torch
import torch.nn as nn
from torch.distributions import Normal

def build_mlp(layers, activation=nn.LeakyReLU(), bn=False, ln=True, dropout: float = 0.1):
    """
    Build multilayer linear perceptron
    bn: batch normalization
    ln: layer normalization
    """
    net = []
    for i in range(1, len(layers)):
        net.append(nn.Linear(layers[i-1], layers[i]))
        if bn:
            net.append(nn.BatchNorm1d(layers[i])) #, momentum=0.01, eps=0.001
        if ln:
            net.append(nn.LayerNorm(layers[i], elementwise_affine=False))
        net.append(activation)
        if dropout > 0:
            net.append(nn.Dropout(dropout))
    return nn.Sequential(*net)

build_mlp([10]+[]+[3],bn=True, ln=False, dropout = 0)
#

def reparameterize_gaussian(mu, var):
    return Normal(mu, var.sqrt()).rsample()

class Encoder(nn.Module):
    """
    dims = [input_dim, [hidden_dims], latent_dim]
    """

    def __init__(self, dims, dropout=0.1, bn=False):
        super().__init__()
        [input_dim, h_dim, z_dim] = dims
        self.encoder = build_mlp([input_dim]+h_dim, bn=bn, dropout=dropout)
        self.mean_encoder = nn.Linear(([input_dim]+h_dim)[-1], z_dim)
        self.var_encoder = nn.Linear(([input_dim]+h_dim)[-1], z_dim)

    def forward(self, x):
        q = self.encoder(x)
        q_m = self.mean_encoder(q)
        q_v = torch.exp(self.var_encoder(q)) + 1e-4
        #latent = reparameterize_gaussian(q_m, q_v)
        #return q_m, q_v, latent
        return q_m, q_v
    
class NB2(nn.Module):
    """
    dims = [latent_dim, [hidden_dims], input_dim]
    decode porportion and dispersion of NB
    """
    def __init__(self, dims, dropout=0.1, bn=False):
        super().__init__()
        [z_dim, h_dim, x_dim] = dims

        self.hidden = build_mlp([z_dim, *h_dim], bn=bn, dropout=dropout)
        
        #positive support
        #mean of NB
        self.alpha_scale_decoder = nn.Sequential(nn.Linear([z_dim, *h_dim][-1], x_dim), nn.Softmax(dim=-1))

        #positive support
        #over dispersion of NB
        self.beta_decoder = nn.Sequential(nn.Linear([z_dim, *h_dim][-1], x_dim), nn.ReLU())

    def forward(self, dispersion, x, library):
        x = self.hidden(x)

        # Clamp to high value: exp(12) ~ 160000 to avoid nans
        #pdb.set_trace()
        alpha_scale = self.alpha_scale_decoder(x)
        alpha = library * alpha_scale
        beta = self.beta_decoder(x) + 1e-8

        return alpha, beta
    
