import numpy as np
import pdb
import torch
import torch.nn as nn
from torch.nn import init
from torch.distributions import Normal, Poisson, kl_divergence as kl
from layer import build_mlp, Encoder, NB2, reparameterize_gaussian
from loss import log_nb_likelihood
import random
import pdb

norm2 = nn.PairwiseDistance(p=2)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path 

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.path + '/checkpoint.pt')     
        self.val_loss_min = val_loss


class VAE(nn.Module):
    def __init__(self, dims, bn=False, dropout=0.1, log_variational=True, dispersion = 'gene'):
        """
        Variational Autoencoder [Kingma 2013] model
        consisting of an encoder/decoder pair for which
        a variational distribution is fitted to the
        encoder. Also known as the M1 model in [Kingma 2014].
        :param dims: x, z and hidden dimensions of the networks
        """
        super(VAE, self).__init__()
        [rna_input_dim, rna_output_dim, z_dim, encoder_hidden, decoder_hidden] = dims
        self.log_variational = log_variational
        self.dispersion = dispersion
        
        if self.dispersion == 'gene':
            #beta is gene specific
            self.beta = torch.nn.Parameter(torch.randn(dims[1]))
        elif self.dispersion == 'gene-cell':
            pass

        #Encoders for rna-seq and atac-seq data
        self.encoder = Encoder([rna_input_dim, encoder_hidden, z_dim], bn=bn, dropout=dropout)
        
        #Decoders for rna-seq and atac-seq data
        self.decoder = NB2([z_dim, decoder_hidden, rna_output_dim], bn = bn, dropout = dropout)

        #record loss
        self.train_loss = []
        self.valid_loss = []

        #initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize weights
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def get_expression_reconstruction_loss(self, x, alpha, beta):
        """
        Return the reconstruction loss (for a minibatch)
        """
        #pdb.set_trace()
        reconst_loss = -log_nb_likelihood(x, alpha, beta).sum(dim=-1)
        return reconst_loss

    def compute_rna_library_size(self, x):
        lib_size = x.sum(dim = -1).reshape(-1, 1).expand(x.size())
        return lib_size

    def inference(self, x, rna_library):
        #x batch size
        #to improve numerical stability
        #pdb.set_trace()
        #rna_library = self.compute_rna_library_size(x)
        #atac_library = self.compute_atac_library_size(y)
        
        if self.log_variational:
            #batch size *  gene
            x_ = torch.log(x + 1)
        
        #x_ = x_[:, variable_genes]
        
        #qz_m batch size * latent dim
        qz_m, qz_v = self.encoder(x_)
        
        #sample from posterior distribution
        z = reparameterize_gaussian(qz_m, qz_v)

        #pdb.set_trace()
        #generate parameters for decoder1
        alpha = self.decoder(self.dispersion, z, rna_library)[0]
        beta = torch.exp(self.beta)

        return dict(
            #local_l_mean=local_l_mean,
            #local_l_var=local_l_var,
            rna_library = rna_library,
            alpha=alpha,
            beta=beta,
            qz_m=qz_m,
            qz_v=qz_v,
            z=z
        )

    def loss_function(self, x, rna_library):

        #pdb.set_trace()
        outputs = self.inference(x, rna_library)
        rna_library = outputs["rna_library"]
        qz_m = outputs["qz_m"]
        qz_v = outputs["qz_v"]
        alpha = outputs["alpha"]
        beta = outputs["beta"]

        #mean batch size * dim of z
        mean = torch.zeros_like(qz_m)
        scale = torch.ones_like(qz_v)
        
        #size: batch size
        kl_divergence_z = kl(Normal(qz_m, torch.sqrt(qz_v)), Normal(mean, scale)).sum(dim=1)

        #pdb.set_trace()
        rna_reconst_loss = self.get_expression_reconstruction_loss(x, alpha, beta)

        return rna_reconst_loss, kl_divergence_z

    def forward(self, x):
        outputs = self.inference(x)
        return outputs['alpha'], outputs['beta'], outputs['z']

    def get_latent(self, dataloader, device='cuda', give_mean = True):
        """
        give_mean: True, give the posterior mean as the latent space; False, use sample from posterior distribution as the latent space
        """
        output = []
        
        for rna_library, x in dataloader:
            #pdb.set_trace()
            #x = x.squeeze(1) 
            x = x.view(x.size(0), -1).float().to(device)
            #rna_library = self.compute_rna_library_size(x)
            rna_library = rna_library.float().to(device)
            
            x_ = torch.log(x + 1)
            #x_ = x_[:, variable_genes]
           
            qz_m, qz_v = self.encoder(x_)

            z = reparameterize_gaussian(qz_m, qz_v)

            if give_mean:
                output.append(qz_m.detach().cpu().data)
            else:
                output.append(z.detach().cpu().data)
        output = torch.cat(output, 0).numpy()
        return output

    def get_RNA_imputation(self, dataloader, device='cuda', use_mean = True):
        """
        Do RNA imputation, no ATAC
        use_mean: True: use posterior mean to be z; False, sample from posterior mean to be z
        """
        output = []
        for rna_library, x in dataloader:
            rna_library = rna_library.to(device)
            x = x.view(x.size(0), -1).float().to(device)
            x_ = torch.log(x + 1)
            #x_ = x_[:, variable_genes]
            qz_m, qz_v = self.encoder(x_)

            if use_mean:
                z = qz_m
            else:
                z = reparameterize_gaussian(qz_m, qz_v)

            alpha, beta = self.decoder(self.dispersion, z, rna_library)
            output.append(alpha.detach().cpu().data)
        output = torch.cat(output, 0).numpy()
        return output

   
def train(model, trainloader, validloader, lr = 0.002, weight_decay=5e-4, beta = 1,
        device = 'cpu', n_epoch = 300, warm_up_epoch = 8, patience = 20, savepath = './'):
    print(device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    #Beta = DeterministicWarmup(warm_epoch, t_max=beta)
    beta = 1e-5
    
    early_stopping = EarlyStopping(path = savepath, patience = patience, verbose=True)
    
    for epoch in range(n_epoch):
        epoch_loss_train = 0
        epoch_loss_valid = 0
        #lr_init = optimizer.param_groups[0]['lr']
        #print(optimizer.param_groups[0]['lr'])
        #adjust_learning_rate(lr_init, optimizer, epoch)
        print(epoch)

        for i, (rna_library, x) in enumerate(trainloader):
            #pdb.set_trace()
            x = x.squeeze(1) # convert [batch size, 1, num of gene] to [batch, num of gene]
            x = x.float().to(device)
            rna_library = rna_library.float().to(device)
            optimizer.zero_grad()
            
            rna_reconst_loss,kl_z = model.loss_function(x, rna_library)

            print("RNA reconstruction loss is {}".format((rna_reconst_loss) / len(x)))
            print("Divergence is {}".format(torch.mean(kl_z)))

            loss = (rna_reconst_loss) / len(x)  + beta * torch.mean(kl_z)

            epoch_loss_train += loss.item()
            print('ELBO is {}'.format(loss))
            print('__________________________________')
            
            #torch.nn.utils.clip_grad_norm(self.parameters(), 10) # clip
            
            loss.backward()
            optimizer.step()
        model.train_loss.append(epoch_loss_train)
        print('train_loss: {}'.format(model.train_loss[-1]))
        
        for i, (rna_library, x) in enumerate(validloader):
            x = x.squeeze(1) 
            x = x.float().to(device)
            rna_library = rna_library.float().to(device)
            
            rna_reconst_loss, kl_z = model.loss_function(x, rna_library)

            loss = (rna_reconst_loss) / len(x)  + beta * torch.mean(kl_z)
            
            print(loss)
            epoch_loss_valid += loss.item()
            
        model.valid_loss.append(epoch_loss_valid)
        
        print('valid_loss: {}'.format(model.valid_loss[-1]))
        
        early_stopping(model.valid_loss[-1], model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        print('_________________________________________')
        
        model.load_state_dict(torch.load(savepath + '/checkpoint.pt'))
    return model
        


class SoftmaxClassifier1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SoftmaxClassifier1, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        logits = self.linear(x)
        probabilities = torch.softmax(logits, dim=1)
        return probabilities

class SoftmaxClassifier2(nn.Module):
    def __init__(self, dims):
        super(SoftmaxClassifier2, self).__init__()
        [input_dim, h_dim, output_dim] = dims
        self.layer = build_mlp([input_dim]+h_dim+[output_dim], dropout = 0)
    def forward(self, x):
        out = self.layer(x)
        probabilities = torch.softmax(out, dim=1)
        return probabilities

class SoftmaxClassifier3(nn.Module):
    def __init__(self, dims):
        super(SoftmaxClassifier3, self).__init__()
        [input_dim, h_dim, output_dim] = dims
        self.layer = build_mlp([input_dim]+h_dim+[output_dim],bn=False, ln=False, dropout = 0)
    def forward(self, x):
        out = self.layer(x)
        probabilities = torch.softmax(out, dim=1)
        return probabilities


"""
class SoftmaxClassifier(nn.Module):
    def __init__(self, dims):
        super(SoftmaxClassifier, self).__init__()
        [input_dim, h_dim, output_dim] = dims
        self.layer = build_mlp([input_dim]+h_dim+[output_dim], dropout = 0)
    def forward(self, x):
        out = self.layer(x)
        probabilities = torch.softmax(out, dim=1)
        return probabilities


class SoftmaxClassifier(nn.Module):
    def __init__(self, dims):
        super(SoftmaxClassifier, self).__init__()
        [input_dim, h_dim, output_dim] = dims
        self.layer1 = nn.Linear(input_dim, h_dim)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(h_dim, output_dim)
    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        probabilities = torch.softmax(out, dim=1)
        return probabilities
"""
