from torch import nn
import torch
from sklearn.cluster import KMeans
import numpy as np
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, dropout=0.0):
        super(MLP, self).__init__()
        mlp_modules = []
        hidden_sizes = [input_size] + hidden_sizes + [latent_size]
        for idx, (input_size, output_size) in enumerate(
            zip(hidden_sizes[:-1], hidden_sizes[1:])
        ):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            activation_func = nn.ReLU()
            if idx != len(hidden_sizes) - 2:
                mlp_modules.append(activation_func)
        self.mlp = nn.Sequential(*mlp_modules)
    def forward(self, x):
        return self.mlp(x)
    
    

class ResidualQuantizationLayer(nn.Module):
    def __init__(self, num_levels, codebook_size, latent_size, decay=0.99, eps=1e-5):
        super(ResidualQuantizationLayer, self).__init__()
        self.latent_size = latent_size
        self.decay = decay
        self.eps = eps
        self.quantization_layer = QuantizationLayer(latent_size, codebook_size, decay, eps)
        
    def forward(self, x):
        batch_size, _ = x.shape
        quantized_x = torch.zeros(batch_size, self.latent_size, device=x.device)
        sum_quant_loss = 0.0
        num_small_clusters = 0.0
        output = torch.empty(batch_size, 1, dtype=torch.long, device=x.device)
        quant, quant_loss, n_small_clusters, output[:, 0] = self.quantization_layer(x)
        x = x - quant
        quantized_x += quant
        sum_quant_loss += quant_loss
        num_small_clusters += n_small_clusters
        return output, quantized_x, num_small_clusters, sum_quant_loss
    
    def generate_codebook(self, x, device):
        x = self.quantization_layer.generate_codebook(x, device)
            

class QuantizationLayer(torch.nn.Module):
    def __init__(self, latent_dimension, code_book_size, decay=0.99, eps=1e-5):
        super(QuantizationLayer, self).__init__()
        self.dim = latent_dimension
        self.n_embed = code_book_size
        self.decay = decay
        self.eps = eps

        embed = torch.zeros(latent_dimension, code_book_size)
        self.embed = torch.nn.Parameter(embed, requires_grad=True)
        self.register_buffer("cluster_size", torch.zeros(code_book_size))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, x):
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(x.dtype)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = x.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            small_clusters = self.cluster_size < 1.0
            n_small_clusters = small_clusters.sum().item()
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)
        else:
            embed_onehot_sum = embed_onehot.sum(0)
            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            small_clusters = self.cluster_size < 1.0
            n_small_clusters = small_clusters.sum().item()
        quant_loss = torch.nn.functional.mse_loss(quantize.detach(), x)
        quantize = (x + (quantize - x).detach())
        return quantize, quant_loss, n_small_clusters, embed_ind
    
    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

    def encode_to_id(self, x):
        flatten = x.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        
        return embed_ind
    
    def generate_codebook(self, x, device):
        kmeans = KMeans(n_clusters=self.n_embed, n_init='auto').fit(x.detach().cpu().numpy())
        self.embed.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device).view(self.dim, self.n_embed)
        self.embed_avg.data = torch.tensor(kmeans.cluster_centers_, dtype=torch.float, device=device).view(self.dim, self.n_embed)
        self.cluster_size.data = torch.tensor(np.bincount(kmeans.labels_), dtype=torch.float, device=device)
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * x @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_ind = embed_ind.view(*x.shape[:-1])
        quantize = self.embed_code(embed_ind)
        return x - quantize