import torch.nn as nn
import torch.nn.functional as F
from .layers import MLP, ResidualQuantizationLayer

class QAE(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size, num_levels, codebook_size, dropout):
        super(QAE, self).__init__()
        self.encoder = MLP(input_size, hidden_sizes, latent_size, dropout=dropout)
        self.quantization_layer = ResidualQuantizationLayer(num_levels, codebook_size, latent_size)
        hidden_sizes.reverse()
        self.decoder = MLP(latent_size, hidden_sizes, input_size, dropout=dropout)

    def forward(self, x):
        encoded = self.encoder(x)
        _, quantized_x, num_small_clusters, quant_loss = self.quantization_layer(encoded)
        decoded = self.decoder(quantized_x)
        return decoded, quant_loss, num_small_clusters
    
    def encode(self, x):
        encoded =  self.encoder(x)
        output, _, _, _ = self.quantization_layer(encoded)
        return output.detach().cpu().numpy()
    
    def encode_with_residual(self, x):
        encoded = self.encoder(x)
        output, quantized_x, _, _ = self.quantization_layer(encoded)
        return output.detach().cpu().numpy(), (encoded - quantized_x).detach().cpu().numpy()

    def generate_codebook(self, x, device):
        encoded = self.encoder(x)
        self.quantization_layer.generate_codebook(encoded, device)