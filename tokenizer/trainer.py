import os
from tqdm import tqdm
from .models.QAE import QAE
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
DEFAULT_CONFIG = {
    'dataset': 'Sports_and_Outdoors',
    'RQ-VAE': {
        'batch_size': 2048,
        'epochs': 5000,
        'lr': 0.001,
        'beta': 0.25,
        'input_dim': 64,
        'hidden_dim': [2048, 1024, 512, 256],
        'latent_dim': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'code_book_size': 256, # 'code_book_size': [4, 16, 256]
        'max_seq_len': 256,
        'val_ratio': 0.05,
        'batch_norm': True,
        'standardize': True
    }
}

def train_rqvae(model, x, device, num_epochs=10000, lr=1e-3, batch_size=2048, beta_1 = 0.25, beta_2=0.25):
    model.to(device)
    print(model)
    model.generate_codebook(torch.Tensor(x).to(device), device)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)
    trainset, validationset = train_test_split(x, test_size=0.05, random_state=42)
    train_dataset = TensorDataset(torch.Tensor(trainset).to(device))
    val_dataset = TensorDataset(torch.Tensor(validationset).to(device))
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    beta = beta_1
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        total_rec_loss = 0.0
        total_quant_loss = 0.0
        total_count = 0
        for batch in dataloader:
            x_batch = batch[0]
            optimizer.zero_grad()
            recon_x, quant_loss, count = model(x_batch)
            reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
            loss = reconstruction_mse_loss + beta * quant_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_rec_loss += reconstruction_mse_loss.item()
            total_quant_loss += quant_loss.item()
            total_count += count
        if (epoch + 1) % 100 == 0:
            total_val_loss = 0.0
            total_val_rec_loss = 0.0
            total_val_quant_loss = 0.0
            total_val_count = 0
            model.eval()
            with torch.no_grad():
                for batch in val_dataloader:
                    x_batch = batch[0]
                    recon_x, quant_loss, count = model(x_batch)
                    reconstruction_mse_loss = F.mse_loss(recon_x, x_batch, reduction='mean')
                    loss = reconstruction_mse_loss + beta * quant_loss
                    total_val_loss += loss.item()
                    total_val_rec_loss += reconstruction_mse_loss.item()
                    total_val_quant_loss += quant_loss.item()
                    total_val_count += count
            print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {total_loss/ len(dataloader)}, unused_codebook:{total_count/ len(dataloader)}")
            print(f"Epoch [{epoch+1}/{num_epochs}], recosntruction_loss: {total_rec_loss/ len(dataloader)}, quantization_loss: {total_quant_loss/ len(dataloader)}")
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {total_val_loss/ len(val_dataloader)}, unused_codebook:{total_val_count/ len(val_dataloader)}")
            print(f"Epoch [{epoch+1}/{num_epochs}], recosntruction_loss: {total_val_rec_loss/ len(val_dataloader)}, quantization_loss: {total_val_quant_loss/ len(val_dataloader)}")
    print("Training complete.")

def train(config, embeddings, device):
    
    input_size = config['input_dim']
    hidden_sizes = config['hidden_dim']
    latent_size = config['latent_dim']
    num_levels = config['num_layers']
    codebook_size = config['code_book_size']
    dropout = config['dropout']
    if config['standardize']:
        embeddings = StandardScaler().fit_transform(embeddings)
    rqvae = QAE(input_size, hidden_sizes, latent_size, num_levels, codebook_size, dropout)
    train_rqvae(rqvae, embeddings, device, batch_size=config['batch_size'], num_epochs=config['epochs'], lr=config['lr'], beta_1=config['beta'])
    rqvae.to(device)
    embeddings_tensor = torch.Tensor(embeddings).to(device)
    rqvae.eval()
    if 'save_residual' in config and config['save_residual']:
        ids = rqvae.encode_with_residual(embeddings_tensor)
        os.makedirs('./tokenizer/ID', exist_ok=True)
        with open(f"./tokenizer/ID/{config['save_name']}_residual", 'wb') as f:
            pickle.dump(ids, f)
    else:
        ids = rqvae.encode(embeddings_tensor)
        # If the ID directory does not exist, create it
        os.makedirs('./tokenizer/ID', exist_ok=True)
        with open(f"./tokenizer/ID/{config['save_name']}", 'wb') as f:
            pickle.dump(ids, f)