import os
from tqdm import tqdm
import numpy as np
from transformers import T5Config,T5ForConditionalGeneration
from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data import Dataset
from transformers.optimization import get_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from .evaluation import evaluate
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
    
class TIGERTrainer(object):
    def __init__(self, trainer_config, model):
        self.total_steps = trainer_config['steps']
        self.batch_size = trainer_config['batch_size']
        self.learning_rate = trainer_config['lr']
        self.eval_batch_size = trainer_config['eval_batch_size']
        self.item_len = trainer_config['item_len']
        self.optimizer = AdamW(model.parameters(), lr=self.learning_rate, weight_decay=trainer_config['weight_decay'])
        self.model = model
        self.device = trainer_config['device']
        model.to(trainer_config['device'])
        self.scheduler = get_scheduler(
        name="cosine",
        optimizer=self.optimizer,
        num_warmup_steps=trainer_config['warmup_steps'],
        num_training_steps=self.total_steps,
        )
        model.train()
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total number of parameters: {total_params}")
        if 'tunning_id' in trainer_config:
            self.writer = SummaryWriter(log_dir=f"./logs/tiger_exp_{trainer_config['exp_id']}/tunning_{trainer_config['tunning_id']}")
            self.save_location = f"./results/tiger_exp_{trainer_config['exp_id']}/tunning_{trainer_config['tunning_id']}"
            if trainer_config['tunning_id'] % 8 == 0:
                self.no_output = False
            else:
                self.no_output = True
        else:
            self.writer = SummaryWriter(log_dir=f"./logs/tiger_exp_{trainer_config['exp_id']}")
            self.save_location = f"./results/tiger_exp_{trainer_config['exp_id']}"
            self.no_output = False
        if not os.path.exists(self.save_location):
            os.makedirs(self.save_location)
        self.expid = trainer_config['exp_id']
        self.scaler = GradScaler(init_scale=2**14)
        self.patience = trainer_config['patience']
        self.auto_save_epochs = trainer_config['auto_save_epochs']
        self.epochs_per_eval = trainer_config['epochs_per_eval']
        if 'epochs_per_all_eval' in trainer_config.keys():
            self.epochs_per_all_eval = trainer_config['epochs_per_all_eval']
        else:
            self.epochs_per_all_eval = trainer_config['epochs_per_eval']
        self.best_performance = {}
        if 'no_behavior_token' in trainer_config:
            self.behavior_token = not trainer_config['no_behavior_token']
        else:
            self.behavior_token = True
        self.reverse_bt = trainer_config['reverse_bt']
        
    def train(self, train_dataset, validation_dataset, validation_all_dataset):
        
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        validation_dataloader = DataLoader(validation_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4)
        if validation_all_dataset is not None:
            validation_all_dataloader = DataLoader(validation_all_dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=4)
        print(f"Total number of epochs{int(np.ceil(self.total_steps / len(train_dataloader)))}")
        best_epoch = 0
        best_ndcg_10 = float('-inf')
        best_perofrmance = {}
        for epoch in range(int(np.ceil(self.total_steps / len(train_dataloader)))):
            if not self.no_output:
                progress_bar = tqdm(range(len(train_dataloader)))
            total_loss = 0.0
            batch_num = 0
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device).to(torch.long)
                attention_mask = batch['attention_mask'].to(self.device).to(torch.long)
                labels = batch['labels'].to(self.device).to(torch.long)
                with autocast():
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = torch.mean(outputs.loss)
                # Scale the loss as necessary
                self.scaler.scale(loss).backward()
                clip_grad_norm_(self.model.parameters(), 1.0)
                # Update optimizer and scheduler using scaled gradients
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                total_loss += loss.item()
                batch_num += 1
                if not self.no_output:
                    progress_bar.set_description(f"Epoch {epoch+1}, Loss: {(total_loss/batch_num):.4f}")
                    progress_bar.update(1)
            avg_loss = total_loss / len(train_dataloader)
            self.writer.add_scalar('Loss/training_loss', avg_loss, epoch)
            if not self.no_output:
                progress_bar.close()
            torch.cuda.empty_cache()
            perf = {}
            if (epoch + 1) % self.epochs_per_eval == 0 and (not self.reverse_bt):
                if self.model is torch.nn.DataParallel:
                    recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(self.model.module, validation_dataloader, self.device, self.item_len, eval_mode = 'Target', no_output=self.no_output, behavior_token=self.behavior_token)
                else:
                    recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(self.model, validation_dataloader, self.device, self.item_len, eval_mode = 'Target', no_output=self.no_output, behavior_token=self.behavior_token)
                perf['Recall@5(target)'] = recall_5
                perf['Recall@10(target)'] = recall_10
                perf['NDCG@5(target)'] = ndcg_5
                perf['NDCG@10(target)'] = ndcg_10
                self.writer.add_scalar('Loss/validation_loss', eval_loss, epoch)
                self.writer.add_scalar('Metrics/Recall@5', recall_5, epoch)
                self.writer.add_scalar('Metrics/Recall@10', recall_10, epoch)
                self.writer.add_scalar('Metrics/NDCG@5', ndcg_5, epoch)
                self.writer.add_scalar('Metrics/NDCG@10', ndcg_10, epoch)
                if ndcg_10 > best_ndcg_10 and (not self.behavior_token):
                        self.best_performance = perf
                        best_ndcg_10 = ndcg_10
                        best_epoch = epoch 
                        self.model.to('cpu')
                        if self.model is torch.nn.DataParallel:
                            torch.save(self.model.module.state_dict(), self.save_location + f"/tiger_best.pt")
                        else:
                            torch.save(self.model.state_dict(), self.save_location + f"/tiger_best.pt")
                        self.model.to(self.device)
            if (epoch + 1) % self.epochs_per_all_eval == 0 and self.behavior_token:
                if validation_all_dataloader is not None:
                    if not self.reverse_bt:
                        # Evaluate Behavior-Specific prediction
                        if self.model is torch.nn.DataParallel:
                            recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(self.model.module, validation_all_dataloader, self.device, self.item_len, no_output=self.no_output, eval_mode = 'Behavior_specific')
                        else:
                            recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(self.model, validation_all_dataloader, self.device, self.item_len, no_output=self.no_output, eval_mode = 'Behavior_specific')
                        self.writer.add_scalar('Loss/validation_loss_behavior_specific', eval_loss, epoch)
                        self.writer.add_scalar('Metrics/Recall@5_behavior_specific', recall_5, epoch)
                        self.writer.add_scalar('Metrics/Recall@10_behavior_specific', recall_10, epoch)
                        self.writer.add_scalar('Metrics/NDCG@5_behavior_specific', ndcg_5, epoch)
                        self.writer.add_scalar('Metrics/NDCG@10_behavior_specific', ndcg_10, epoch)
                        perf['Recall@5_behavior_specific'] = recall_5
                        perf['Recall@10_behavior_specific'] = recall_10
                        perf['NDCG@5_behavior_specific'] = ndcg_5
                        perf['NDCG@10_behavior_specific'] = ndcg_10
                        # Evaluate Behavior-Item prediction
                    if self.model is torch.nn.DataParallel:
                        recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(self.model.module, validation_all_dataloader, self.device, self.item_len, no_output=self.no_output, eval_mode = 'Behavior_item', reverse_bt = self.reverse_bt)
                    else:
                        recall_5,recall_10,ndcg_5,ndcg_10, eval_loss = evaluate(self.model, validation_all_dataloader, self.device, self.item_len, no_output=self.no_output, eval_mode = 'Behavior_item', reverse_bt = self.reverse_bt)
                    self.writer.add_scalar('Loss/validation_loss_all', eval_loss, epoch)
                    self.writer.add_scalar('Metrics/Recall@5_all', recall_5, epoch)
                    self.writer.add_scalar('Metrics/Recall@10_all', recall_10, epoch)
                    self.writer.add_scalar('Metrics/NDCG@5_all', ndcg_5, epoch)
                    self.writer.add_scalar('Metrics/NDCG@10_all', ndcg_10, epoch)
                    perf['Recall@5_all'] = recall_5
                    perf['Recall@10_all'] = recall_10
                    perf['NDCG@5_all'] = ndcg_5
                    perf['NDCG@10_all'] = ndcg_10
                    if ndcg_10 > best_ndcg_10:
                        self.best_performance = perf
                        best_ndcg_10 = ndcg_10
                        best_epoch = epoch 
                        self.model.to('cpu')
                        if self.model is torch.nn.DataParallel:
                            torch.save(self.model.module.state_dict(), self.save_location + f"/tiger_best.pt")
                        else:
                            torch.save(self.model.state_dict(), self.save_location + f"/tiger_best.pt")
                        self.model.to(self.device)
            if best_epoch + self.patience < epoch:
                break
            torch.cuda.empty_cache()
        self.model.load_state_dict(torch.load(self.save_location + f"/tiger_best.pt"))
        self.writer.close()
        with open(self.save_location + f"/results.txt", 'w') as f:
            f.write("Best Validation Results:\n")
            for key, value in self.best_performance.items():
                f.write(f"{key}: {value:.4f}\n")
        return self.best_performance
    
    def test(self, test_dataset, test_all_dataset, load_best=True, num_beams = 50):
        if load_best:
            self.model.load_state_dict(torch.load(self.save_location + f"/tiger_best.pt"))
        test_dataloader = DataLoader(test_dataset, batch_size=self.eval_batch_size//2, shuffle=False)
        test_all_dataloader = DataLoader(test_all_dataset, batch_size=self.eval_batch_size//2, shuffle=False)
        
        results = {}
        
        # Evaluate on test dataset
        if not self.reverse_bt:
            recall_5, recall_10, ndcg_5, ndcg_10, eval_loss = evaluate(self.model, test_dataloader, self.device, self.item_len, eval_mode='Target', no_output=self.no_output, behavior_token=self.behavior_token, num_beams= num_beams)
            results['Test Loss(target)'] = eval_loss
            results['Recall@5(target)'] = recall_5
            results['Recall@10(target)'] = recall_10
            results['NDCG@5(target)'] = ndcg_5
            results['NDCG@10(target)'] = ndcg_10

        print("Target Evaluation Metrics:")
        for key, value in results.items():
            print(f"{key}: {value}")
        if self.behavior_token:
            # Evaluate on all behavioral data
            if not self.reverse_bt:
                recall_5, recall_10, ndcg_5, ndcg_10, eval_loss = evaluate(self.model, test_all_dataloader, self.device, self.item_len, no_output=self.no_output, eval_mode='Behavior_specific', num_beams= num_beams)
                results['Test Loss (Behavior_specific)'] = eval_loss
                results['Recall@5 (Behavior_specific)'] = recall_5
                results['Recall@10 (Behavior_specific)'] = recall_10
                results['NDCG@5 (Behavior_specific)'] = ndcg_5
                results['NDCG@10 (Behavior_specific)'] = ndcg_10

                print("Behavior-specific Evaluation Metrics:")
                for key, value in results.items():
                    if "Behavior_specific" in key:
                        print(f"{key}: {value}")

            # Evaluate on behavior item data
            recall_5, recall_10, ndcg_5, ndcg_10, eval_loss = evaluate(self.model, test_all_dataloader, self.device, self.item_len, no_output=self.no_output, eval_mode='Behavior_item', num_beams= num_beams, reverse_bt = self.reverse_bt)
            results['Test Loss (Behavior_item)'] = eval_loss
            results['Recall@5 (Behavior_item)'] = recall_5
            results['Recall@10 (Behavior_item)'] = recall_10
            results['NDCG@5 (Behavior_item)'] = ndcg_5
            results['NDCG@10 (Behavior_item)'] = ndcg_10

            print("Behavior-item Evaluation Metrics:")
            for key, value in results.items():
                if "Behavior_item" in key:
                    print(f"{key}: {value}")
        
        with open(self.save_location + f"/results.txt", 'a') as f:
            f.write("Test Results:\n")
            for key, value in results.items():
                f.write(f"{key}: {value:.4f}\n")
        return results