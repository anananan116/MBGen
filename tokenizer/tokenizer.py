import numpy as np
import sklearn.cluster
from .utils import SequenceGenerator, BasekMBConverter, read_tsv_data, QAE_Kmeans_item_Tokenizer
from concurrent.futures import ProcessPoolExecutor, as_completed
from .trainer import train
from tqdm import tqdm
import torch
import os
import pickle
import sklearn

def custom_chunk(data, num_chunks):
    chunk_size = len(data) // num_chunks
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i + chunk_size])
    if len(chunks) > num_chunks:
        # Handle the last few elements by merging them with the last chunk
        last_chunk = chunks[-2] + chunks[-1]
        chunks = chunks[:-2] + [last_chunk]
    return chunks

def process_chunk(user_chunk, sequence_chunk, behavior_chunk, num_user_tokens, id_offsets, sequence_generator, behavior_token=True):
    train_sequence, train_attention_mask, train_label = [], [], []
    for i in range(len(user_chunk)):
        input_ids, attention_mask, labels = sequence_generator.generate_training_sequence(
            user_chunk[i] % num_user_tokens + id_offsets[-1],
            sequence_chunk[i],
            behavior_chunk[i],
            behavior_token
        )
        train_sequence.extend(input_ids)
        train_attention_mask.extend(attention_mask)
        train_label.extend(labels)
    return train_sequence, train_attention_mask, train_label

class Tokenizer(object):
    def __init__(self, config):
        self.config = config
        self.max_seqence_length = config['max_seq_len']
        self.item2idx = {}
        self.id_offsets = config['id_offsets']
        self.item_len = config['item_len']
        self.num_user_tokens = config['num_user_tokens']
        self.min_seq_len = config['min_seq_len']
        if 'no_behavior_token' in config.keys():
            self.behavior_token = not config['no_behavior_token']
        else:
            self.behavior_token = True
        self.sequence_generator = None
        
    def tokenize(self, users, sequences, behaviors):
        num_cores = 16
        user_chunks = custom_chunk(users, num_cores)
        sequence_chunks = custom_chunk(sequences, num_cores)
        behavior_chunks = custom_chunk(behaviors, num_cores)

        combined_sequences = []
        combined_masks = []
        combined_labels = []
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(process_chunk, user_chunks[i], sequence_chunks[i], behavior_chunks[i], self.num_user_tokens, self.id_offsets, self.sequence_generator, behavior_token = self.behavior_token) for i in range(num_cores)]
            for future in as_completed(futures):
                combined_sequences.extend(future.result()[0])
                combined_masks.extend(future.result()[1])
                combined_labels.extend(future.result()[2])

        return {
            'input_ids': torch.tensor(np.array(combined_sequences, dtype=np.int32), dtype=torch.int32),
            'attention_mask': torch.tensor(np.array(combined_masks, dtype=np.int8), dtype=torch.int8),
            'labels': torch.tensor(np.array(combined_labels, dtype=np.int32), dtype=torch.int32)
        }

    def tokenize_evaluation(self, users, sequences, behaviors):
        eval_sequence = []
        eval_attention_mask = []
        eval_label = []
        if self.behavior_token:
            for i in tqdm(range(len(sequences))):
                input_ids, attention_mask, labels = self.sequence_generator.generate_input_sequence(users[i]%self.num_user_tokens + self.id_offsets[-1], sequences[i], behaviors[i])
                eval_sequence.append(input_ids)
                eval_attention_mask.append(attention_mask)
                eval_label.append(labels)
        else:
            for i in tqdm(range(len(sequences))):
                input_ids, attention_mask, labels = self.sequence_generator.generate_input_sequence(users[i]%self.num_user_tokens + self.id_offsets[-1], sequences[i])
                eval_sequence.append(input_ids)
                eval_attention_mask.append(attention_mask)
                eval_label.append(labels)
        return {'input_ids': torch.tensor(np.array(eval_sequence), dtype=torch.long), 'attention_mask': torch.tensor(np.array(eval_attention_mask), dtype=torch.long), 'labels': torch.tensor(np.array(eval_label), dtype=torch.long)}
    
    def tokenize_from_file(self, path):
        user, sequences, behaviors = read_tsv_data(path)
        # if os.path.exists(f"{path}_{self.config['tokenizer_type']}_exp{self.config['exp_id']}.pkl"):
        #     with open(f"{path}_{self.config['tokenizer_type']}_exp{self.config['exp_id']}.pkl", 'rb') as f:
        #         processed = pickle.load(f)
        #         return processed
        processed = self.tokenize(user, sequences, behaviors)
        # with open(f"{path}_{self.config['tokenizer_type']}_exp{self.config['exp_id']}.pkl", 'wb') as f:
        #     pickle.dump(processed, f)
        return processed
    
    def tokenize_eval_from_file(self, path):
        user, sequences, behaviors = read_tsv_data(path)
        return self.tokenize_evaluation(user, sequences, behaviors)

def identity(x):
    return x

class BasekTokenizer(Tokenizer):
    def __init__(self, config, device, item_map=None):
        super().__init__(config)
        self.K = config['K']
        self.item2idx = BasekMBConverter(self.id_offsets, self.K,item_map if item_map is not None else identity, self.item_len, reverse_bt=config['reverse_bt'])
        self.sequence_generator = SequenceGenerator(self.id_offsets[-1] + self.num_user_tokens, self.max_seqence_length, self.item2idx, min_seq_len=self.min_seq_len)
           
class QAE_Kmeans_tokenizer(Tokenizer):
    def __init__(self, config, device, item_map=None):
        super().__init__(config)
        if not os.path.exists(f"./tokenizer/ID/{config['RQVAE']['save_name']}_residual"):
            embeddings = self.generate_embedding(config)
            train(config['RQVAE'], embeddings,  device=device)
        id_list, residual = pickle.load(open(f"./tokenizer/ID/{config['RQVAE']['save_name']}_residual", 'rb'))
        if not os.path.exists(f"./tokenizer/ID/{config['RQVAE']['save_name']}_kmeans.pkl"):
            index_map = {}
            for i, id in enumerate(id_list):
                id = id[0]
                if id not in index_map.keys():
                    index_map[id] = [i]
                else:
                    index_map[id].append(i)
            new_id_list = np.zeros((residual.shape[0], 2))
            for i, (k, v) in enumerate(index_map.items()):
                first_id = k
                residuals = residual[v]
                kmeans = sklearn.cluster.KMeans(n_clusters=config['num_id2'], n_init='auto').fit(residuals)
                labels = kmeans.predict(residuals)
                if config['expand_id2']:
                    for index, label in zip(v, labels):
                        new_id_list[index] = [first_id, label + i * config['num_id2']]
                else:
                    for index, label in zip(v, labels):
                        new_id_list[index] = [first_id, label]
            pickle.dump(new_id_list, open(f"./tokenizer/ID/{config['RQVAE']['save_name']}_kmeans.pkl", 'wb'))
        else:
            new_id_list = pickle.load(open(f"./tokenizer/ID/{config['RQVAE']['save_name']}_kmeans.pkl", 'rb'))
        self.id_map = {i: v for i,v in enumerate(new_id_list)}
        self.item2idx = QAE_Kmeans_item_Tokenizer(self.id_offsets, self.id_map ,item_map if item_map is not None else identity, reverse_bt=config['reverse_bt'])
        self.sequence_generator = SequenceGenerator(self.id_offsets[-1] + self.num_user_tokens, self.max_seqence_length, self.item2idx, min_seq_len=self.min_seq_len)
    def generate_embedding(self, config):
        return pickle.load(open('./tokenizer/embedding.pkl', 'rb')).detach().numpy()