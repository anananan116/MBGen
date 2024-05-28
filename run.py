import argparse
import yaml
import os
from trainer.Trainer import TIGERTrainer
from tokenizer.tokenizer import BasekTokenizer, QAE_Kmeans_tokenizer
from model.PBA_transformer import PBATransformerConfig, PBATransformersForConditionalGeneration
from tokenizer.utils import generate_random_map
from data.preprocessing import preprocess
from transformers import T5Config,T5ForConditionalGeneration
import torch
from itertools import product
import multiprocessing
import threading
from multiprocessing import Manager, Process, Queue
import copy
import gc
import time
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data):
        data['input_ids'] = data['input_ids'].to(torch.int32)
        data['attention_mask'] = data['attention_mask'].to(torch.int8)
        data['labels'] = data['labels'].to(torch.int32)
        self.data = data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.data.items()}
        return item
    
tokenizer_dict = {'Chunked': BasekTokenizer,
                  'QAE_Kmeans': QAE_Kmeans_tokenizer}

def generate_config_combinations(tuning_config, defaults):
    def generate_combinations(options):
        keys = options.keys()
        values = options.values()
        print(product(*values))
        for combination in product(*values):
            yield dict(zip(keys, combination))
    def merge_dicts(default, override):
        result = copy.deepcopy(default)
        for key, value in override.items():
                result[key] = value
        return result

    combinations = list(generate_combinations(tuning_config))
    combined_configs = []
    for combination in combinations:
        temp_config = copy.deepcopy(defaults)
        temp_config = merge_dicts(temp_config, combination)
        combined_configs.append(temp_config)
    for i, one_comb in enumerate(combined_configs):
        one_comb['tunning_id'] = i
        print(i, one_comb)
        print()
    return combined_configs, combinations

def start_training(config, training_data, validation_data, test_data, validation_all_data, test_all_data, device, multiGPU=False, device_ids=[]):
    if 'T5_type' in config.keys() and config['T5_type'] == 'PBA':
        if 'Moe_behavior_only' not in config.keys():
            config['Moe_behavior_only'] = False
        if 'shared_expert' not in config.keys():
            config['shared_expert'] = False
        model_config = PBATransformerConfig(
            behavior_injection= config['behavior_injection'],
            behavior_embedding_dim = config['behavior_embedding_dim'],
            num_positions = config['num_positions'],
            num_behavior = config['num_behavior'],
            num_layers=config['num_layers'], 
            num_decoder_layers=config['num_decoder_layers'],
            num_sparse_encoder_layers = config['num_sparse_encoder_layers'],
            num_sparse_decoder_layers = config['num_sparse_decoder_layers'],
            sparse_layers_encoder = config['sparse_layers_encoder'],
            sparse_layers_decoder = config['sparse_layers_decoder'],
            behavior_injection_decoder=config['behavior_injection_decoder'],
            behavior_injection_encoder=config['behavior_injection_encoder'],
            Moe_behavior_only = config['Moe_behavior_only'],
            shared_expert = config['shared_expert'],
            d_model=config['d_model'],
            d_ff=config['d_ff'],
            num_heads=config['num_heads'],
            d_kv=config['d_kv'],
            dropout_rate=config['dropout_rate'],
            activation_function=config['activation_function'],
            vocab_size=config['id_offsets'][-1] + config['num_user_tokens'] + 1,
            pad_token_id=0,
            eos_token_id=config['id_offsets'][-1] + config['num_user_tokens'],
            decoder_start_token_id=0,
            feed_forward_proj=config['feed_forward_proj'],
            n_positions=config['n_positions'],
        )
        model = PBATransformersForConditionalGeneration(model_config)
    if multiGPU:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    config['device'] = device
    trainer = TIGERTrainer(config, model)
    train_dataset = CustomDataset(training_data)
    validation_dataset = CustomDataset(validation_data)
    validation_all_dataset = CustomDataset(validation_all_data)
    test_dataset = CustomDataset(test_data)
    test_all_dataset = CustomDataset(test_all_data)
    best_val = trainer.train(train_dataset, validation_dataset, validation_all_dataset)
    test_result = trainer.test(test_dataset, test_all_dataset)
    del(model)
    torch.cuda.empty_cache()
    gc.collect()
    if 'tunning_id' in config.keys():
        return {'tunning_id': config['tunning_id'], 'best_val': best_val, 'test_result': test_result}
    return {'best_val': best_val, 'test_result': test_result}

def worker(device_id, configs, training_data, validation_data, test_data, validation_data_all, test_data_all, performance_dict):
    while not configs.empty():
        config, section_combination = configs.get()
        performance = start_training(config, training_data, validation_data, test_data, validation_data_all, test_data_all,  torch.device(f'cuda:{device_id}'))
        performance_dict[str(section_combination)] = performance
        time.sleep(5)

def manage_training(configs, device_ids, training_data, validation_data, test_data, validation_data_all, test_data_all):
    manager = Manager()
    performance_dict = manager.dict()
    config_queue = Queue()

    # Fill the device queue with available device IDs
    for config in configs:
        config_queue.put(config)

    # Start worker processes
    processes = [Process(target=worker, args=(device_id, config_queue, training_data, validation_data, test_data, validation_data_all, test_data_all, performance_dict)) for device_id in device_ids]

    for process in processes:
        process.start()
        time.sleep(5)

    for process in processes:
        process.join()

    # Log performance results
    with open('performance_log.txt', 'w') as f:
        for section_combination, performance in performance_dict.items():
            f.write(f'{section_combination}: {performance}\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config/VQVAE_tokenizer/vqvae_embedding_noexpand.yaml', help='config file')
    parser.add_argument('--dataset', type=str, default='retail', help='dataset name')
    parser.add_argument('--device', type=str, nargs='+', default=['0'], help='device(s) to use for training. Default is 0. Multiple devices can be specified separated by spaces.')
    parser.add_argument("--tunning", action='store_true')
    parser.add_argument("--multiGPU", action='store_true')
    parser.add_argument('--tunning_config', type=str, default='./config/detached_tiger_tunning.yaml', help='hyperparameter tunning config file')
    args, unparsed = parser.parse_known_args()
    if args.tunning:
        multiprocessing.set_start_method('forkserver')
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if 'reverse_bt' not in config.keys():
            config['reverse_bt'] = False
    dataset_full_name = f"{args.dataset}_argumented_{config['item_augmented']}_seqaug_{config['seq_augmented']}"
    config['dataset'] = dataset_full_name
    if not os.path.exists(f'./data/processed/{config["dataset"]}'):
        preprocess.preprocess(args.dataset, config['item_augmented'], config['seq_augmented'], path = os.path.join("data", "raw_dataset"), output_path = os.path.join("data", "processed"))
    random_map = generate_random_map([f'./data/processed/{config["dataset"]}/{config["dataset"]}.train.inter',
                                      f'./data/processed/{config["dataset"]}/{config["dataset"]}.val.inter',
                                      f'./data/processed/{config["dataset"]}/{config["dataset"]}.test.inter',
                                     f'./data/processed/{config["dataset"]}/{config["dataset"]}.val_all.inter',
                                     f'./data/processed/{config["dataset"]}/{config["dataset"]}.test_all.inter'])
    if 'RQVAE' in config.keys():
        config['train_data'] = f'./data/processed/{config["dataset"]}/{config["dataset"]}.train.inter'
    if 'random_map' in config.keys() and config['random_map']:
        tokenizer = tokenizer_dict[config['tokenizer_type']](config, torch.device(f'cuda:{args.device[0]}'), item_map=random_map)
    else:
        tokenizer = tokenizer_dict[config['tokenizer_type']](config, torch.device(f'cuda:{args.device[0]}'))
    print(f"Using device: {args.device}")
    training_data = tokenizer.tokenize_from_file(f'./data/processed/{config["dataset"]}/{config["dataset"]}.train.inter')
    validation_data = tokenizer.tokenize_eval_from_file(f'./data/processed/{config["dataset"]}/{config["dataset"]}.val.inter')
    test_data = tokenizer.tokenize_eval_from_file(f'./data/processed/{config["dataset"]}/{config["dataset"]}.test.inter')
    validation_data_all = tokenizer.tokenize_eval_from_file(f'./data/processed/{config["dataset"]}/{config["dataset"]}.val_all.inter')
    test_data_all = tokenizer.tokenize_eval_from_file(f'./data/processed/{config["dataset"]}/{config["dataset"]}.test_all.inter')
    torch.cuda.empty_cache()
    gc.collect()
    if args.tunning:
        with open(args.tunning_config, 'r') as f:
            tuning_config = yaml.load(f, Loader=yaml.FullLoader)
        combined_configs, section_combinations = generate_config_combinations(tuning_config, config)
        configs = list(zip(combined_configs, section_combinations))

        manage_training(configs, args.device, training_data, validation_data, test_data, validation_data_all, test_data_all)
    else:
        start_training(config, training_data, validation_data, test_data, validation_data_all, test_data_all, f'cuda:{args.device[0]}')
