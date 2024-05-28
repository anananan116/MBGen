import numpy as np
import argparse
from tqdm import tqdm
import os
from collections import defaultdict

BEHAVIOR_MAP = {"retail":{'buy': 0, 'pv': 1, 'fav': 2, 'cart': 3},\
                "ijcai":{'buy': 0, 'pv': 1, 'fav': 2, 'cart': 3},\
                "yelp":{'pos': 0, 'neg': 1, 'neutral': 2, 'tip': 3}}

def split_history(input_list, max_seq_len = 50):
    sequences = []
    n = len(input_list)
    max_seq_len = min(max_seq_len, n - 1)

    for end_index in range(1, max_seq_len):
        sequence = input_list[:end_index]
        next_item = input_list[end_index]
        sequences.append((sequence, next_item))
    for start_index in range(n - max_seq_len):
        sequence = input_list[start_index:start_index + max_seq_len]
        next_item = input_list[start_index + max_seq_len]
        sequences.append((sequence, next_item))
    
    train = sequences[:-2] if len(sequences) > 2 else []
    val = [sequences[-2]] if len(sequences) > 1 else []
    test = [sequences[-1]] if sequences else []

    return train, val, test

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def preprocess(dataset, augmented, seq_augmented, path = os.path.join("..", "raw_dataset"), output_path = os.path.join("..", "processed"), MBHT = False, interaction = False):
    if dataset not in ["retail", "ijcai", "yelp"]:
        raise ValueError("Wrong Argument")
    data = []
    user_sequence_map = defaultdict(list)
    behavior_sequence_map = defaultdict(list)
    original_user_sequence_map = defaultdict(list)
    with open(os.path.join(path, dataset+".txt"), 'r') as f:
        behavior_map = BEHAVIOR_MAP[dataset]
        for line in f.readlines():
            d = line.split('\t')
            u, i, b, t = int(d[0]), int(d[1]), str(d[2]), float(d[3].strip())
            if augmented:
                augmented_item = i * len(behavior_map) + behavior_map[b]
            else:
                augmented_item = i
            data.append((u, augmented_item, t, behavior_map[b]))
            user_sequence_map[u].append(augmented_item)
            original_user_sequence_map[u].append(i)
            behavior_sequence_map[u].append(behavior_map[b]+1) # add 1 to avoid padding conflict
    item_count = defaultdict(int)
    eligible_items = set()
    # filter out items with less than 5 interactions, since we only take the last 50 interaction sequence, let's insure that the item has at least 5 interactions in the actual training sequence
    if augmented:
        filtering_user_sequence_map = original_user_sequence_map
    else:
        filtering_user_sequence_map = user_sequence_map
    for u in filtering_user_sequence_map:
        train_seq, _, _ = split_history(filtering_user_sequence_map[u])
        if len(train_seq) > 0:
            for i in train_seq[-1][0]:
                item_count[i] += 1
    if augmented:
        for i in item_count:
            if item_count[i] >= 5:
                eligible_items.add(i * len(behavior_map))
                eligible_items.add(i * len(behavior_map) + 1)
                eligible_items.add(i * len(behavior_map) + 2)
                eligible_items.add(i * len(behavior_map) + 3)
    else:
        for i in item_count:
            if item_count[i] >= 5:
                eligible_items.add(i)
    print(f"Number of eligible items: {len(eligible_items)}")
    filtered_user_sequence_map = defaultdict(list)
    filtered_behavior_sequence_map = defaultdict(list)
    id_remap = {}
    interaction_count = 0
    for i, item in enumerate(eligible_items):
        id_remap[item] = i + 1 # add 1 to avoid padding conflict
    if augmented:
        id_remap = {}
        for i, item in enumerate(eligible_items):
            id_remap[item] = item # do not do remapping if item is augmented
    for u in user_sequence_map:
        for i in range(len(user_sequence_map[u])):
            if user_sequence_map[u][i] in eligible_items:
                filtered_user_sequence_map[u].append(id_remap[user_sequence_map[u][i]])
                filtered_behavior_sequence_map[u].append(behavior_sequence_map[u][i])
                interaction_count += 1
    user_sequence_map = filtered_user_sequence_map
    behavior_sequence_map = filtered_behavior_sequence_map
    eligible_users = set()
    for u in user_sequence_map:
        if len(user_sequence_map[u]) >= 4:
            eligible_users.add(u)
    print(f"Number of eligible users: {len(eligible_users)}")
    print(f"Number of interactions: {interaction_count}")
    output_dir = os.path.join(output_path, f"{dataset}_argumented_{augmented}_seqaug_{seq_augmented}")
    file_path = f"{dataset}_argumented_{augmented}_seqaug_{seq_augmented}"
    if MBHT:
        output_dir = os.path.join(output_path, f"{dataset}_argumented_{augmented}_seqaug_{seq_augmented}_MBHT")
        file_path = f"{dataset}_argumented_{augmented}_seqaug_{seq_augmented}_MBHT"
    if interaction:
        output_dir = os.path.join(output_path, f"{dataset}_argumented_{augmented}_seqaug_{seq_augmented}_interaction")
        file_path = f"{dataset}_argumented_{augmented}_seqaug_{seq_augmented}_interaction"
    create_directory_if_not_exists(output_dir)
    
    tvt_sequence_map = defaultdict(tuple)
    tvt_behavior_map = defaultdict(tuple)
    if seq_augmented:
        for u in user_sequence_map:
            if len(user_sequence_map[u]) >= 4:
                tvt_sequence_map[u] = split_history(user_sequence_map[u][-50:], max_seq_len=49 if MBHT else 50)
                tvt_behavior_map[u] = split_history(behavior_sequence_map[u][-50:], max_seq_len=49 if MBHT else 50)
    else:
        for u in user_sequence_map:
            if len(user_sequence_map[u]) >= 4:
                tvt_sequence_map[u] = split_history(user_sequence_map[u], max_seq_len=49 if MBHT else 50)
                tvt_behavior_map[u] = split_history(behavior_sequence_map[u], max_seq_len=49 if MBHT else 50)
    print("Writing training sequences")
    with open(os.path.join(output_dir, file_path+".train.inter"), 'w') as f:
        if interaction:
            f.write('user_id:token\titem_id:token\tbehavior_id:float\n')
        else:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\tbehavior_list:float_seq\n')
        for u in tqdm(tvt_sequence_map):
            if seq_augmented:
                if MBHT:
                    if dataset == 'yelp':
                        last_subsequence_i, last_subsequence_b = None, None
                        for subsequence_i, subsequence_b in zip(tvt_sequence_map[u][0], tvt_behavior_map[u][0]):
                            if subsequence_b[1] == 1: # last interaction is target interaction
                                last_subsequence_i, last_subsequence_b = subsequence_i, subsequence_b
                        if last_subsequence_i is not None:
                            item_id_list = " ".join(map(str, last_subsequence_i[0]))
                            behavior_list = " ".join(map(lambda x: str(x - 1), last_subsequence_b[0]))
                            f.write(f"{u}\t{item_id_list}\t{last_subsequence_i[1]}\t{behavior_list}\n")
                    else:
                        for subsequence_i, subsequence_b in zip(tvt_sequence_map[u][0], tvt_behavior_map[u][0]):
                            if subsequence_b[1] == 1: # last interaction is target interaction
                                item_id_list = " ".join(map(str, subsequence_i[0]))
                                behavior_list = " ".join(map(lambda x: str(x - 1), subsequence_b[0])) # MBHT uses 0-based indexing for behavior
                                f.write(f"{u}\t{item_id_list}\t{subsequence_i[1]}\t{behavior_list}\n")
                else:
                    for subsequence_i, subsequence_b in zip(tvt_sequence_map[u][0], tvt_behavior_map[u][0]):
                        item_id_list = " ".join(map(str, subsequence_i[0]))
                        behavior_list = " ".join(map(str, subsequence_b[0]))
                        f.write(f"{u}\t{item_id_list}\t{subsequence_i[1]}\t{behavior_list}\n")
            elif interaction:
                subsequence_i, subsequence_b = tvt_sequence_map[u][0][-1], tvt_behavior_map[u][0][-1]
                for i,b in zip(subsequence_i[0] + [subsequence_i[1]], subsequence_b[0] + [subsequence_b[1]]):
                    f.write(f"{u}\t{i}\t{b}\n")
            else:
                subsequence_i, subsequence_b = tvt_sequence_map[u][0][-1], tvt_behavior_map[u][0][-1]
                item_id_list = " ".join(map(str, subsequence_i[0]))
                behavior_list = " ".join(map(str, subsequence_b[0] + [subsequence_b[1]]))
                f.write(f"{u}\t{item_id_list}\t{subsequence_i[1]}\t{behavior_list}\n")
    print("Writing validation and test sequences")
    with open(os.path.join(output_dir, file_path+".val.inter"), 'w') as f:
        if not interaction:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\tbehavior_list:float_seq\n')
            for u in tqdm(tvt_sequence_map):
                if (tvt_behavior_map[u][1][0][1] == 1): # last interaction is target interaction
                    if not MBHT:
                        behavior_list = tvt_behavior_map[u][1][0][0] + [tvt_behavior_map[u][1][0][1]]
                    else:
                        behavior_list = map(lambda x: x - 1, tvt_behavior_map[u][1][0][0])
                    behavior_list = " ".join(map(str, behavior_list))
                    item_id_list = " ".join(map(str, tvt_sequence_map[u][1][0][0]))
                    f.write(f"{u}\t{item_id_list}\t{tvt_sequence_map[u][1][0][1]}\t{behavior_list}\n")
        else:
            f.write('user_id:token\titem_id:token\tbehavior_id:float\n')
            for u in tqdm(tvt_sequence_map):
                if (tvt_behavior_map[u][1][0][1] == 1): # last interaction is target interaction
                    behavior = tvt_behavior_map[u][1][0][1]
                    target_item = tvt_sequence_map[u][1][0][1]
                    f.write(f"{u}\t{target_item}\t{behavior}\n")
    with open(os.path.join(output_dir, file_path+".test.inter"), 'w') as f:
        if not interaction:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\tbehavior_list:float_seq\n')
            for u in tqdm(tvt_sequence_map):
                if (tvt_behavior_map[u][2][0][1] == 1): # last interaction is target interaction
                    if not MBHT:
                        behavior_list = tvt_behavior_map[u][2][0][0] + [tvt_behavior_map[u][2][0][1]]
                    else:
                        behavior_list = map(lambda x: x - 1, tvt_behavior_map[u][2][0][0])
                    behavior_list = " ".join(map(str, behavior_list))
                    item_id_list = " ".join(map(str, tvt_sequence_map[u][2][0][0]))
                    f.write(f"{u}\t{item_id_list}\t{tvt_sequence_map[u][2][0][1]}\t{behavior_list}\n")
        else:
            f.write('user_id:token\titem_id:token\tbehavior_id:float\n')
            for u in tqdm(tvt_sequence_map):
                if (tvt_behavior_map[u][2][0][1] == 1): # last interaction is target interaction
                    behavior = tvt_behavior_map[u][2][0][1]
                    target_item = tvt_sequence_map[u][2][0][1]
                    f.write(f"{u}\t{target_item}\t{behavior}\n")
    # save all behavioral types in a seperate file
    with open(os.path.join(output_dir, file_path+".val_all.inter"), 'w') as f:
        if not interaction:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\tbehavior_list:float_seq\n')
            for u in tqdm(tvt_sequence_map):
                if not MBHT:
                    behavior_list = tvt_behavior_map[u][1][0][0] + [tvt_behavior_map[u][1][0][1]]
                else:
                    behavior_list = map(lambda x: x - 1, tvt_behavior_map[u][1][0][0])
                behavior_list = " ".join(map(str, behavior_list))
                item_id_list = " ".join(map(str, tvt_sequence_map[u][1][0][0]))
                f.write(f"{u}\t{item_id_list}\t{tvt_sequence_map[u][1][0][1]}\t{behavior_list}\n")
        else:
            f.write('user_id:token\titem_id:token\tbehavior_id:float\n')
            for u in tqdm(tvt_sequence_map):
                behavior = tvt_behavior_map[u][1][0][1]
                target_item = tvt_sequence_map[u][1][0][1]
                f.write(f"{u}\t{target_item}\t{behavior}\n")
    with open(os.path.join(output_dir, file_path+".test_all.inter"), 'w') as f:
        if not interaction:
            f.write('user_id:token\titem_id_list:token_seq\titem_id:token\tbehavior_list:float_seq\n')
            for u in tqdm(tvt_sequence_map):
                if not MBHT:
                    behavior_list = tvt_behavior_map[u][2][0][0] + [tvt_behavior_map[u][2][0][1]]
                else:
                    behavior_list = map(lambda x: x - 1, tvt_behavior_map[u][2][0][0])
                behavior_list = " ".join(map(str, behavior_list))
                item_id_list = " ".join(map(str, tvt_sequence_map[u][2][0][0]))
                f.write(f"{u}\t{item_id_list}\t{tvt_sequence_map[u][2][0][1]}\t{behavior_list}\n")
        else:
            f.write('user_id:token\titem_id:token\tbehavior_id:float\n')
            for u in tqdm(tvt_sequence_map):
                behavior = tvt_behavior_map[u][2][0][1]
                target_item = tvt_sequence_map[u][2][0][1]
                f.write(f"{u}\t{target_item}\t{behavior}\n")