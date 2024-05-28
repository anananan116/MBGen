import numpy as np
import csv
from tqdm import tqdm
from collections import defaultdict

def defaultdict_int():
    return defaultdict(int)

def defaultdict_defaultdict_int():
    return defaultdict(defaultdict_int)

def defaultdict_defaultdict_defaultdict_int():
    return defaultdict(defaultdict_defaultdict_int)

class ItemTokenizer(object):
    """
    Base class for item tokenizers, which convert item IDs to semantic IDs.
    
    Attributes:
        id_offsets (list): ID offset of each position of the semantic IDs.
    """
    def __init__(self, id_offsets, random_map):
        self.id_offsets = id_offsets
        self.random_map = random_map
    
    def expand_id(self, id):
        """
        Expands a given ID into multiple IDs based on predefined k-values.
        
        Args:
            id (tuple): A tuple representing the IDs to be expanded.
        
        Returns:
            tuple: A tuple containing the expanded IDs.
        """
        return tuple(id[i] + self.id_offsets[i] for i in range(len(id)))
    
    def convert(self, value):
        raise NotImplementedError("Non-MB tokenizer is not implemented")
    
    def convert_mb(self, value, behaviors):
        raise NotImplementedError("MB tokenizer is not implemented")
    
    def __call__(self, value, behaviors = None):
        """
        Convert a given item ID to a tuple of semantic IDs.
        
        Args:
            value (int or str): The value to convert, can be a str representing an int or an int.
            behaviors (int, Optional): The behavior of the interaction as behavior ID. Will try calling the non-MB tokenizer if not provided.
        
        Returns:
            tuple: A tuple containing the expanded IDs.
        """
        if behaviors is not None:
            return self.expand_id(self.convert_mb(self.random_map(value), behaviors))
        return self.expand_id(self.convert(self.random_map(value)))

class KthDecimalTokenizer(ItemTokenizer):
    """
    Item tokenizer that converts item IDs to their representation in a specified base.
    
    Attributes:
        base (int): The base for conversion.
        max_length (int): The maximum length of the output list.
    """
    
    def __init__(self, id_offsets, base, random_map, max_length=3, reverse_bt = False):
        """
        Initializes the converter with a specified base and maximum output length.
        
        Args:
            base (int): The base (k) for the conversion.
            max_length (int): The maximum length of the output list.
            id_offsets (list): ID offset of each position of the semantic IDs.
        """
        super().__init__(self, id_offsets, random_map)
        self.base = base
        self.max_length = max_length
        self.reverse_bt = reverse_bt

    def convert(self, value):
        """
        Converts a given value to its representation in the specified base, truncated or padded to max_length.
        
        Args:
            value (int or str): The value to convert, can be a str representing an int or an int.
        
        Returns:
            List[int]: A list of integers representing the value in the specified base.
        """
        if isinstance(value, str):
            value = int(value)
        
        converted = []
        while value > 0 and len(converted) < self.max_length:
            converted.append(value % self.base)
            value //= self.base    
        converted.reverse()
        converted = [0] * (self.max_length - len(converted)) + converted
        
        return converted

    
class BasekMBConverter(ItemTokenizer):
    """
    Item tokenizer that converts item IDs to their representation in a specified base. This tokenizer is used for MB tokenization, MB token is added before the converted IDs.
    
    Attributes:
        base (int): The base for conversion.
        max_length (int): The maximum length of the output list.
    """
    
    def __init__(self, id_offsets, base, random_map, max_length=3, reverse_bt = False):
        """
        Initializes the converter with a specified base and maximum output length.
        
        Args:
            base (int): The base (k) for the conversion.
            max_length (int): The maximum length of the output list.
            id_offsets (list): ID offset of each position of the semantic IDs.
        """
        super().__init__(id_offsets, random_map)
        self.base = base
        self.max_length = max_length
        self.reverse_bt = reverse_bt

    def convert_mb(self, value, behavior):
        """
        Converts a given value to its representation in the specified base, truncated or padded to max_length, MB token is added before the converted IDs.
        
        Args:
            value (int or str): The value to convert, can be a str representing an int or an int.
        
        Returns:
            List[int]: A list of integers representing the value in the specified base.
        """
        if isinstance(value, str):
            value = int(value)
        
        converted = []
        while value > 0 and len(converted) < self.max_length:
            converted.append(value % self.base)
            value //= self.base
        converted.reverse()
        converted = [0] * (self.max_length - len(converted)) + converted
        if self.reverse_bt:
            converted = converted + [behavior]
        else:
            converted = [behavior] + converted
        return converted

class QAE_Kmeans_item_Tokenizer(ItemTokenizer):
    """
    Item tokenizer that converts item IDs to their representation in a specified base. This tokenizer is used for MB tokenization, MB token is added before the converted IDs.
    
    Attributes:
        base (int): The base for conversion.
        max_length (int): The maximum length of the output list.
    """
    
    def __init__(self, id_offsets, item_map, random_map, reverse_bt = False):
        """
        Initializes the converter with a specified base and maximum output length.
        
        Args:
            base (int): The base (k) for the conversion.
            max_length (int): The maximum length of the output list.
            id_offsets (list): ID offset of each position of the semantic IDs.
        """
        super().__init__(id_offsets, random_map)
        self.semantic_ids = item_map
        self.semantic_id_2_item = defaultdict(defaultdict_defaultdict_int)
        self.item_2_semantic_id = {}
        for i in range(len(self.semantic_ids)):
            id = self.semantic_ids[i]
            id_dict = self.semantic_id_2_item[id[0]][id[1]]
            id_dict[len(id_dict)] = i+1
            self.item_2_semantic_id[i+1] = [*id, len(id_dict)]
        self.reverse_bt = reverse_bt
    
    def convert_mb(self, value, behavior):
        if isinstance(value, str):
            value = int(value)
        converted = self.item_2_semantic_id[value]
        
        if self.reverse_bt:
            return [behavior] + converted
        else:
            return [behavior] + converted
    
    def convert(self, value):
        if isinstance(value, str):
            value = int(value)
        converted = self.item_2_semantic_id[value]
        return converted


class SequenceGenerator(object):
    """
    Generates input sequences for models, including padding and attention masks, based on user and item interactions.
    
    Attributes:
        k_values (List[int]): K values for ID expansion.
        max_sequence_length (int): The maximum length of the input sequence.
    """
    
    def __init__(self, EOS, max_sequence_length, item_2_semantic_id, min_seq_len = 1):
        """
        Initializes the sequence generator with custom k-values for ID expansion and a maximum sequence length.
        
        Args:
            k_values (List[int]): A list of integers representing the K values for ID expansion.
            max_sequence_length (int): The maximum length of the input sequence.
        """
        self.min_sequence_length = min_seq_len + 1
        self.max_sequence_length = max_sequence_length
        self.item_2_semantic_id = item_2_semantic_id
        self.EOS = EOS
        self.BOS = 0
        
    def pad_sequence(self, sequence, length):
        """
        Pads the sequence to a given length, reserving a position for <EOS>.
        
        Args:
            sequence (List[int]): The input sequence.
            length (int): The desired length of the sequence after padding.
        
        Returns:
            List[int]: A list representing the padded sequence.
        """
        if len(sequence) + 1 >= length:
            return [sequence[0]] + sequence[len(sequence) - (length-2):len(sequence)] + [self.EOS]
        return sequence + [self.EOS] + [0] * (length - len(sequence) - 1)

    def pad_sequence_attention(self, sequence, length):
        """
        Generates an attention mask for a padded sequence.
        
        Args:
            sequence (List[int]): The input sequence.
            length (int): The desired length of the sequence after padding.
        
        Returns:
            List[int]: A list representing the attention mask for the padded sequence.
        """
        if len(sequence) + 1 >= length:
            return [sequence[0]] + sequence[len(sequence) - (length-2):len(sequence)] + [1]
        return sequence + [1] + [0] * (length - len(sequence) - 1)

    def generate_input_sequence(self, user_id, user_sequence, behavior_seq = None):
        """
        Generates input IDs, attention mask, and labels for a given user sequence.
        
        Args:
            user_id (int): The user ID.
            user_sequence (List[int]): The sequence of user interactions.
        
        Returns:
            tuple: A tuple of numpy arrays representing input IDs, attention mask, and labels.
        """
        input_ids = [user_id]
        attention_mask = [1]
        labels = []
        for i in range(len(user_sequence)):
            if i == len(user_sequence) - 1:
                if behavior_seq is not None:
                    labels.extend(self.item_2_semantic_id(user_sequence[i], behavior_seq[i]))
                else:
                    labels.extend(self.item_2_semantic_id(user_sequence[i]))
            else:
                if behavior_seq is not None:
                    new_item = self.item_2_semantic_id(user_sequence[i], behavior_seq[i])
                    attention_mask.extend([1]*len(new_item))
                    input_ids.extend(new_item)
                else:
                    new_item = self.item_2_semantic_id(user_sequence[i])
                    attention_mask.extend([1]*len(new_item))
                    input_ids.extend(new_item)
                
        labels = np.array(labels + [self.EOS], dtype=np.int32)
        input_ids = np.array(self.pad_sequence(input_ids, self.max_sequence_length), dtype=np.int32)
        attention_mask = np.array(self.pad_sequence_attention(attention_mask, self.max_sequence_length), dtype=np.int8)
        return input_ids, attention_mask, labels
    
    def generate_training_sequence(self, user_id, user_sequence, behavior_seq = None, behavior_token=True):
        """
        Generates input IDs, attention mask, and labels for a given user sequence. Augment the sequence so it's ready for training.
        
        Args:
            user_id (int): The user ID.
            user_sequence (List[int]): The sequence of user interactions.
        
        Returns:
            tuple: A tuple of numpy arrays representing input IDs, attention mask, and labels.
        """
        train_sequence = []
        train_attention_mask = []
        train_label = []
        if behavior_token:
            for j in range(self.min_sequence_length, len(user_sequence)+1):
                input_ids, attention_mask, labels = self.generate_input_sequence(user_id, user_sequence[:j], behavior_seq[:j])
                train_sequence.append(input_ids)
                train_attention_mask.append(attention_mask)
                train_label.append(labels)
        else:
            for j in range(self.min_sequence_length, len(user_sequence)+1):
                if behavior_seq[j-1] == 1:
                    input_ids, attention_mask, labels = self.generate_input_sequence(user_id, user_sequence[:j])
                    train_sequence.append(input_ids)
                    train_attention_mask.append(attention_mask)
                    train_label.append(labels)
        return train_sequence, train_attention_mask, train_label

def count_lines(file_path):
    """Quickly count the number of lines in a file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for _ in file)

def read_tsv_data(file_path):
    user_ids = []
    sequences = []
    behavior_sequences = []
    total_lines = count_lines(file_path) - 1  # Adjust for header if present

    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        
        # Skip the header or any initial lines if necessary
        next(reader)  # Uncomment or remove based on whether your file has a header
        
        for row in tqdm(reader, total=total_lines, desc="Processing"):
            user_id = int(row[0])
            item_id_list = list(map(int, row[1].split() + [row[2]]))
            behavior_list = list(map(int, map(float, row[3].split())))
            
            user_ids.append(user_id)
            sequences.append(item_id_list)
            behavior_sequences.append(behavior_list)
    return user_ids, sequences, behavior_sequences

class random_map(object):
    def __init__(self, random_dict):
        self.random_dict = random_dict
    def __call__(self, item):
        return self.random_dict[item]

def generate_random_map(file_paths, random_seed = 2024):
    unique_items = set()
    for file_path in file_paths:
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file, delimiter='\t')
            next(reader)
            for row in reader:
                unique_items.update(map(int, row[1].split() + [row[2]]))
    unique_items = list(unique_items)
    np.random.seed(random_seed)
    unique_items_original = unique_items.copy()
    np.random.shuffle(unique_items)
    item_map = {unique_items_original[i]: unique_items[i] for i in range(len(unique_items))}
    print(len(item_map))
    return random_map(item_map)