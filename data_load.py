import json
from torch.utils.data import DataLoader
from encode_dataset import encodeDataset
from tokenizers import Tokenizer

def data_load(max_len,batch_size):
    # Load the JSON dataset
    with open("d:/LLM/train_dataset_translation.json", 'r', encoding='utf-8') as f:
        train_raw_data = json.load(f)
    train_dataset = encodeDataset(train_raw_data, max_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    with open("d:/LLM/valid_dataset_translation.json", 'r', encoding='utf-8') as f:
        valid_raw_data = json.load(f)
    valid_dataset = encodeDataset(valid_raw_data, max_len)
    valid_loader = DataLoader(valid_dataset, batch_size= batch_size, shuffle=True)
    with open("d:/LLM/test_dataset_translation.json", 'r', encoding='utf-8') as f:
        test_raw_data = json.load(f)    
    test_dataset = encodeDataset(test_raw_data, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, valid_loader, test_loader

def max_length(data):
    max_len = 0
    tokenizer_en = Tokenizer.from_file("./tokenizer_en/tokenizer.json")
    tokenizer_ms = Tokenizer.from_file("./tokenizer_ms/tokenizer.json")
    ignore_index = tokenizer_en.token_to_id("[PAD]")
    source_voc_Size = tokenizer_en.get_vocab_size()
    target_voc_size = tokenizer_ms.get_vocab_size()
    for d in data:
        max_len = max(max_len,len(tokenizer_en.encode(d["en"]).ids))
    return max_len,source_voc_Size,target_voc_size, ignore_index