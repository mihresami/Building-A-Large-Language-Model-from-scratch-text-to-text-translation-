from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path

# def data_load():
#   train_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ms", split="train")
#   test_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ms", split="test")
#   valid_dataset = load_dataset("Helsinki-NLP/opus-100", "en-ms", split="validation")
#   return train_dataset,test_dataset,valid_dataset

def load_local_data():
  dataset_en = []
  dataset_ms = []
  # en_files = Path("./dataset-en").glob("*.txt")
  # ms_files = Path("./dataset-ms").glob("*.txt")
  path_en = [str(file) for file in Path('./dataset-en').glob("**/*.txt")]
  path_ms = [str(file) for file in Path('./dataset-ms').glob('**/*.txt')]
  print(path_en)
  
  for en_file, ms_file in zip(path_en, path_ms):
      with open(en_file, 'r', encoding="utf-8") as f:
          dataset_en.extend(f.read().splitlines())
      with open(ms_file, 'r', encoding="utf-8") as f:
          dataset_ms.extend(f.read().splitlines())
  return dataset_en, dataset_ms

# def data_save():
#   file_count = 1
#   train_dataset, _ , _ = data_load()
#   for data in tqdm(train_dataset['translation']):
#     dataset_en.append(data['en'].replace("\n",""))
#     dataset_ms.append(data['ms'].replace("\n",""))
#     if(len(dataset_en) == 50000):
#       with open(f"./dataset-en/{file_count}.txt", 'w', encoding="utf-8") as f:
#         f.write('\n'.join(dataset_en))
#         dataset_en = []
#       with open(f"./dataset-ms/{file_count}.txt", 'w', encoding="utf-8") as f:
#         f.write('\n'.join(dataset_ms))
#         dataset_ms = []
#         file_count += 1
# def token_save():
#   path_en = [str(file) for file in Path('./dataset-en').glob("**/*.txt")]
#   path_ms = [str(file) for file in Path('./dataset-ms').glob('**/*.txt')]
#   tokenzier_en = Tokenizer(BPE(unk_token="[UNK]"))
#   trainer_en = BpeTrainer(min_frequency=2 ,special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
#   tokenzier_en.pre_tokenizers = Whitespace()
#   tokenzier_en.train(files=path_en, trainer=trainer_en)
#   tokenzier_en.save("./tokenizer_en/tokenizer.json")
#   tokenzier_ms = Tokenizer(BPE(unk_token="[UNK]"))
#   trainer_ms = BpeTrainer(min_frequency=2, special_tokens=["[UNk]", "[CLS]", "[SEP]", "[PAD]", "[MASk]"])
#   tokenzier_ms.pre_tokenizers = Whitespace()
#   tokenzier_ms.train(files=path_ms, trainer=trainer_ms)
#   tokenzier_ms.save("./tokenizer_ms/tokenizer.json")



if __name__ == "__main__":
  dataset_en, dataset_ms = load_local_data()
  # print(dataset_en[:5])
  # print(dataset_ms[:5])