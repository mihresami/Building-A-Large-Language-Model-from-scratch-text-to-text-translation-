import torch
from torch.utils.data import Dataset
from tokenizers import Tokenizer

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class encodeDataset(Dataset):
  def __init__(self,raw_data,max_length):
    super(encodeDataset,self).__init__()
    self.source_text = [data['en'] for data in raw_data] 
    self.target_text = [data['ms'] for data in raw_data]
    self.max_length = max_length
  def __len__(self):
    return len(self.source_text)
  def __getitem__(self, index):
    source_text = self.source_text[index]
    target_text = self.target_text[index]
    tokenzier_en = Tokenizer.from_file("./tokenizer_en/tokenizer.json")
    tokenzier_ms = Tokenizer.from_file("./tokenizer_ms/tokenizer.json")
    # source_voc_Size = tokenzier_en.get_vocab_size()
    # target_voc_size = tokenzier_ms.get_vocab_size()
    CLS_ID = torch.tensor([tokenzier_ms.token_to_id("[CLS]")],dtype=torch.int64)
    SEP_ID = torch.tensor([tokenzier_ms.token_to_id("[SEP]")],dtype=torch.int64)
    PAD_ID = torch.tensor([tokenzier_ms.token_to_id("[PAD]")],dtype=torch.int64)
    source_text_encoded = torch.tensor(tokenzier_en.encode(source_text).ids,dtype=torch.int64)
    target_text_encoded = torch.tensor(tokenzier_ms.encode(target_text).ids,dtype=torch.int64)
    num_source_padding = self.max_length - len(source_text_encoded) - 2 
    num_target_padding = self.max_length - len(target_text_encoded) - 1 
    encoder_padding = torch.tensor([PAD_ID] * num_source_padding, dtype = torch.int64)
    decoder_padding = torch.tensor([PAD_ID] * num_target_padding, dtype = torch.int64)
    encoder_input = torch.cat([CLS_ID, source_text_encoded, SEP_ID, encoder_padding])
    decoder_input = torch.cat([CLS_ID, target_text_encoded, decoder_padding])
    target_label = torch.cat([target_text_encoded,SEP_ID,decoder_padding])
    encoder_mask = (encoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int()
    decoder_mask = (decoder_input != PAD_ID).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0))
    return {
              'encoder_input': encoder_input,
              'decoder_input': decoder_input,
              'target_label': target_label,
              'encoder_mask': encoder_mask,
              'decoder_mask': decoder_mask,
              'source_text': source_text,
              'target_text': target_text
          }

def causal_mask(size):
  # dimension of causal mask (batch_size, seq_len, seq_len)
  mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
  return mask == 0