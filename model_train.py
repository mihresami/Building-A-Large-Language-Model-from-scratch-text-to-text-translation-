import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from transformer import Transformer
from encode_dataset import encodeDataset
from tokenizers import Tokenizer
from pathlib import Path
import numpy as np
import os
from data_load import max_length
from data_load import data_load
from torch.nn import CrossEntropyLoss
import argparse
import json
import time
from tqdm import tqdm
import logging
from data_tokenizer import load_local_data
from LLM import LL_model

def log_config(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO)
    logging.info("Training started at: {}".format(time.ctime()))
    logging.info("\n")

def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal = 1).type(torch.int)
    return mask == 0

def arg_parse():
    with open("./train_dataset_translation.json", 'r', encoding='utf-8') as f:
        train_raw_data = json.load(f)
        max_len,source_voc_Size,target_voc_size,ignore_index = max_length(train_raw_data)
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=16),
    parser.add_argument("--num_epochs", type=int, default=40),
    parser.add_argument("--max_length", type=int, default=max_len),
    parser.add_argument("--d_model", type=int, default=512),
    parser.add_argument("--d_ff", type=int, default=2048),
    parser.add_argument("--num_heads", type=int, default=8),
    parser.add_argument("--num_encoder_layers", type=int, default=6),
    parser.add_argument("--num_decoder_layer", type=int, default=6),
    parser.add_argument("--dropout_rate", type=float, default=0.1),
    parser.add_argument("--learning_rate", type=float, default=5e-4),
    parser.add_argument("--warmup_steps", type=int, default=4000),
    parser.add_argument("--log_interval", type=int, default=100),
    parser.add_argument("--save_interval", type=int, default=2),
    parser.add_argument("--log_file", type=str, default="./train_log/train.txt"),
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoint"),
    parser.add_argument("--source_vocab_size", type=int, default=source_voc_Size),
    parser.add_argument("--target_vocab_size", type=int, default=target_voc_size),
    parser.add_argument("--ignore_index", type=int, default=ignore_index)
    args = parser.parse_args()
    return args

def main(preload_epcoch=None):
    args = arg_parse()
    log_config(args.log_file)
    tokenizer_ms = Tokenizer.from_file("tokenizer_ms/tokenizer.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, test_loader = data_load(args.max_length, args.batch_size)
    model = LL_model(args.source_vocab_size, args.max_length, args.d_ff, args.d_model, num_heads=args.num_heads, num_blocks=args.num_encoder_layers, dropout_rate=args.dropout_rate)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = LambdaLR(optimizer, lr_lambda=lambda step: (args.d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) / args.warmup_steps ** 1.5))
    criterion = CrossEntropyLoss(ignore_index=args.ignore_index)
    if preload_epcoch:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, "LL_model_{}.pt".format(preload_epcoch))))
    for epoch in tqdm(range(args.num_epochs)):
        model.train()
        total_loss = 0
        for idx, data in enumerate(train_loader):
            encoder_input = data["encoder_input"].to(device)
            decoder_input = data["decoder_input"].to(device)
            target_label = data["target_label"].to(device)
            encoder_mask = data["encoder_mask"].to(device)
            decoder_mask = data["decoder_mask"].to(device)
            optimizer.zero_grad()
            output = model(encoder_input, decoder_input, encoder_mask, decoder_mask)
            if torch.isnan(output).any():
                print("Nan detected in model output")
                break
            loss = criterion(output.view(-1, output.shape[-1]), target_label.view(-1))
            if torch.isnan(loss).any():
                print("Nan detected in loss")
                break

            predictions = torch.argmax(output,dim=-1)
            batch_size = predictions.shape[0]
            total_acc = 0.0
            for i in range(batch_size):
                mask = target_label[i] != args.ignore_index
                if mask.sum().item() > 0:
                    correct = (predictions[i] == target_label[i]) & mask
                    seq_acc = correct.sum().item() / mask.sum().item()
                total_acc += seq_acc
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(),max_norm=1.0)
            optimizer.step()
            scheduler.step()
            # for param in optimizer.param_groups:
            #     print("current learing Rate:", param['lr'])
            total_loss += loss.item()
            batch_acc = total_acc / batch_size if batch_size > 0 else 0.0
            if idx % 10 == 0:
                logging.info("Epoch: {} Index: {} Batch_Loss: {} accuracy: {}".format(epoch +1,idx + 1, loss.item(),batch_acc))
        if epoch + 1 % args.save_interval == 0:
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "LL_model_{}_{}.pt".format(epoch, idx)))
        logging.info("Epoch: {} Total Loss: {}".format(epoch, total_loss))
        model.eval()
        total_loss = 0
        with torch.inference_mode():
            for idx, data in enumerate(valid_loader):
                encoder_input = data["encoder_input"].to(device)
                # decoder_input = data["decoder_input"].to(device)
                target_label = data["target_label"].to(device)
                encoder_mask = data["encoder_mask"].to(device)
                decoder_mask = data["decoder_mask"].to(device)
                source_text = data['source_text']
                target_text = data['target_text']
                encoder_output = model.model.encode(encoder_input,encoder_mask)
                decoder_input = torch.empty(1,1).fill_(tokenizer_ms.token_to_id('[CLS]')).type_as(encoder_input).to(device)
                while True:
                    if decoder_input.size(1) == args.max_length:
                        break
                    decoder_mask = causal_mask(decoder_input.size(1)).type_as(encoder_mask).to(device)
                    decoder_output = model.model.decode(decoder_input,decoder_mask,encoder_output,encoder_mask)
                    projection = model.model.project(decoder_output[:, -1])
                    _, new_token = torch.max(projection, dim=1)
                    new_token = torch.empty(1,1). type_as(encoder_input).fill_(new_token.item()).to(device)
                    decoder_input = torch.cat([decoder_input, new_token], dim=1)
                    if new_token == tokenizer_ms.token_to_id('[SEP]'):
                        break
                decoder_output = decoder_input.sequeeze(0)
                predicted_text = tokenizer_ms.decode(decoder_output.detach().cpu.numpy())
                print(f'SOURCE TEXT": {source_text}')
                print(f'TARGET TEXT": {target_text}')
                print(f'PREDICTED TEXT": {predicted_text}')
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "LL_model_{}.pt".format(epoch)))

if __name__ == "__main__":
    main()
 
    
