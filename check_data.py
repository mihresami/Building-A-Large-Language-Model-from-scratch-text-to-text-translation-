from data_load import data_load
import torch
if __name__ == "__main__":
    train_loader, valid_loader, _ = data_load(max_len=1400,batch_size=3)
    for i, data in enumerate(train_loader):
        print(data['encoder_input'].shape, data["encoder_mask"].shape, data['decoder_input'].shape,data["decoder_mask"].shape)
        break
