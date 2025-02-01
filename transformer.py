import torch.nn as nn
from embedding import EmbeddingLayer
from multi_head import MultiHeadAttention
from pos_embedding import PositionalEncoding
from feed_forward import FeedForward
from project_layer import ProjectionLayer
from encoder import Encoder
from decoder import Decoder
from decoder_block import DecoderBlock
from encoder_block import EncoderBlock
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, d_ff, num_heads, num_blocks, max_len, dropout_rate):
        super(Transformer, self).__init__()
        self.source_embed = EmbeddingLayer(vocab_size, d_model)
        self.target_embed = EmbeddingLayer(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(max_len, d_model, dropout_rate)
        self.multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.masked_multihead_attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.feed_forward = FeedForward(d_model, d_ff, dropout_rate)
        self.encoder_block = EncoderBlock(d_model,self.multihead_attention, self.feed_forward, dropout_rate)
        self.decoder_block = DecoderBlock(d_model,self.masked_multihead_attention, self.multihead_attention, self.feed_forward, dropout_rate)
        self.projection_layer = ProjectionLayer(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        encoderblocklist = []
        decoderblocklist = []
        for _ in range(num_blocks):
            encoderblocklist.append(self.encoder_block)
        for _ in range(num_blocks):
            decoderblocklist.append(self.decoder_block)
        encoderblocklist = nn.ModuleList(encoderblocklist)
        decoderblocklist = nn.ModuleList(decoderblocklist)
        self.encoder = Encoder(d_model,encoderblocklist)
        self.decoder = Decoder(d_model,decoderblocklist)
    def encode(self, encoder_input, encoder_mask):
        encoder_input = self.source_embed(encoder_input)
        encoder_input = self.positional_encoding(encoder_input)
        encoder_output = self.encoder(encoder_input, encoder_mask)
        return encoder_output
    def decode(self, decoder_input, decoder_mask, encoder_output, encoder_mask):
        decoder_input = self.target_embed(decoder_input)
        decoder_input = self.positional_encoding(decoder_input)
        decoder_output = self.decoder(decoder_input, decoder_mask, encoder_output, encoder_mask)
        return decoder_output
    def project(self, decoder_output):
        return self.projection_layer(decoder_output)
    def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
        encoder_output = self.encode(encoder_input, encoder_mask)
        decoder_output = self.decode(decoder_input, decoder_mask, encoder_output, encoder_mask)
        projected_output = self.project(decoder_output)
        return projected_output


        
    # def __init__(self, source_embed: EmbeddingLayer, target_embed: EmbeddingLayer, positional_encoding: PositionalEncoding, multihead_attention: MultiHeadAttention, masked_multihead_attention: MultiHeadAttention, feed_forward: FeedForward, encoder: Encoder, decoder: Decoder, projection_layer: ProjectionLayer, dropout_rate: float):        
        # super(Transformer, self).__init__()
        # Transformer model is initialized with source_embed, target_embed, positional_encoding, multihead_attention, masked_multihead_attention, feed_forward, encoder, decoder, projection_layer and dropout_rate as input parameters. 
        # self.source_embed = source_embed
        # self.target_embed = target_embed
        # self.positional_encoding = positional_encoding
        # self.multihead_attention = multihead_attention        
        # self.masked_multihead_attention = masked_multihead_attention
        # self.feed_forward = feed_forward
        # self.encoder = encoder
        # self.decoder = decoder
        # self.projection_layer = projection_layer
        # self.dropout = nn.Dropout(dropout_rate)
    # Encode function takes in encoder input and encoder mask as input and returns the encoder output.
    # def encode(self, encoder_input, encoder_mask):
    #     encoder_input = self.source_embed(encoder_input)
    #     encoder_input = self.positional_encoding(encoder_input)
    #     encoder_output = self.encoder(encoder_input, encoder_mask)
    #     return encoder_output
    # # Decode function takes in decoder input, decoder mask, encoder output and encoder mask as input and returns the decoder output.
    # def decode(self, decoder_input, decoder_mask, encoder_output, encoder_mask):
    #     decoder_input = self.target_embed(decoder_input)
    #     decoder_input = self.positional_encoding(decoder_input)
    #     decoder_output = self.decoder(decoder_input, decoder_mask, encoder_output, encoder_mask)
    #     return decoder_output
    # # Project function takes in decoder output as input and returns the projected output.
    # def project(self, decoder_output):
    #     return self.projection_layer(decoder_output)
    # # Forward function defines the forward pass of the transformer model.
    # def forward(self, encoder_input, decoder_input, encoder_mask, decoder_mask):
    #     encoder_output = self.encode(encoder_input, encoder_mask)
    #     decoder_output = self.decode(decoder_input, decoder_mask, encoder_output, encoder_mask)
    #     projected_output = self.project(decoder_output)
    #     return projected_output