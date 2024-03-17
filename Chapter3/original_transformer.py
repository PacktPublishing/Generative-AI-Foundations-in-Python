# !pip install transformers tokenizers

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np


# Train the tokenizer to learn the vocabulary
def train_tokenizer(texts):
    # Instantiate the tokenizer
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    # set the pre-tokenizer to whitespace
    tokenizer.pre_tokenizer = Whitespace()
    # train the tokenizer including special tokens
    trainer = WordPieceTrainer(
        vocab_size=5000,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<sos>", "<eos>"],
    )
    tokenizer.train_from_iterator(texts, trainer)

    return tokenizer


# Tensorize the data to prepare for training
def tensorize_data(text_data, tokenizer):
    # token index the data (i.e., numericalize)
    numericalized_data = [
        torch.tensor(tokenizer.encode(text).ids) for text in text_data
    ]
    # pad the sequences so they are all the same length (default is 0)
    padded_data = pad_sequence(numericalized_data, batch_first=True)

    # return shape (batch_size, max_len)
    return padded_data


# Create the dataset domain model
class TextDataset(Dataset):
    def __init__(self, src_data, tgt_data):
        self.src_data = src_data
        self.tgt_data = tgt_data

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]


# Embeddings
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x)


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=None):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0.0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# Multi-Head Attention
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, nhead):
        super(MultiHeadSelfAttention, self).__init__()
        # Instantiate the linear transformation layers for Q, K, and V
        self.attention = nn.MultiheadAttention(d_model, nhead)

    def forward(self, x):
        # Return both the attention output and the attention weights
        return self.attention(x, x, x)


# FFN
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        # Instantiate FFN layers and dropout
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(0.1)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # Apply linear transformation and ReLU non-linearity with dropout
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))


# Encoder Stack
class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super(EncoderLayer, self).__init__()
        # Instantiate the Multi-Head Attention and FFN layers
        self.self_attn = MultiHeadSelfAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, d_ff)
        # Instantiate layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # transpose x to match the shape expected by the self-attention layer
        x = x.transpose(0, 1)
        # Apply the self-attention layer
        attn_output, _ = self.self_attn(x)
        # Apply dropout and layer normalization
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        # Apply the FFN layer
        ff_output = self.feed_forward(x)
        # Apply dropout and layer normalization
        x = x + self.dropout(ff_output)
        # Transpose x back to its original shape
        return self.norm2(x).transpose(0, 1)


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, vocab_size, max_len):
        super(Encoder, self).__init__()
        # Instantiate the Embeddings and Positional Encoding layers
        self.embedding = Embeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(d_model, nhead, d_ff) for _ in range(num_layers)]
        )
        # Define the model hyperparameters
        self.d_model = d_model  # Embedding dimension
        self.nhead = nhead  # Number of attention heads
        # Define the FFN hyperparameters and Instantiate the FFN layer
        self.feed_forward = FeedForward(d_model, d_ff)

    def forward(self, x):
        # Apply the Embeddings and Positional Encoding layers
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x


# Decoder Stack
class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff):
        super(DecoderLayer, self).__init__()
        # Instantiate the Multi-Head Attention and FFN layers
        self.self_attn = MultiHeadSelfAttention(d_model, nhead)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, d_ff)
        # Instantiate layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, memory):
        # Transpose x and memory to match the shape expected by the self-attention layer
        x = x.transpose(0, 1)
        memory = memory.transpose(0, 1)
        # Apply the self-attention layer
        attn_output, _ = self.self_attn(x)
        # Apply dropout and layer normalization
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        attn_output, _ = self.cross_attn(x, memory, memory)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)
        # Apply the FFN layer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        # Transpose x back to its original shape
        return self.norm3(x).transpose(0, 1)


class Decoder(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_layers, vocab_size, max_len):
        super(Decoder, self).__init__()
        # Instantiate the Embeddings and Positional Encoding layers
        self.embedding = Embeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(d_model, nhead, d_ff) for _ in range(num_layers)]
        )
        # Instantiate the linear transformation and softmax function
        self.linear = nn.Linear(d_model, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, memory):
        # Apply the Embeddings and Positional Encoding layers
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.decoder_layers:
            x = layer(x, memory)
        # Apply the linear transformation and softmax function
        x = self.linear(x)
        return self.softmax(x)


# Complete Transformer
class Transformer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        d_ff,
        num_encoder_layers,
        num_decoder_layers,
        src_vocab_size,
        tgt_vocab_size,
        max_len,
    ):
        super(Transformer, self).__init__()
        # Instantiate the Encoder and Decoder
        self.encoder = Encoder(
            d_model, nhead, d_ff, num_encoder_layers, src_vocab_size, max_len=max_len
        )
        self.decoder = Decoder(
            d_model, nhead, d_ff, num_decoder_layers, tgt_vocab_size, max_len=max_len
        )

    def forward(self, src, tgt):
        # Apply the Encoder and Decoder
        memory = self.encoder(src)
        output = self.decoder(tgt, memory)
        return output


def train(model, loss_fn, optimizer, NUM_EPOCHS=10):
    # Iterate through epochs
    for epoch in range(NUM_EPOCHS):
        # Set model to training mode
        model.train()
        total_loss = 0
        for (
            batch
        ) in (
            batch_iterator
        ):  # Assume batch_iterator yields batches of tokenized and numericalized text
            src, tgt = batch
            # Forward pass
            optimizer.zero_grad()
            # Call the model
            output = model(src, tgt)
            # Compute the loss
            loss = loss_fn(output.view(-1, TGT_VOCAB_SIZE), tgt.view(-1))
            # Backward pass
            loss.backward()
            # Update parameters
            optimizer.step()
            # Update total loss
            total_loss += loss.item()

        # Print the loss every epoch
        print(f"Epoch {epoch}, Loss {total_loss / len(batch_iterator)}")


def translate(model, src_text, src_tokenizer, tgt_tokenizer, max_target_length=50):
    # Set model to evaluation mode
    model.eval()

    # Tokenize and numericalize the source text
    src_tokens = src_tokenizer.encode(src_text).ids
    src_tensor = torch.LongTensor(src_tokens).unsqueeze(0)  # Add batch dimension

    # Define the SOS and EOS token indices for the target vocabulary
    tgt_sos_idx = tgt_tokenizer.token_to_id("<sos>")
    tgt_eos_idx = tgt_tokenizer.token_to_id("<eos>")

    # Initialize the target tensor with the SOS token index
    tgt_tensor = torch.LongTensor([tgt_sos_idx]).unsqueeze(0)  # Add batch dimension

    # Loop until the maximum target length is reached or the EOS token is generated
    for i in range(max_target_length):
        # Call the model to generate the output
        with torch.no_grad():  # Disable gradient calculation to save memory during inference
            output = model(src_tensor, tgt_tensor)

        # Retrieve the predicted token
        predicted_token_idx = output.argmax(dim=2)[0, -1].item()
        # Check if the predicted token is the EOS token
        if predicted_token_idx == tgt_eos_idx:
            break
        # Concatenate the predicted token to the target tensor
        tgt_tensor = torch.cat(
            (tgt_tensor, torch.LongTensor([[predicted_token_idx]])), dim=1
        )

    # Convert the target tensor to a list of token indices, decode to tokens, and join to form the translated text
    translated_token_ids = tgt_tensor[0, 1:].tolist()  # Exclude the SOS token
    translated_text = tgt_tokenizer.decode(
        translated_token_ids
    )  # Convert token ids to text

    return translated_text


if __name__ == "__main__":
    from dataclasses import dataclass

    # Instructions:
    # Run the script with the following command: python original_transformer.py
    # Ensure to have the data.csv file in the same directory as this script

    # DEFINE HYPERPARAMETERS
    @dataclass
    class ConfigHyperparams:
        # Number of layers in the encoder and decoder
        NUM_ENCODER_LAYERS = 2
        NUM_DECODER_LAYERS = 2

        # Dropout rate
        DROPOUT_RATE = 0.1

        # Model dimensionality
        EMBEDDING_DIM = 512

        # Number of attention heads
        NHEAD = 8

        # Feed-forward network hidden dimensionality
        FFN_HID_DIM = 2048

        # Batch size
        BATCH_SIZE = 31

        # Learning rate
        LEARNING_RATE = 0.001

        # maximum length of the sequence
        MAX_LEN = 100

        # Number of epochs
        NUM_EPOCHS = 10

        def set_vocab_sizes(self, src_vocab_size, tgt_vocab_size):
            self.SRC_VOCAB_SIZE = src_vocab_size
            self.TGT_VOCAB_SIZE = tgt_vocab_size

    # Instantiate the hyperparameters
    hp = ConfigHyperparams()

    # Load demo data
    data = pd.read_csv("data.csv")

    # Arbitrarily cap at 100 characters for demonstration to avoid long training times
    def demo_limit(vocab, limit=hp.MAX_LEN):
        return [i[:limit] for i in vocab]

    # Separate English and French lexicons
    EN_TEXT = demo_limit(data.en.to_numpy().tolist())
    FR_TEXT = demo_limit(data.fr.to_numpy().tolist())

    # Instantiate the tokenizer
    en_tokenizer = train_tokenizer(EN_TEXT)
    fr_tokenizer = train_tokenizer(FR_TEXT)

    # Establish the vocabulary size
    SRC_VOCAB_SIZE = len(en_tokenizer.get_vocab())
    TGT_VOCAB_SIZE = len(fr_tokenizer.get_vocab())

    hp.set_vocab_sizes(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE)

    # Numericalize and tensorize the data
    # Source tensor with dimensions (batch_size, max_len)
    src_tensor = tensorize_data(EN_TEXT, en_tokenizer)
    # Target tensor with dimensions (batch_size, max_len)
    tgt_tensor = tensorize_data(FR_TEXT, fr_tokenizer)

    # Instantiate the dataset
    dataset = TextDataset(src_tensor, tgt_tensor)

    # Instantiate the model
    model = Transformer(
        hp.EMBEDDING_DIM,
        hp.NHEAD,
        hp.FFN_HID_DIM,
        hp.NUM_ENCODER_LAYERS,
        hp.NUM_DECODER_LAYERS,
        hp.SRC_VOCAB_SIZE,
        hp.TGT_VOCAB_SIZE,
        hp.MAX_LEN,
    )
    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.LEARNING_RATE)

    # Instantiate the batch iterator, dropping the last batch to ensure all batches are the same size
    batch_iterator = DataLoader(
        dataset, batch_size=hp.BATCH_SIZE, shuffle=True, drop_last=True
    )

    # Train the model
    train(model, loss_fn, optimizer, NUM_EPOCHS=hp.NUM_EPOCHS)

    # Translate a sample sentence
    src_text = "hello, how are you?"
    translated_text = translate(model, src_text, en_tokenizer, fr_tokenizer)
    print("Source text:", src_text)
    print("Translated text:", translated_text)
