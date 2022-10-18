from typing import List
import numpy as np
import os
from datetime import datetime

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

class BiLSTM_Pool_LSTM(nn.Module):
    def __init__(self,
                input_size,
                output_size,
                bilstm_num_layers: int = 2,
                lstm_num_layers: int = 2,
                lstm_hidden_size: int = 100,
                dr_rate: float = 0.5) -> None:
        super().__init__()


        self.input_size = input_size
        self.hidden_size = input_size # Keeping hidden size same at Input size
        self.bilstm_num_layers = bilstm_num_layers
        self.lstm_num_layers = lstm_num_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.output_size = output_size

        # Bi-LSTM
        self.bilstm_cell = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.bilstm_num_layers,
                                   bidirectional=True,
                                   dropout=dr_rate,
                                   batch_first=True)

        # ManyToManyLSTM
        self.seqtoseq_lstm = nn.LSTM(input_size=self.hidden_size * 3,
                                    hidden_size=self.lstm_hidden_size,
                                    num_layers=self.lstm_num_layers,
                                    dropout=dr_rate,
                                    batch_first=True)
        
        # FC Layers
        self.fc1 = nn.Linear(self.lstm_hidden_size, self.lstm_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.lstm_hidden_size, self.output_size)

        self.dropout = nn.Dropout(dr_rate)

    def forward(self, x: torch.tensor, x_lengths: torch.tensor):
        # x.shape [batch_size, seq_len, 400]

        # Passing though BiLSTM
        packed_bilstm_x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        bilstm_out, _ = self.bilstm_cell(packed_bilstm_x) # torch.Size([batch_size, seq_len, 800]
        padded_bilstm_x, lens_unpacked = pad_packed_sequence(bilstm_out, batch_first=True, total_length=max(x_lengths))

        # Pooling and concat
        mean_pool = torch.mean(padded_bilstm_x, dim=1) # torch.Size([batch_size, 800]
        max_pool, _ = torch.max(padded_bilstm_x, dim=1) # torch.Size([batch_size, 800]
        pooling_concat = torch.concat([max_pool, mean_pool], dim=1) # torch.Size([batch_size, 1600]

        x = torch.cat((x,  torch.unsqueeze(max_pool, dim=1).repeat(1, x.shape[1], 1)), dim=-1 ) # torch.Size([batch_size, seq_len, 400+800]
        # x = torch.cat((x,  torch.unsqueeze(mean_pool, dim=1).repeat(1, x.shape[1], 1)), dim=-1 ) # torch.Size([batch_size, seq_len, 400+800]

        # ManyToMany LSTM
        packed_lstm_x = pack_padded_sequence(x, x_lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.seqtoseq_lstm(packed_lstm_x) # torch.Size([batch_size, seq_len, 48]
        padded_lstm_x, lens_unpacked = pad_packed_sequence(lstm_out, batch_first=True, total_length=max(x_lengths))

        # Softmax for logits
        out = self.relu(self.fc1(padded_lstm_x))
        out = self.dropout(out)
        out = self.fc2(out)

        return out

    def log_weights_and_grads(self, writer: SummaryWriter, global_step: None):
        ## Log BiLSTM Grad
        writer.add_histogram("Grads/BiLSTM/weight_hh_l0", self.bilstm_cell.weight_hh_l0.grad, global_step)
        writer.add_histogram("Grads/BiLSTM/weight_hh_l0_reverse", self.bilstm_cell.weight_hh_l0_reverse.grad, global_step)
        writer.add_histogram("Grads/BiLSTM/weight_ih_l0", self.bilstm_cell.weight_ih_l0.grad, global_step)
        writer.add_histogram("Grads/BiLSTM/weight_ih_l0_reverse", self.bilstm_cell.weight_ih_l0_reverse.grad, global_step)
        
        ## Log LSTM Grad
        writer.add_histogram("Grads/LSTM/weight_hh_l0", self.seqtoseq_lstm.weight_hh_l0.grad, global_step)
        writer.add_histogram("Grads/LSTM/weight_ih_l0", self.seqtoseq_lstm.weight_ih_l0.grad, global_step)
        
        # Log FC Weights and Bias
        writer.add_histogram("Grads/fc1/weight", self.fc1.weight.grad, global_step)
        writer.add_histogram("Grads/fc1/bias", self.fc1.bias.grad, global_step)
        
        writer.add_histogram("Grads/fc2/weight", self.fc2.weight.grad, global_step)
        writer.add_histogram("Grads/fc2/bias", self.fc2.bias.grad, global_step)
        
        # Log BiLSTM Weight
        writer.add_histogram("Weights/BiLSTM/weight_hh_l0", self.bilstm_cell.weight_hh_l0, global_step)
        writer.add_histogram("Weights/BiLSTM/weight_hh_l0_reverse", self.bilstm_cell.weight_hh_l0_reverse, global_step)
        writer.add_histogram("Weights/BiLSTM/weight_ih_l0", self.bilstm_cell.weight_ih_l0, global_step)
        writer.add_histogram("Weights/BiLSTM/weight_ih_l0_reverse", self.bilstm_cell.weight_ih_l0_reverse, global_step)

        # Log LSTM Weight
        writer.add_histogram("Weights/LSTM/weight_hh_l0", self.seqtoseq_lstm.weight_hh_l0, global_step)
        writer.add_histogram("Weights/LSTM/weight_ih_l0", self.seqtoseq_lstm.weight_ih_l0, global_step)

        # Log FC Weights and Bias
        writer.add_histogram("Weights/fc1/weight", self.fc1.weight, global_step)
        writer.add_histogram("Weights/fc1/bias", self.fc1.bias, global_step)
        
        # Log FC Weights and Bias
        writer.add_histogram("Weights/fc2/weight", self.fc2.weight, global_step)
        writer.add_histogram("Weights/fc2/bias", self.fc1.bias, global_step)

if __name__ == "__main__":
    INPUT_SIZE = 400
    HIDDEN_SIZE = 400
    NUM_LAYERS = 2
    NUM_CLASSES = 48

    seq1 = torch.tensor(np.random.rand(1, 4, 400)).float()
    # seq2 = np.random.rand(8, 400)

    model = BiLSTM_Pool_LSTM(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES
    )

    # pred = model(seq1, [4])
    summary(model, input_data=[seq1, [4]])