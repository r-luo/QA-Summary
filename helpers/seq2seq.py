import torch
import torch.nn as nn


class Seq2SeqEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_weights,
        embedding_args={},
        rnn_module='lstm',
        rnn_module_args={},
        batch_first=True,
        embedding_padding_index=0,
    ):
        vocab_size = embedding_weights.shape[0]
        embedding_dim = embedding_weights.shape[1]
        super().__init__()
        if rnn_module == 'lstm':
            self.rnn_module = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=batch_first,
                **rnn_module_args
            )
        elif rnn_module == 'gru':
            self.rnn_module = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=batch_first,
                **rnn_module_args
            )
        else:
            self.rnn_module = rnn_module(**rnn_module_args)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=embedding_padding_index,
            _weight=embedding_weights,
            **embedding_args
        )
        # Fix embedding layer weights
        for param in self.embedding.parameters():
            param.requires_grad = False

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn_module(x, hidden)
        return output, hidden


class Seq2SeqDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        embedding_weights,
        embedding_args={},
        rnn_module='lstm',
        rnn_module_args={},
        batch_first=True,
        embedding_padding_index=0,
    ):
        vocab_size = embedding_weights.shape[0]
        embedding_dim = embedding_weights.shape[1]
        super().__init__()
        if rnn_module == 'lstm':
            self.rnn_module = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=batch_first,
                **rnn_module_args
            )
        elif rnn_module == 'gru':
            self.rnn_module = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                batch_first=batch_first,
                **rnn_module_args
            )
        else:
            self.rnn_module = rnn_module(**rnn_module_args)
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=embedding_padding_index,
            _weight=embedding_weights,
            **embedding_args
        )
        # Fix embedding layer weights
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(
            in_features=hidden_dim * 2 if self.rnn_module.bidirectional else hidden_dim,
            out_features=embedding_weights.shape[0],
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, hidden):
        x = self.embedding(x)
        output, hidden = self.rnn_module(x, hidden)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden

