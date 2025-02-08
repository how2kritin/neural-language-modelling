from typing import Literal, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class RNNLM(nn.Module):
    """
    Recurrent Neural Network Language Model for next word prediction.
    Does not require a context window for next word prediction.

    RNN Ref: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
    LSTM Ref: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    GRU Ref: https://pytorch.org/docs/stable/generated/torch.nn.GRU.html
    """

    def __init__(self, learning_rate: float, vocab: dict, hidden_size: int, n_layers: int,
                 pretrained_embeds: torch.Tensor, rnn_type: Literal['rnn', 'lstm', 'gru'], dropout_rate: float = 0.0,
                 n_epochs: int = 10, patience: int = 3, device: str = "cpu"):
        """
        RNN model with the ability to choose either RNN/LSTM/GRU layers. Used for next word prediction (language modeling).
        Includes a train_model() loop, as well as a function to predict the top k candidates for the next word.

        :param learning_rate: Learning rate of the model to start training with.
        :param vocab: The vocabulary.
        :param hidden_size: Number of features in the hidden state.
        :param n_layers: Number of RNN/LSTM/GRU layers.
        :param pretrained_embeds: Pretrained embedding tensor.
        :param rnn_type: Type of recurrent layer to use ('rnn', 'lstm', or 'gru').
        :param dropout_rate: 0 by default. Set a value between 0 and 1 to enable dropout.
        :param n_epochs: Number of epochs this algorithm is supposed to run for in the training loop.
        :param patience: Number of consecutive iterations of validation loss being worse than the best validation loss (achieved during training) before terminating training.
        """
        super(RNNLM, self).__init__()

        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = pretrained_embeds.size(1)
        self.dropout_rate = dropout_rate
        self.device = device
        self.patience = patience
        self.rnn_type = rnn_type.lower()

        rnn_layer = {'rnn': nn.RNN, 'lstm': nn.LSTM, 'gru': nn.GRU}[self.rnn_type]
        self.rnn = rnn_layer(input_size=self.embedding_dim, hidden_size=hidden_size, num_layers=n_layers,
            batch_first=True, dropout=dropout_rate if n_layers > 1 else 0  # dropout between RNN's hidden layers
        )
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embeddings.weight.data.copy_(pretrained_embeds)
        self.dropout = nn.Dropout(dropout_rate)  # to apply dropout after RNN's output (before FC)
        self.emb_dropout = nn.Dropout(dropout_rate * 0.5)  # dropout after embeddings
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01
            # L2 regularization
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'], reduction='none')

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)  # init hidden states

        if self.rnn_type == 'lstm':  # as LSTM needs both hidden state and cell state
            c0 = torch.zeros(self.n_layers, batch_size, self.hidden_size).to(self.device)
            hidden = (h0, c0)
        else:  # as RNN and GRU only need hidden state
            hidden = h0

        embeds = self.embeddings(x)
        embeds = self.emb_dropout(embeds)
        rnn_out, _ = self.rnn(embeds, hidden)  # shape: (batch_size, seq_length, hidden_dim)
        rnn_out = self.layer_norm(rnn_out)
        rnn_out = self.dropout(rnn_out)
        logits = self.fc(rnn_out)
        return logits

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        """
        Training loop for the model.

        :param train_loader: DataLoader for training data
        :param val_loader: Optional DataLoader for validation data
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.train()
            train_total_loss = 0
            train_total_tokens = 0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.n_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                mask = (targets.view(-1) != self.vocab['<PAD>']).float()

                num_valid_tokens = mask.sum()
                masked_loss = (loss * mask).sum()
                mean_loss = masked_loss / num_valid_tokens

                mean_loss.backward()
                self.optimizer.step()

                train_total_loss += masked_loss.item()
                train_total_tokens += num_valid_tokens.item()

            avg_train_loss = train_total_loss / train_total_tokens
            train_perplexity = torch.exp(torch.tensor(avg_train_loss))

            if val_loader is not None:
                self.eval()
                val_total_loss = 0
                val_total_tokens = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self(inputs)
                        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                        mask = (targets.view(-1) != self.vocab['<PAD>']).float()
                        num_valid_tokens = mask.sum()
                        masked_loss = (loss * mask).sum()

                        val_total_loss += masked_loss.item()
                        val_total_tokens += num_valid_tokens.item()

                    avg_val_loss = val_total_loss / val_total_tokens
                    val_perplexity = torch.exp(torch.tensor(avg_val_loss))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.state_dict(), 'pretrained_models/best_model.pt')
                else:
                    patience_counter += 1

                print(f'Epoch {epoch + 1}, '
                      f'Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}, '
                      f'Val Loss: {avg_val_loss:.4f}, Val Perplexity: {val_perplexity:.4f}')

                if patience_counter >= self.patience:
                    print(f'Early stopping after {epoch + 1} epochs')
                    break
            else:
                print(f'Epoch {epoch + 1}, '
                      f'Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}')

        torch.save(self.state_dict(), 'pretrained_models/final_model.pt')

    def predict_top_k(self, x: torch.Tensor, k: int) -> list:
        """
        To predict the top-k most probable candidates for the next word, given a context.
        :param x: Input tensor (batch_size, sequence_length) for indices.
        :param k: Number of top candidates to return.
        :return: List of tuples [(indices, probs)] for the batch.
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.softmax(self(x))
            last_position_probs = probabilities[:, -1, :]  # shape: (batch_size, vocab_size)
            top_k_probs, top_k_indices = torch.topk(last_position_probs, k, dim=-1)
            return [(indices.tolist(), probs.tolist()) for indices, probs in zip(top_k_indices, top_k_probs)]
