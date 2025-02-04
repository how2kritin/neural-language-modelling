from typing import Optional, Literal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import math


class LSTMLM(nn.Module):
    """
    Ref: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    def __init__(self, learning_rate: float, vocab: dict, hidden_size: int, n_layers: int, embedding_dim: int, pretrained_embeds: torch.Tensor,
                 dropout_rate: float = 0.0, n_epochs: int = 10, patience: int = 3, device: str = "cpu"):
        """
        RNN model implemented using LSTM layers. Used for next word prediction (language modelling).
        Includes a train_model() loop, as well as a function to predict the top k candidates for the next word.

        :param learning_rate: Learning rate of the model to start training with.
        :param vocab: The vocabulary.
        :param hidden_size: Number of features in the hidden state.
        :param n_layers: Number of LSTM layers.
        :param embedding_dim: Dimensions of each embedding vector.
        :param dropout_rate: 0 by default. Set a value between 0 and 1 to enable dropout.
        :param n_epochs: Number of epochs this algorithm is supposed to run for in the training loop.
        """
        super(LSTMLM, self).__init__()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = len(vocab)
        self.dropout_rate = dropout_rate
        self.device = device
        self.patience = patience

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=dropout_rate)
        self.embeddings = nn.Embedding(self.vocab_size, embedding_dim)
        self.embeddings.weight.data.copy_(pretrained_embeds)
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_dropout = nn.Dropout(dropout_rate * 0.5)
        self.fc = nn.Linear(hidden_size, self.vocab_size)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01  # L2 regularization
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'])

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeds = self.embeddings(x)
        embeds = self.emb_dropout(embeds)
        lstm_out, _ = self.lstm(embeds)  # shape: (batch_size, seq_length, hidden_dim)
        lstm_out = self.layer_norm(lstm_out)
        lstm_out = self.dropout(lstm_out)
        logits = self.fc(lstm_out)
        return logits

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.train()
            train_total_loss = 0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.n_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                self.optimizer.step()
                train_total_loss += loss.item()

            avg_train_loss = train_total_loss / len(train_loader)
            train_perplexity = torch.exp(torch.tensor(avg_train_loss))

            if val_loader is not None:
                self.eval()
                val_total_loss = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self(inputs)
                        loss = self.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                        val_total_loss += loss.item()

                avg_val_loss = val_total_loss / len(val_loader)
                val_perplexity = torch.exp(torch.tensor(avg_val_loss))

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(self.state_dict(), 'best_model.pt')
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

        torch.save(self.state_dict(), 'final_model.pt')

    def predict_top_k(self, x: torch.Tensor, k: int) -> list:
        """
        To predict the top-k most probable candidates for the next word, given a context.
        :param x: Input tensor (batch_size, sequence_length, embedding_dim).
        :param k: Number of top candidates to return.
        :return: List of top-k word indices and their probabilities.
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.softmax(self(x))
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=-1)
            return [(indices.tolist(), probs.tolist()) for indices, probs in zip(top_k_indices, top_k_probs)]