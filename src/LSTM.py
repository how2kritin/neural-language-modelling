from typing import Optional, Literal
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class LSTMLM(nn.Module):
    """
    Ref: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
    """

    def __init__(self, learning_rate: float, n_features: int, hidden_size: int, n_layers: int, vocab_size: int,
                 dropout_rate: float = 0.0, n_epochs: int = 10, enable_norm: bool = False, device: str = "cpu"):
        """
        RNN model implemented using LSTM layers. Used for next word prediction (language modelling).
        Includes a train_model() loop, as well as a function to predict the top k candidates for the next word.

        :param n_features: Number of expected features in the input.
        :param hidden_size: Number of features in the hidden state.
        :param n_layers: Number of LSTM layers.
        :param vocab_size: Size of the vocabulary.
        :param dropout_rate: 0 by default. Set a value between 0 and 1 to enable dropout.
        :param n_epochs: Number of epochs this algorithm is supposed to run for in the training loop.
        :param enable_norm: False by default. Set to True to enable LayerNormalization.
        """
        super(LSTMLM, self).__init__()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate
        self.device = device
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden_size, num_layers=n_layers, batch_first=True,
                            dropout=dropout_rate).to(self.device)
        self.enable_norm = enable_norm
        if self.enable_norm:
            self.norm = nn.LayerNorm(hidden_size).to(self.device)
        self.fc = nn.Linear(hidden_size, vocab_size).to(self.device)
        self.softmax = nn.Softmax(dim=-1).to(self.device)
        self.optimizer_name = optimizer
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1, self.n_features)
        h_initial = torch.zeros(self.n_layers, x.size(0), self.hidden_size, device=self.device)
        c_initial = torch.zeros(self.n_layers, x.size(0), self.hidden_size, device=self.device)
        _, (hidden, _) = self.lstm(x, (h_initial, c_initial))
        output = hidden[-1]
        if self.enable_norm:
            output = self.norm(output)
        logits = self.fc(output)
        probabilities = self.softmax(logits)
        return probabilities

    def train_model(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> None:
        for epoch in range(self.n_epochs):
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.n_epochs}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                _, predicted = outputs.max(dim=1)  # getting IDs of predicted tokens
                train_total += targets.size(0)
                train_correct += (predicted == targets).sum().item()

            train_loss /= len(train_loader)
            train_accuracy = train_correct / train_total

            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                        outputs = self(inputs)

                        val_loss += self.loss_fn(outputs, targets).item()
                        _, predicted = outputs.max(dim=1)
                        val_total += targets.size(0)
                        val_correct += (predicted == targets).sum().item()

                val_loss /= len(val_loader)
                val_accuracy = val_correct / val_total

                print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
            else:
                print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')

    def predict_top_k(self, x: torch.Tensor, k: int) -> list:
        """
        To predict the top-k most probable candidates for the next word, given a context.
        :param x: Input tensor (batch_size, sequence_length, embedding_dim).
        :param k: Number of top candidates to return.
        :return: List of top-k word indices and their probabilities.
        """
        self.eval()
        with torch.no_grad():
            probabilities = self(x)
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=-1)
            return [(indices.tolist(), probs.tolist()) for indices, probs in zip(top_k_indices, top_k_probs)]
