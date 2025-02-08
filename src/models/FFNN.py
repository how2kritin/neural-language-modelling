from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class FFNNLM(nn.Module):
    """
    Feed-Forward Neural Network Language Model for next word prediction.
    Uses a context window of previous words to predict the next word.
    """

    def __init__(self, learning_rate: float, vocab: dict, hidden_sizes: list[int], pretrained_embeds: torch.Tensor,
                 context_size: int = 4, dropout_rate: float = 0.0, n_epochs: int = 10, patience: int = 3,
                 device: str = "cpu"):
        """
        :param learning_rate: Learning rate of the model to start training with
        :param vocab: The vocabulary
        :param hidden_sizes: List of hidden layer sizes (e.g., [512, 256])
        :param pretrained_embeds: Pretrained embedding tensor
        :param context_size: Number of previous words to use as context
        :param dropout_rate: Dropout rate between layers
        :param n_epochs: Number of epochs for training
        :param patience: Early stopping patience
        :param device: Device to run the model on
        """
        super(FFNNLM, self).__init__()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_dim = pretrained_embeds.size(1)
        self.context_size = context_size
        self.dropout_rate = dropout_rate
        self.device = device
        self.patience = patience

        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embeddings.weight.data.copy_(pretrained_embeds)
        self.emb_dropout = nn.Dropout(dropout_rate * 0.5)

        input_size = self.context_size * self.embedding_dim

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [nn.Linear(prev_size, hidden_size), nn.GELU(), nn.Dropout(dropout_rate), nn.LayerNorm(hidden_size)])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, self.vocab_size))

        self.feed_forward = nn.Sequential(*layers)

        for layer in self.feed_forward:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

        self.softmax = nn.Softmax(dim=-1)
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01, eps=1e-8)  # weight_decay for L2 regularization
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>'], reduction='none')

        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape (batch_size, context_size)
        :return: Output logits of shape (batch_size, vocab_size)
        """
        embeds = self.embeddings(x)
        embeds = self.emb_dropout(embeds)

        batch_size = embeds.size(0)
        flattened = embeds.view(batch_size, -1)

        logits = self.feed_forward(flattened)
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
                loss = self.loss_fn(outputs, targets)
                mask = (targets != self.vocab['<PAD>']).float()

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
                        loss = self.loss_fn(outputs, targets)
                        mask = (targets != self.vocab['<PAD>']).float()
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
        :param x: Input tensor (batch_size, context_size)
        :param k: Number of top candidates to return
        :return: List of top-k word indices and their probabilities
        """
        self.eval()
        with torch.no_grad():
            probabilities = self.softmax(self(x))
            top_k_probs, top_k_indices = torch.topk(probabilities, k, dim=-1)
            return [(indices.tolist(), probs.tolist()) for indices, probs in zip(top_k_indices, top_k_probs)]
