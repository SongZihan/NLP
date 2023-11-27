import torch
import torch.nn as nn

class ReviewClassifierRNN(nn.Module):
    """ RNN-based classifier with Embedding Layer """

    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, device):
        """
        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int): The size of each embedding vector.
            hidden_dim (int): Number of features in the hidden state of the RNN.
            n_layers (int): Number of stacked RNN layers.
            device (torch.device): The device to run the model on.
        """
        super(ReviewClassifierRNN, self).__init__()

        self.device = device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # RNN Layer
        self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_in):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, seq_length)

        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        """
        batch_size = x_in.size(0)

        # Embedding
        x_embedded = self.embedding(x_in)

        # Initializing hidden state for first input
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x_embedded, hidden)

        # Fully connected layer
        out = self.fc(out[:, -1, :]).squeeze()  # get the output of the last time step and pass through the fc layer

        return torch.sigmoid(out)

    def init_hidden(self, batch_size):
        # Generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden