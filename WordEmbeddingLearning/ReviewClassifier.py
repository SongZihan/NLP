
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from rich.progress import Progress


class ReviewClassifier(nn.Module):
    """ a simple perceptron based classifier """

    def __init__(self, num_features):
        """
        Args:
            num_features (int): the size of the input feature vector
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(num_features,1024)
        self.activation1 = nn.ReLU()

        self.fc2 = nn.Linear(1024,512)
        self.activation2 = nn.ReLU()

        self.fc3 = nn.Linear(512,256)
        self.activation3 = nn.ReLU()

        self.fc4 =  nn.Linear(256,1)
    def forward(self, x_in, apply_sigmoid=True):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, num_features)
            apply_sigmoid (bool): a flag for the sigmoid activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        """
        y1 = self.activation1(self.fc1(x_in))
        y2 = self.activation2(self.fc2(y1))
        y3 = self.activation3(self.fc3(y2))

        y_out = self.fc4(y3).squeeze()

        if apply_sigmoid:
            y_out = torch.sigmoid(y_out)
        return y_out


class ReviewClassifierRNN(nn.Module):
    """ RNN-based classifier """

    def __init__(self, num_features, hidden_dim, n_layers,device):
        """
        Args:
            num_features (int): the size of the input feature vector per time step.
            hidden_dim (int): number of features in the hidden state of the RNN.
            n_layers (int): number of stacked RNN layers.
        """
        super(ReviewClassifierRNN, self).__init__()

        self.device =device
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # RNN Layer
        self.rnn = nn.RNN(num_features, hidden_dim, n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_in):
        """The forward pass of the classifier

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, seq_length, num_features)

        Returns:
            the resulting tensor. tensor.shape should be (batch,)
        """
        batch_size = x_in.size(0)

        # Initializing hidden state for first input using method defined below
        hidden = self.init_hidden(batch_size)

        # Passing in the input and hidden state into the model and obtaining outputs
        out, hidden = self.rnn(x_in, hidden)

        # Fully connected layer
        out = self.fc(out[:, -1, :]).squeeze()  # get the output of the last time step and pass through the fc layer

        return torch.sigmoid(out)

    def init_hidden(self, batch_size):
        # Generates the first hidden state of zeros which we'll use in the forward pass
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_dim).to(self.device)
        return hidden