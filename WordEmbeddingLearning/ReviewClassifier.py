
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