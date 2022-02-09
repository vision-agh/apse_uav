import torch.nn as nn
import torch


class AssociationHead(nn.Module):

    def __init__(self, roi_size, input_depth, embedding_dim=128):

        super().__init__()
        self.embedding_dim = embedding_dim
        self.roi_size = roi_size
        self.input_depth = input_depth
        self.fc = nn.Linear(self.input_depth * self.roi_size * self.roi_size, self.embedding_dim)


    def forward(self, x):

        """
            Args:
            x - torch.Tensor of shape (N, C, roi_size, roi_size)
        """

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        x = nn.functional.normalize(x, p=2, dim=1)

        return x


    def num_flat_features(self, x):

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features