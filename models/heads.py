from torch import nn
import torch.nn.functional as F

# projection head for contrast learning
# TODO: projector dimensions
class Projection(nn.Module):
    def __init__(self, dim_in, proj_dim):
        super(Projection, self).__init__()
        # TODO: num_proj_layers as params
        self.net = nn.Sequential(
            nn.Linear(dim_in, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward(self, x):
        features = self.net(x)
        return features

class Predictor(nn.Module):
    def __init__(self, dim_in, dim_mid, dim_out):
        super(Predictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_mid, bias=False),
            nn.BatchNorm1d(dim_mid),
            nn.ReLU(inplace=True),
            nn.Linear(dim_mid, dim_out)
        )

    def forward(self, x):
        features = self.net(x)
        return features
