# stdlib
import warnings
# 3p
import torch
import torch.nn as nn
from torch_points3d.applications.kpconv import KPConv
from torch_points3d.core.common_modules import MLP

warnings.filterwarnings("ignore")


class FMRegNet(nn.Module):
    """Implement Functional map regularizer layer of GeomFNet.
       Take as input computed descriptors and returns functional map matrix."""
    def __init__(self, lambda_=1e-3):
        """Init layer.
        Keyword Arguments:
            lambda_ {float} -- regularization parameter (default: {1e-3})
        """
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, data, features):
        """One pass in Regularizer layer.

        Returns:
            torch.Tensor -- Functional map matrix from source to target. Size: batch_size x neig x neig
        """
        tot = 0
        neig = data.evecs_x.shape[1]
        A_l, B_l, evals_x_l, evals_y_l = [], [], [], []
        for i in range(0, len(data.nv), 2):
            # get x features and spectral
            evals_x_l.append(data.evals_x[neig * i: neig * (i + 1)].unsqueeze(0))
            x = features[tot: tot + data.nv[i]]
            evecs_trans_x = data.evecs_trans_x[tot: tot + data.nv[i]]
            A_l.append((evecs_trans_x.T @ x).unsqueeze(0))
            tot += data.nv[i]
            # get y features and spectral
            evals_y_l.append(data.evals_x[neig * (i + 1): neig * (i + 2)].unsqueeze(0))
            y = features[tot: tot + data.nv[i + 1]]
            evecs_trans_y = data.evecs_trans_x[tot: tot + data.nv[i + 1]]
            B_l.append((evecs_trans_y.T @ y).unsqueeze(0))
            tot += data.nv[i + 1]

        A = torch.cat(A_l, dim=0)
        B = torch.cat(B_l, dim=0)
        evals_x = torch.cat(evals_x_l, dim=0)
        evals_y = torch.cat(evals_y_l, dim=0)

        A_t = A.transpose(1, 2)
        A_A_t = torch.bmm(A, A_t)
        B_A_t = torch.bmm(B, A_t)

        D = torch.repeat_interleave(evals_x.unsqueeze(1), repeats=evals_x.size(1), dim=1)
        D = (D - torch.repeat_interleave(evals_y.unsqueeze(2), repeats=evals_x.size(1), dim=2)) ** 2

        C_i = []
        for i in range(evals_x.size(1)):
            D_i = torch.cat([torch.diag(D[bs, i, :].flatten()).unsqueeze(0) for bs in range(evals_x.size(0))], dim=0)
            C = torch.bmm(torch.inverse(A_A_t + self.lambda_ * D_i), B_A_t[:, i, :].unsqueeze(1).transpose(1, 2))
            C_i.append(C.transpose(1, 2))
        C = torch.cat(C_i, dim=1)

        return C


class FeatureRegressor(torch.nn.Module):
    """ Allows segregated segmentation in case the category of an object is known.
    This is the case in ShapeNet for example.

    Parameters
    ----------
    in_features -
        size of the input channel
    n_feat: number of output features
    """

    def __init__(self, in_features, n_feat, dropout_proba=0.5, bn_momentum=0.1):
        super().__init__()

        up_factor = 3

        self.channel_rasing = MLP(
            [in_features, n_feat * up_factor], bn_momentum=bn_momentum, bias=False
        )
        if dropout_proba:
            self.channel_rasing.add_module("Dropout", torch.nn.Dropout(p=dropout_proba))

        self.final_mlp = MLP([n_feat * up_factor, n_feat], bias=True)

    def forward(self, features, **kwargs):
        assert features.dim() == 2
        features = self.channel_rasing(features)
        features = self.final_mlp(features)

        return features


class KPConvFeatureExtractor(torch.nn.Module):
    def __init__(self, n_feat, in_grid_size):
        super().__init__()

        self.unet = KPConv(
            architecture="unet",
            input_nc=0,
            num_layers=4,
            in_grid_size=in_grid_size
        )
        self.feature_regressor = FeatureRegressor(self.unet.output_nc, n_feat)

    @property
    def conv_type(self):
        """ This is needed by the dataset to infer which batch collate should be used"""
        return self.unet.conv_type

    def forward(self, data):
        # Forward through unet and feature_regressor
        data_features = self.unet(data)
        self.output = self.feature_regressor(data_features.x)

        return self.output

    def get_spatial_ops(self):
        return self.unet.get_spatial_ops()


class GeomFmapNet(nn.Module):
    def __init__(self, n_feat, in_grid_size, lambda_):
        super().__init__()
        self.feature_extractor = KPConvFeatureExtractor(n_feat=n_feat, in_grid_size=in_grid_size)

        self.fmreg_net = FMRegNet(lambda_=lambda_)

    def forward(self, batch):
        features = self.feature_extractor(batch)
        C = self.fmreg_net(batch, features)
        return C
