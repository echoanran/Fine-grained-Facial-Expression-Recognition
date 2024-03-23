import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# pre-trained backbone
import torchvision.models as models

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D

        h_n = self.lin_n(torch.einsum('nkld,kj->njld', h, A))
        h_r = self.lin_r(h[:, :, :-1])
        h_n[:, :, 1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, kernel_size=9):
        super(SpatialAttention, self).__init__()

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        pad = (self.kernel_size - 1) // 2  # Padding on one side for stride 1

        self.grp1_conv1k = nn.Conv2d(self.in_channels,
                                     self.in_channels // 2,
                                     (1, self.kernel_size),
                                     padding=(0, pad))
        self.grp1_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp1_convk1 = nn.Conv2d(self.in_channels // 2,
                                     1, (self.kernel_size, 1),
                                     padding=(pad, 0))
        self.grp1_bn2 = nn.BatchNorm2d(1)

        self.grp2_convk1 = nn.Conv2d(self.in_channels,
                                     self.in_channels // 2,
                                     (self.kernel_size, 1),
                                     padding=(pad, 0))
        self.grp2_bn1 = nn.BatchNorm2d(self.in_channels // 2)
        self.grp2_conv1k = nn.Conv2d(self.in_channels // 2,
                                     1, (1, self.kernel_size),
                                     padding=(0, pad))
        self.grp2_bn2 = nn.BatchNorm2d(1)

    def forward(self, input_):
        # Generate Group 1 Features
        grp1_feats = self.grp1_conv1k(input_)
        grp1_feats = F.relu(self.grp1_bn1(grp1_feats))
        grp1_feats = self.grp1_convk1(grp1_feats)
        grp1_feats = F.relu(self.grp1_bn2(grp1_feats))

        # Generate Group 2 features
        grp2_feats = self.grp2_convk1(input_)
        grp2_feats = F.relu(self.grp2_bn1(grp2_feats))
        grp2_feats = self.grp2_conv1k(grp2_feats)
        grp2_feats = F.relu(self.grp2_bn2(grp2_feats))

        added_feats = torch.sigmoid(torch.add(grp1_feats, grp2_feats))
        added_feats = added_feats.expand_as(input_).clone()

        return added_feats


class Model(nn.Module):
    r"""Baseline

    Args:
        num_class (int): Number of classes for the classification task
        temporal_model (str): choose from 'single', 'lstm' and 'tcn'
        backbone (str): choose from 'simple', 'resnet50', 'resnet101', 'vgg16', 'alexnet'
        hidden_size (int): hidden_size for lstm
        num_layers (int): num_layers for lstm
        num_channels (list): num_channel for tcn
        kernel_size (int): kernel_size for tcn
        batch_norm (bool): for backbone: 'simple' 
        dropout (int): for all the model
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, (T_{in}), C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, num_class)` 
          
    """
    def __init__(self,
                 num_class,
                 backbone='simple',
                 temporal_model='single',
                 hidden_size=256,
                 num_layers=2,
                 num_channels=[512, 256, 256],
                 kernel_size=2,
                 batch_norm=True,
                 dropout=0.3,
                 subject=False,
                 pooling=False,
                 nf=None,
                 n_blocks=6,
                 n_hidden=1,
                 normalize=False,
                 activation='sigmoid',
                 **kwargs):
        super().__init__()
        self.num_class = num_class
        self.backbone = backbone
        self.temporal_model = temporal_model

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.num_channels = num_channels
        self.kernel_size = kernel_size

        self.batch_norm = batch_norm
        self.dropout = dropout

        self.subject = subject
        self.pooling = pooling

        self.nf = nf
        self.n_blocks = n_blocks
        self.n_hidden = n_hidden

        self.normalize = normalize
        self.activation = activation

        if self.backbone == 'resnet34':
            self.encoder = nn.Sequential(
                *list(models.resnet34(pretrained=False).children())
                [:-2],  # [N, 512, image_size // (2^4), _]
            )
            self.output_channel = 512
            self.output_size = 8

        if self.temporal_model == 'single':
            pass
        elif self.temporal_model == 'lstm':
            self.rnn = nn.LSTM(input_size=self.output_channel,
                               hidden_size=self.hidden_size,
                               num_layers=self.num_layers,
                               batch_first=True,
                               dropout=self.dropout)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.pooling == False:
            self.spa_attn = nn.ModuleList([
                SpatialAttention(self.output_channel, kernel_size=3)
                for _ in range(num_class)
            ])
            if self.activation == 'sigmoid':
                self.final = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(
                            self.output_channel * self.output_size *
                            self.output_size, 64),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(64, 1),
                        nn.Sigmoid(),
                    ) for _ in range(num_class)
                ])
            else:
                self.final = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(
                        self.output_channel * self.output_size *
                        self.output_size, 64),
                    nn.LeakyReLU(inplace=True),
                    nn.Linear(64, 1),
                ) for _ in range(num_class)
            ])
        else:
            self.spa_attn = nn.ModuleList([
                SpatialAttention(self.output_channel, kernel_size=3)
                for _ in range(num_class)
            ])
            if self.activation == 'sigmoid':
                self.final = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.output_channel, 64),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(64, 1),
                        nn.Sigmoid(),
                    ) for _ in range(num_class)
                ])
            else:
                self.final = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.output_channel, 64),
                        nn.LeakyReLU(inplace=True),
                        nn.Linear(64, 1),
                    ) for _ in range(num_class)
                ])

    def mixup_data(self, x, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        return mixed_x, lam

    def forward(self, image):
        '''
        image for cnn: [N, C, H, W] if single
                        [N, T, C, H, W] if sequential model (time_model is set)
        '''
        if len(image.shape) < 5:
            N, _, _, _ = image.shape
            T = 1
            x = image
        else:
            N, T, _, _, _ = image.shape
            x = image.view(-1, image.shape[2], image.shape[3], image.shape[4])
        x = self.encoder(x)

        features = []
        feat_w_attns = []
        attn_weights = []
        for idx in range(self.num_class):
            attn_weight = self.spa_attn[idx](x)
            feat_w_attn = torch.mul(x, attn_weight)

            if self.pooling == False:
                feat_w_attn = feat_w_attn.view(feat_w_attn.shape[0], -1)
            else:
                feat_w_attn = self.avgpool(feat_w_attn)
                feat_w_attn = feat_w_attn.view(feat_w_attn.shape[0], -1)

            if self.normalize:
                feat_w_attn = F.normalize(feat_w_attn, p=2, dim=1)

            features.append(feat_w_attn)
            feat_w_attns.append(feat_w_attn)
            attn_weights.append(attn_weight[:, 0, :, :])

        x = torch.stack(feat_w_attns, dim=-1)  # [N*T, D, num_class]
        feature = torch.stack(features, dim=-1)
        attn_weights = torch.stack(attn_weights, dim=-1)

        if self.temporal_model == 'single':
            cls_outputs = []
            for idx in range(self.num_class):
                cls_outputs.append(self.final[idx](x[:, :, idx]))
            output = torch.stack(cls_outputs, dim=-1).squeeze(1)

            # return output, feature, attn_weights
            return output

        elif self.temporal_model == 'lstm':
            _, D, C = x.shape
            x = x.view(N, T, D, C)
            x = x.reshape((N, C, T, D))

            # x: N X K X L X D
            full_shape = x.shape

            # reshape: N*K, L, D
            x = x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
            h, _ = self.rnn(x)

            # resahpe: N, K, L, H
            h = h.reshape(
                (full_shape[0], full_shape[1], h.shape[1], h.shape[2]))

            h = self.gcn(h, self.A)

            # reshappe N*K*L,H
            h = h.reshape((-1, h.shape[3]))
            x = x.reshape((-1, full_shape[3]))

            log_prob = self.nf.log_prob(x, h)
            log_prob = log_prob.reshape([full_shape[0], -1])
            log_prob = log_prob.mean(dim=1)

            cls_outputs = []
            for idx in range(self.num_class):
                cls_outputs.append(self.final[idx](h[:, :, idx]))
            output = torch.stack(cls_outputs, dim=-1).squeeze(1)

            return feature, output, log_prob

    
    def forward_fc(self, x):
        cls_outputs = []
        for idx in range(self.num_class):
            cls_outputs.append(self.final[idx](x[:, :, idx]))
        output = torch.stack(cls_outputs, dim=-1).squeeze(1)

        return output, None, None