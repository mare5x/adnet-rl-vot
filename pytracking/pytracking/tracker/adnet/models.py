import pathlib
from collections import OrderedDict

import scipy.io
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pytracking.evaluation.environment import env_settings



pretrained_settings = {
    'vggm': {
        'imagenet': {
            'url': 'http://data.lip6.fr/cadene/pretrainedmodels/vggm-786f2434.pth',
            'input_space': 'BGR',
            'input_size': [3, 221, 221],
            'input_range': [0, 255],
            'mean': [123.68, 116.779, 103.939],
            'std': [1, 1, 1],
            'num_classes': 1000
        }
    }
}


class SpatialCrossMapLRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x


class VGGM(nn.Module):

    def __init__(self, num_classes=1000):
        super(VGGM, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, (7, 7), (2, 2)),  # conv1
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(96, 256, (5, 5), (2, 2), (1, 1)),  # conv2
            nn.ReLU(),
            SpatialCrossMapLRN(5, 0.0005, 0.75, 2),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1)),  # conv3
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),  # conv4
            nn.ReLU(),
            nn.Conv2d(512, 512, (3, 3), (1, 1), (1, 1)),  # conv5
            nn.ReLU(),
            nn.MaxPool2d((3, 3), (2, 2), (0, 0), ceil_mode=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(18432, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ADNet(nn.Module):
    # NOTE: must use train() and eval() because of Dropout ...
    
    # First time load the pretrained VGG-M backbone.
    # Otherwise, load adnet_init.pth

    def __init__(self, load_backbone=False, n_actions=11, n_action_history=10):
        super().__init__()

        self.action_history_size = n_actions * n_action_history

        # conv1-3 from VGG-m
        layers = nn.Sequential(OrderedDict([
                ('conv1', nn.Sequential(nn.Conv2d(3, 96, kernel_size=7, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv2', nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                        nn.ReLU(inplace=True),
                                        nn.LocalResponseNorm(2),
                                        nn.MaxPool2d(kernel_size=3, stride=2))),
                ('conv3', nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1),
                                        nn.ReLU(inplace=True))),
                ('fc4',   nn.Sequential(nn.Linear(512 * 3 * 3, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5))),
                ('fc5',   nn.Sequential(nn.Linear(512, 512),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5)))]))
        self.backbone = layers[:3]
        self.fc4_5 = layers[3:]

        # Action probability
        self.fc6 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, n_actions)
            # nn.Softmax(dim=1)
        )

        # Binary confidence
        self.fc7 = nn.Sequential(
            nn.Linear(512 + self.action_history_size, 2)
            # nn.Softmax(dim=1)
        )

        branches = nn.ModuleList([self.fc6, self.fc7])

        # Initialize weights
        for m in layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.1)
        for m in branches.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        if load_backbone:
            self.load_backbone()

    def forward(self, feats, actions, skip_backbone=False, log_softmax=False):
        if skip_backbone:
            out = feats
        else:
            out = self.backbone(feats)
            out = out.view(out.size(0), -1)  # Batch size x Flat features
        out = self.fc4_5(out)
        out = torch.cat((out, actions), dim=1)  # Concatenate actions
        out1 = self.fc6(out)
        out2 = self.fc7(out)

        if log_softmax:
            out1 = F.log_softmax(out1, dim=1)
            out2 = F.log_softmax(out2, dim=1)
        else:
            out1 = F.softmax(out1, dim=1)
            out2 = F.softmax(out2, dim=1)
        return out1, out2

    def extract_features(self, imgs):
        out = self.backbone(imgs)
        out = out.view(out.size(0), -1)  # Batch size x Flat features
        return out

    def load_backbone(self, path=None):
        if path is None:
            env = env_settings()
            path = pathlib.Path(env.network_path) / "imagenet-vgg-m.mat"
            # path = pathlib.Path(env.network_path) / "imagenet-vgg-m-conv1-3.mat"

        print(f"Loading {path.name} ...")

        mat = scipy.io.loadmat(path)
        mat_layers = list(mat['layers'])[0]

        # copy conv weights
        for i in range(3):
            weight, bias = mat_layers[i * 4]['weights'].item()[0]
            self.backbone[i][0].weight.data = torch.from_numpy(np.transpose(weight, (3, 2, 0, 1)))
            self.backbone[i][0].bias.data = torch.from_numpy(bias[:, 0])

    def load_network(self, path, freeze_backbone=True):
        if path.suffix == '.mat':
            self.load_backbone(path)
        else:
            print(f"Loading {path.name} ...")

            state = torch.load(path)
            if 'model' in state:  # If loading a training checkpoint
                state = state['model']
            self.load_state_dict(state)

        self.backbone.requires_grad_(not freeze_backbone)  # Freeze backbone
