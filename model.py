import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class GroupNorm32(torch.nn.GroupNorm):
    def __init__(self, num_channels, num_groups=32, **kargs):
        super().__init__(num_groups, num_channels, **kargs)


class ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, small_kernel=True, backbone='resnet18', args=None):
        super(ResNet, self).__init__()

        # Load the pretrained ResNet model
        if args.norm_type == 'bn':
            resnet_model = models.__dict__[backbone](pretrained=pretrained)
        else:
            resnet_model = models.__dict__[backbone](pretrained=pretrained, norm_layer=GroupNorm32)

        if small_kernel:
            conv1_out_ch = resnet_model.conv1.out_channels
            if args.dset in ['fmnist']:
                resnet_model.conv1 = nn.Conv2d(1, conv1_out_ch, kernel_size=3, stride=1, padding=1, bias=False)  # Small dataset filter size used by He et al. (2015)
            else:
                resnet_model.conv1 = nn.Conv2d(3, conv1_out_ch, kernel_size=3, stride=1, padding=1, bias=False)  # Small dataset filter size used by He et al. (2015)
        resnet_model.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)

        # Isolate the feature extraction layers
        self.features = nn.Sequential(*list(resnet_model.children())[:-1])

        # Isolate the classifier layer
        self.classifier = nn.Linear(resnet_model.fc.in_features, num_classes)

        if args.ETF_fc:
            weight = torch.sqrt(torch.tensor(num_classes / (num_classes - 1))) * (
                    torch.eye(num_classes) - (1 / num_classes) * torch.ones((num_classes, num_classes)))
            weight /= torch.sqrt((1 / num_classes * torch.norm(weight, 'fro') ** 2))

            self.classifier.weight = nn.Parameter(torch.mm(weight, torch.eye(num_classes, resnet_model.fc.in_features)))
            self.classifier.weight.requires_grad_(False)

        if args.ckpt not in ['', 'null', 'none']:
            pretrain_wt = torch.load(args.ckpt)
            if args.load_fc:  # load both feature extractor and fc
                pass
            else:             # not load fc
                pretrain_wt = {k: v for k, v in pretrain_wt.items() if 'classifier' not in k}
            self.load_state_dict(pretrain_wt, strict=False)

    def forward(self, x, ret_feat=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        if ret_feat:
            return out, x
        else:
            return out


class MLP(nn.Module):
    def __init__(self, hidden, depth=6, fc_bias=True, num_classes=10):
        # Depth means how many layers before final linear layer

        super(MLP, self).__init__()
        layers = [nn.Linear(3072, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]
        for i in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(num_features=hidden), nn.ReLU()]

        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, num_classes, bias=fc_bias)
        print(fc_bias)

    def forward(self, x, ret_feat=False):
        x = x.view(x.shape[0], -1)
        x = self.layers(x)
        features = F.normalize(x)
        x = self.fc(x)
        if ret_feat:
            return x, features
        else:
            return x
