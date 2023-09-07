import torch
import torch.nn as nn
import torchvision.models as models


class Detached_ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, small_kernel=True, backbone='resnet18', args=None):
        super(Detached_ResNet, self).__init__()

        # Load the pretrained ResNet model
        resnet_model = models.__dict__[backbone](pretrained=pretrained)
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

    def forward(self, x, ret_feat=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        if ret_feat:
            return out, x
        else:
            return out
