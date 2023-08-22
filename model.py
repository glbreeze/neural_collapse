import torch.nn as nn
import torchvision.models as models


class Detached_ResNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=10, small_kernel=True, backbone='resnet18'):
        super(Detached_ResNet, self).__init__()

        # Load the pretrained ResNet model
        resnet_model = models.__dict__[backbone](pretrained=pretrained)
        if small_kernel:
            conv1_out_ch = resnet_model.conv1.out_channels
            resnet_model.conv1 = nn.Conv2d(3, conv1_out_ch, kernel_size=3, stride=1, padding=1, bias=False) # Small dataset filter size used by He et al. (2015)

        # Isolate the feature extraction layers
        self.features = nn.Sequential(*list(resnet_model.children())[:-1])

        # Isolate the classifier layer
        self.classifier = nn.Linear(resnet_model.fc.in_features, num_classes)

    def forward(self, x, ret_feat=False):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)

        if ret_feat:
            return out, x
        else:
            return out
