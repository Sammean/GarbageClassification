import torchvision
from torch import nn
from torch.nn import Linear


class Clean(nn.Module):
    def __init__(self, classes_num, weights=True) -> None:
        super().__init__()
        self.flag = weights
        self.model = torchvision.models.resnet50(weights='ResNet50_Weights.DEFAULT') if weights \
            else torchvision.models.resnet50()
        self.classes_num = classes_num
        self.model.fc.add_module('add_linear', Linear(self.model.fc.in_features, self.classes_num))

    def forward(self, x):
        return self.model(x)
