from typing import Any, List, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from collections import OrderedDict
from typing import Optional, Dict

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.resnet import ResNet, resnet50
from torchvision.models.segmentation.fcn import FCNHead


__all__ = [
    "deeplabv3_resnet50",
]

class DeepLabV3_4decoders(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        classifier: nn.Module,
        classifier2: nn.Module,
        classifier3: nn.Module,
        classifier4: nn.Module,
        drop_rate: float = 0
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.classifier2 = classifier2
        self.classifier3 = classifier3
        self.classifier4 = classifier4
        self.drop_rate = drop_rate
        print(f"The drop rate of the model is {self.drop_rate}")


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_shape = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        features = features["out"]
        x = nn.functional.dropout(features, p=self.drop_rate)
        x = self.classifier(x)
        x = F.interpolate(x, size=input_shape, mode='bicubic', align_corners=False)
        x = torch.sigmoid(x)
        result["out"] = x

        x2 = nn.functional.dropout(features, p=self.drop_rate)
        x2 = self.classifier2(x2)
        x2 = F.interpolate(x2, size=input_shape, mode='bicubic', align_corners=False)
        x2 = torch.sigmoid(x2)
        result["out2"] = x2

        x3 = nn.functional.dropout(features, p=self.drop_rate)
        x3 = self.classifier3(x3)
        x3 = F.interpolate(x3, size=input_shape, mode='bicubic', align_corners=False)
        x3 = torch.sigmoid(x3)
        result["out3"] = x3

        x4 = nn.functional.dropout(features, p=self.drop_rate)
        x4 = self.classifier4(x4)
        x4 = F.interpolate(x4, size=input_shape, mode='bicubic', align_corners=False)
        x4 = torch.sigmoid(x4)
        result["out4"] = x4

        return result


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        super().__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super().__init__()
        modules = []
        modules.append(
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU())
        )

        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


def _deeplabv3_resnet(
    backbone: ResNet,
    num_classes: int,
    **kwargs: Any,
) -> DeepLabV3_4decoders:
    return_layers = {"layer4": "out"}
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    classifier = DeepLabHead(2048, num_classes)
    classifier2 = DeepLabHead(2048, num_classes)
    classifier3 = DeepLabHead(2048, num_classes)
    classifier4 = DeepLabHead(2048, num_classes)

    return DeepLabV3_4decoders(backbone, classifier, classifier2, classifier3, classifier4, **kwargs)

def deeplabv3_resnet50(
    num_classes:int = 1,
    **kwargs: Any,
) -> DeepLabV3_4decoders:
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.
    Args:
        weights (DeepLabV3_ResNet50_Weights, optional): The pretrained weights for the model
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int, optional): number of output classes of the model (including the background)
        aux_loss (bool, optional): If True, it uses an auxiliary loss
        weights_backbone (ResNet50_Weights, optional): The pretrained weights for the backbone
    """
    backbone = resnet50(pretrained = True, replace_stride_with_dilation=[False, True, True])
    model = _deeplabv3_resnet(backbone, num_classes, **kwargs)
    return model