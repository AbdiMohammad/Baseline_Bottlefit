from torch import nn
from torchdistill.models.custom.bottleneck.base import BottleneckBase
from torchdistill.models.custom.bottleneck.classification.resnet import CustomResNet
from torchdistill.models.custom.bottleneck.processor import get_bottleneck_processor
from torchdistill.models.registry import register_model_class, register_model_func
from .resnet_1d import resnet18


@register_model_class
class Bottleneck4ResNet18_1D(BottleneckBase):
    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, bottleneck_ver='2', compressor=None, decompressor=None):
        if bottleneck_ver == '1':
            modules = [
                nn.Conv1d(2, bottleneck_channel, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(bottleneck_channel, 64, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.AvgPool1d(kernel_size=2, stride=1)
            ]
        elif bottleneck_ver == '2':
            modules = [
                nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.Conv1d(bottleneck_channel, 128, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=2, stride=1, bias=False),
                nn.AvgPool1d(kernel_size=2, stride=1)
            ]
        else:
            modules = [
                nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 512, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.ConvTranspose1d(bottleneck_channel, 512, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.AvgPool1d(kernel_size=2, stride=1)
            ]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)


@register_model_func
def custom_resnet18_1d(bottleneck_channel=12, bottleneck_idx=7, bottleneck_ver='2', compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['layer3', 'layer4', 'avgpool', 'fc']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])

    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = Bottleneck4ResNet18_1D(bottleneck_channel, bottleneck_idx, bottleneck_ver, compressor, decompressor)
    org_model = resnet18(**kwargs)
    return CustomResNet(bottleneck, short_module_names, org_model)
