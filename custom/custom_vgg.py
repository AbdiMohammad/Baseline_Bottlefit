from torch import nn
from torchdistill.models.custom.bottleneck.base import BottleneckBase
from torchdistill.models.custom.bottleneck.processor import get_bottleneck_processor
from torchdistill.models.registry import register_model_class, register_model_func
import custom
from collections import OrderedDict


@register_model_class
class Bottleneck4VGG19(BottleneckBase):
    def __init__(self, bottleneck_channel=12, bottleneck_idx=7, bottleneck_ver='2', compressor=None, decompressor=None):
        if bottleneck_ver == '1':
            modules = [
                nn.Conv2d(3, bottleneck_channel, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channel, 64, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=1)
            ]
        elif bottleneck_ver == '2':
            modules = [
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channel, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=1)
            ]
        elif bottleneck_ver == 'cifar10':
            modules = [
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, bottleneck_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(bottleneck_channel),
                nn.Conv2d(bottleneck_channel, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512)
            ]
        else:
            modules = [
                nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 512, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, bottleneck_channel, kernel_size=2, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(bottleneck_channel, 512, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=2, stride=1, bias=False),
                nn.AvgPool2d(kernel_size=2, stride=1)
            ]
        encoder = nn.Sequential(*modules[:bottleneck_idx])
        decoder = nn.Sequential(*modules[bottleneck_idx:])
        super().__init__(encoder=encoder, decoder=decoder, compressor=compressor, decompressor=decompressor)

@register_model_class
class CustomVGG(nn.Sequential):
    def __init__(self, bottleneck, short_module_names, org_vgg):
        module_dict = OrderedDict()
        module_dict['bottleneck'] = bottleneck
        short_module_set = set(short_module_names)
        for child_name, child_module in org_vgg.named_children():
            if child_name in short_module_set:
                if child_name == 'classifier':
                    module_dict['flatten'] = nn.Flatten(1)
                module_dict[child_name] = child_module

        super().__init__(module_dict)

@register_model_func
def custom_cifar10_vgg19(bottleneck_channel=12, bottleneck_idx=7, bottleneck_ver='2', compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['block4', 'pool0', 'pool1', 'pool2', 'pool3', 'pool4', 'classifier']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])

    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = Bottleneck4VGG19(bottleneck_channel, bottleneck_idx, bottleneck_ver, compressor, decompressor)
    org_model = custom.vgg.vgg19_bn(**kwargs)
    return CustomVGG(bottleneck, short_module_names, org_model)
