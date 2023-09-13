import torch
from torch import nn
from torchdistill.models.custom.bottleneck.base import BottleneckBase
from torchdistill.models.custom.bottleneck.processor import get_bottleneck_processor
from torchdistill.models.registry import register_model_class, register_model_func
import custom
from collections import OrderedDict
from torchvision.models import vgg16_bn


@register_model_class
class Bottleneck4VGG(BottleneckBase):
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
        elif bottleneck_ver == 'vgg16':
            modules = [
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(64, bottleneck_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channel, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            ]
        elif bottleneck_ver == 'vgg11_cifar10':
            modules = [
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.Conv2d(256, bottleneck_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channel, 512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512)
            ]
        elif bottleneck_ver == 'vgg19_cifar10':
            modules = [
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.Conv2d(128, bottleneck_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(bottleneck_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(bottleneck_channel, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
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
def custom_vgg16(bottleneck_channel=9, bottleneck_idx=8, bottleneck_ver='vgg16', compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['block3', 'block4', 'avgpool', 'classifier']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])

    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = Bottleneck4VGG(bottleneck_channel, bottleneck_idx, bottleneck_ver, compressor, decompressor)
    org_model = vgg16_bn(**kwargs)
    org_model = custom.vgg_bottleneck.VGG_Layered(org_model, [2, 2, 3, 3, 3])
    return CustomVGG(bottleneck, short_module_names, org_model)

@register_model_func
def custom_vgg11_cifar10(bottleneck_channel=12, bottleneck_idx=7, bottleneck_ver='vgg11_cifar10', compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['block4', 'pool0', 'pool1', 'pool2', 'pool3', 'pool4', 'classifier']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])

    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = Bottleneck4VGG(bottleneck_channel, bottleneck_idx, bottleneck_ver, compressor, decompressor)
    org_model = custom.vgg_cifar10.vgg11_bn(**kwargs)
    return CustomVGG(bottleneck, short_module_names, org_model)

@register_model_func
def custom_vgg19_cifar10(bottleneck_channel=13, bottleneck_idx=7, bottleneck_ver='vgg19_cifar10', compressor=None, decompressor=None,
                     short_module_names=None, **kwargs):
    if short_module_names is None:
        short_module_names = ['block3', 'block4', 'pool0', 'pool1', 'pool2', 'pool3', 'pool4', 'classifier']

    if compressor is not None:
        compressor = get_bottleneck_processor(compressor['name'], **compressor['params'])

    if decompressor is not None:
        decompressor = get_bottleneck_processor(decompressor['name'], **decompressor['params'])

    bottleneck = Bottleneck4VGG(bottleneck_channel, bottleneck_idx, bottleneck_ver, compressor, decompressor)
    org_model = custom.vgg_cifar10.vgg19_bn(**kwargs)
    return CustomVGG(bottleneck, short_module_names, org_model)

class VGG_Layered(nn.Module):
    def __init__(
        self, vgg_orig: nn.Module, n_conv_per_block: list
    ) -> None:
        super().__init__()
        self.n_conv_per_block = n_conv_per_block
        for block_idx in range(len(self.n_conv_per_block)):
            exec(f'self.block{block_idx} = OrderedDict()')
        block_idx = 0
        conv_cnt = 0
        module_idx = 0
        for module in vgg_orig.features.children():
            if isinstance(module, torch.nn.Conv2d):
                conv_cnt += 1
                if conv_cnt > self.n_conv_per_block[block_idx]:
                    block_idx += 1
                    conv_cnt = 1
                    module_idx = 0                    
            exec(f"self.block{block_idx}['{module_idx}'] = module")
            module_idx += 1
        for block_idx in range(len(self.n_conv_per_block)):
            exec(f'self.block{block_idx} = torch.nn.Sequential(self.block{block_idx})')
        for child_name, child in vgg_orig.named_children():
            if child_name != 'features':
                exec(f'self.{child_name} = child')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # for block_idx in range(len(self.n_conv_per_block)):
        #     exec(f'x = self.block{block_idx}(x)')
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
