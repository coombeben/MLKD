"""
Modified version of the timm ResNet
Manual modifications have been indicated with comments
"""
import re
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from timm.layers import StdConv2d, GroupNormAct, get_norm_act_layer, get_act_layer, make_divisible, ClassifierHead
from timm.models import named_apply
from timm.models.resnetv2 import (is_stem_deep, PreActBottleneck, ResNetStage, _init_weights, _load_weights,
                                  create_resnetv2_stem)

__all__ = ['MultiResNetV2']


class BranchBottleneck(nn.Module):
    """Modified bottleneck block from https://github.com/luanyunteng/pytorch-be-your-own-teacher"""
    def __init__(self, channel_in: int, channel_out: int, kernel_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        middle_channel = channel_out // 4
        stride = kernel_size

        self.op = nn.Sequential(
            nn.Conv2d(channel_in, middle_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),

            nn.Conv2d(middle_channel, middle_channel, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm2d(middle_channel),
            nn.ReLU(),

            nn.Conv2d(middle_channel, channel_out, kernel_size=1, stride=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU()
        )

    def forward(self, x):
        return self.op(x)


class MultiResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""
    def __init__(
            self,
            layers=None,
            channels=(256, 512, 1024, 2048),
            num_classes=1000,
            in_chans=3,
            global_pool='avg',
            output_stride=32,
            width_factor=1,
            stem_chs=64,
            stem_type='fixed',
            avg_down=False,
            preact=True,
            act_layer=nn.ReLU,
            norm_layer=partial(GroupNormAct, num_groups=32),
            conv_layer=partial(StdConv2d, eps=1e-8),
            drop_rate=0.,
            drop_path_rate=0.,
            zero_init_last=False,
    ):
        """
        Args:
            layers (List[int]) : number of layers in each block
            channels (List[int]) : number of channels in each block:
            num_classes (int): number of classification classes (default 1000)
            in_chans (int): number of input (color) channels. (default 3)
            global_pool (str): Global pooling type. One of 'avg', 'max', 'avgmax', 'catavgmax' (default 'avg')
            output_stride (int): output stride of the network, 32, 16, or 8. (default 32)
            width_factor (int): channel (width) multiplication factor
            stem_chs (int): stem width (default: 64)
            stem_type (str): stem type (default: '' == 7x7)
            avg_down (bool): average pooling in residual downsampling (default: False)
            preact (bool): pre-activiation (default: True)
            act_layer (Union[str, nn.Module]): activation layer
            norm_layer (Union[str, nn.Module]): normalization layer
            conv_layer (nn.Module): convolution module
            drop_rate: classifier dropout rate (default: 0.)
            drop_path_rate: stochastic depth rate (default: 0.)
            zero_init_last: zero-init last weight in residual path (default: False)
        """
        # Original timm code
        super().__init__()
        if layers is None:
            layers = [3, 4, 6, 3]
        self.num_classes = num_classes
        self.drop_rate = drop_rate

        wf = width_factor
        block_expansion = 4
        norm_layer = get_norm_act_layer(norm_layer, act_layer=act_layer)
        act_layer = get_act_layer(act_layer)

        self.feature_info = []
        stem_chs = make_divisible(stem_chs * wf)
        self.stem = create_resnetv2_stem(
            in_chans,
            stem_chs,
            stem_type,
            preact,
            conv_layer=conv_layer,
            norm_layer=norm_layer,
        )
        stem_feat = ('stem.conv3' if is_stem_deep(stem_type) else 'stem.conv')
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=stem_feat))

        prev_chs = stem_chs
        curr_stride = 4
        dilation = 1
        block_dprs = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(layers)).split(layers)]
        block_fn = PreActBottleneck
        stages = []
        for stage_idx, (d, c, bdpr) in enumerate(zip(layers, channels, block_dprs)):
            out_chs = make_divisible(c * wf)
            stride = 1 if stage_idx == 0 else 2
            if curr_stride >= output_stride:
                dilation *= stride
                stride = 1
            stage = ResNetStage(
                prev_chs,
                out_chs,
                stride=stride,
                dilation=dilation,
                depth=d,
                avg_down=avg_down,
                act_layer=act_layer,
                conv_layer=conv_layer,
                norm_layer=norm_layer,
                block_dpr=bdpr,
                block_fn=block_fn,
            )
            prev_chs = out_chs
            curr_stride *= stride
            self.feature_info += [dict(num_chs=prev_chs, reduction=curr_stride, module=f'stages.{stage_idx}')]
            stages.append(stage)

        self.num_features = prev_chs

        # Modifications I made
        self.stages0 = stages[0]
        self.bottleneck0_1 = BranchBottleneck(64 * block_expansion, 512 * block_expansion, kernel_size=8)
        self.avgpool0 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc0 = ClassifierHead(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
            use_conv=True,
        )

        self.stages1 = stages[1]
        self.bottleneck1_1 = BranchBottleneck(128 * block_expansion, 512 * block_expansion, kernel_size=4)
        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc1 = ClassifierHead(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
            use_conv=True,
        )

        self.stages2 = stages[2]
        self.bottleneck2_1 = BranchBottleneck(256 * block_expansion, 512 * block_expansion, kernel_size=2)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.middle_fc2 = ClassifierHead(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
            use_conv=True,
        )

        self.stages3 = stages[3]
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Original timm code
        self.norm = norm_layer(self.num_features)
        self.head = ClassifierHead(
            self.num_features,
            self.num_classes,
            pool_type=global_pool,
            drop_rate=self.drop_rate,
            use_conv=True,
        )

        self.init_weights(zero_init_last=zero_init_last)
        self.grad_checkpointing = False

    @torch.jit.ignore
    def init_weights(self, zero_init_last=True):
        named_apply(partial(_init_weights, zero_init_last=zero_init_last), self)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix='resnet/'):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^stem',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+)\.blocks\.(\d+)', None),
                (r'^norm', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    @torch.jit.ignore
    def load_from_resnet(self, checkpoint_path: str):
        """Loads a checkpoint of a ResNetv2-50 model"""
        state_dict = self.state_dict()

        for key, value in torch.load(checkpoint_path).items():
            # As the stages are no longer sequential, they are now named
            # stage0 instead of stage.0. Use regex to fix this
            new_key = re.sub(r'stages\.(\d)', r'stages\1', key)
            state_dict[new_key] = value

        self.load_state_dict(state_dict)

    @torch.jit.ignore
    def save(self, checkpoint_path: str):
        """Saves the state dict as a ResNetv2-50 model"""
        state_dict = OrderedDict()

        for key, value in self.state_dict().items():
            if not key.startswith(('bottleneck', 'avgpool', 'middle')):
                new_key = re.sub(r'stages(\d)', r'stages.\1', key)
                state_dict[new_key] = value

        torch.save(state_dict, checkpoint_path)

    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)

    def forward_head(self, x, pre_logits: bool = False):
        return self.head(x, pre_logits=pre_logits)

    def forward(self, x):
        """Custom forward method, returns the inner features and fully-connected layers"""
        # Stem
        x = self.stem(x)

        # Stages
        x = self.stages0(x)
        middle_output0 = self.bottleneck0_1(x)
        middle_output0 = self.avgpool0(middle_output0)
        middle_output0 = self.middle_fc0(middle_output0)

        x = self.stages1(x)
        middle_output1 = self.bottleneck1_1(x)
        middle_output1 = self.avgpool1(middle_output1)
        middle_output1 = self.middle_fc1(middle_output1)

        x = self.stages2(x)
        middle_output2 = self.bottleneck2_1(x)
        middle_output2 = self.avgpool2(middle_output2)
        middle_output2 = self.middle_fc2(middle_output2)

        x = self.stages3(x)
        x = self.norm(x)

        # Head
        x = self.forward_head(x)
        return x, middle_output0, middle_output1, middle_output2
