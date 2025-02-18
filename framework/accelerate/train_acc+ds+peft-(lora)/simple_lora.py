import torch 
from torch import nn
import math 
import torch.nn.functional as F

LORA_ALPHA = 1    # lora的a权重
LORA_R = 8    # lora的秩
filter_names = ['to_q', 'to_k', 'to_v']  # 注入lora层的名称


# Lora实现，封装linear，替换到父module里
class LoraLayer(nn.Module):
    def __init__(self, raw_linear, in_features, out_features, r, alpha):
        super().__init__()
        self.r = r 
        self.alpha = alpha

        self.lora_a = nn.Parameter(torch.empty((in_features, r)))
        self.lora_b = nn.Parameter(torch.zeros((r, out_features)))
    
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.raw_linear = raw_linear
    
    def forward(self, x):    # x:(batch_size,in_features)
        raw_output = self.raw_linear(x)   
        lora_output = x@((self.lora_a@self.lora_b)*self.alpha/self.r)    # matmul(x,matmul(lora_a,lora_b)*alpha/r)
        return raw_output+lora_output


def inject_lora(model, name, layer):
    name_cols = name.split('.')

    # 逐层下探到linear归属的module
    children = name_cols[:-1]
    cur_layer = model 
    for child in children:
        cur_layer = getattr(cur_layer, child)
    
    #print(layer==getattr(cur_layer,name_cols[-1]))
    lora_layer = LoraLayer(layer, layer.in_features,layer.out_features, LORA_R, LORA_ALPHA)
    setattr(cur_layer, name_cols[-1], lora_layer)


def add_lora(model):
     # 向nn.Linear层注入Lora
    for name, layer in model.named_modules():
        name_cols=name.split('.')
        # 过滤出cross attention使用的linear权重
        if any(n in name_cols for n in filter_names) and isinstance(layer, nn.Linear):
            inject_lora(model, name, layer)

    # 冻结非Lora参数
    for name, param in model.named_parameters():
        if name.split('.')[-1] not in ['lora_a', 'lora_b']:  # 非Lora部分不计算梯度
            param.requires_grad=False
        else:
            param.requires_grad=True




# 512x512训练--unet支持64*64输入 原unet输入32*32
from latentsync.models.resnet import InflatedConv3d, InflatedGroupNorm, Upsample3D, Downsample3D

class DownUpSampleWrapper(nn.Module):
    def __init__(self, model, unet_in_channels=13, unet_out_channels=4):
        super().__init__()
        self.model = model
        self.downsample = Downsample3D(unet_in_channels, use_conv=True, out_channels=unet_in_channels, padding=1, name="op")  # 2x 下采样
        self.upsample = Upsample3D(unet_out_channels, use_conv=True, out_channels=unet_out_channels)  # 2x 上采样

    def forward(self, x, timesteps, audio_embeds):
        x = self.downsample(x)  # 下采样
        x = self.model(x, timesteps, audio_embeds).sample       # 经过原始模型
        x = self.upsample(x)    # 上采样
        return x


