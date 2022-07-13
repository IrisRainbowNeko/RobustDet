import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DynamicLinear(nn.Module):
    def __init__(self,  in_features: int, out_features: int, bias: bool = True, K: int = 4, init_weight: bool=True) -> None:
        super(DynamicLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K

        self.weight = nn.Parameter(torch.Tensor(K, out_features, in_features), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_features))
        else:
            self.register_parameter('bias', None)

        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, x, softmax_attention):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        batch_size, in_channels = x.size()
        x = x.unsqueeze(0) #靠组卷积实现动态Linear [1, B*1, N]
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, 1, self.in_channels) # [B*OC, 1, N]
        self.aggregate_weight = aggregate_weight
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1) # [B*OC]
            output = F.conv1d(x, weight=aggregate_weight, bias=aggregate_bias, groups=self.groups * batch_size)
        else:
            output = F.conv1d(x, weight=aggregate_weight, bias=None, groups=batch_size)

        output = output.view(batch_size, self.out_channels)
        return output
