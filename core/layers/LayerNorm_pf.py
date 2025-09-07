# import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2D(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        # [N, C, H, W] -> [N, H, W, C] 且连续
        x = x.permute(0, 2, 3, 1).contiguous()
        # 在最后一维做 LayerNorm
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # [N, H, W, C] -> [N, C, H, W] 且连续
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    

class LayerNorm2D_(nn.LayerNorm):
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x):
        return F.layer_norm(
            x.permute(0, 2, 1), self.normalized_shape, self.weight, self.bias, self.eps
        ).permute(0, 1, 2)