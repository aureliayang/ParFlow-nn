__author__ = 'chen yang'

import torch
import torch.nn as nn
from core.layers.SpatioTemporalLSTMCell_pf import SpatioTemporalLSTMCell
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):

        super(RNN, self).__init__()
        
        self.configs = configs
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.beta = configs.decouple_beta
        
        cell_list = []
        self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = configs.img_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(SpatioTemporalLSTMCell(in_channel, configs.act_channel, 
                             num_hidden[i], configs.filter_size, configs.stride,))
            
        self.cell_list = nn.ModuleList(cell_list)
        
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], configs.img_channel, 
                                   kernel_size=1, bias=False)
        
        self.adapter = nn.Conv2d(num_hidden[num_layers - 1], num_hidden[num_layers - 1],
                                 kernel_size=1, bias=False)
        
        self.memory_encoder = nn.Conv2d(configs.init_cond_channel, num_hidden[0], kernel_size=1, bias=True)
        
        self.cell_encoder = nn.Conv2d(configs.static_channel, sum(num_hidden), kernel_size=1, bias=True)

        if configs.attn_mode == "pool":
            self.pool = nn.AdaptiveAvgPool2d((1, 1))

        elif configs.attn_mode == "xyconv":
            C = self.num_hidden[-1]
            self.xy_reduce = nn.Sequential(
                # nn.Conv2d(C, C, kernel_size=1, bias=False),  # 可选：通道混合，想纯深度卷积可去掉
                nn.Conv2d(C, C, kernel_size=(configs.patch_size, 1), stride=(configs.patch_size, 1),
                        groups=C, bias=False),             # 压 Y -> 1
                nn.Conv2d(C, C, kernel_size=(1, configs.patch_size), stride=(1, configs.patch_size),
                        groups=C, bias=False)              # 压 X -> 1
            )

        if configs.attn_mode != "none":
            self.embed_dim = self.num_hidden[-1]
            self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=8,
                                                batch_first=True, dropout=0.1)
            self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, forcings, net, net_temp, h_t_temp, h_t, c_t, memory, delta_c_list, delta_m_list):

        # batch, timesteps, channels, height, width = forcings.shape
        batch, channels, height, width = forcings.shape

        decouple_loss = []
        
        action = forcings

        h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, action, h_t[0], c_t[0], memory)
        delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
        delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)

        for i in range(1, self.num_layers):
            h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], action, h_t[i], c_t[i], memory)
            delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
            delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
        for i in range(self.num_layers):
            decouple_loss.append(torch.mean(torch.abs(
                         torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))
 
        if self.configs.attn_mode == "pool":
            pooled = self.pool(h_t[self.num_layers - 1])  # (B, C, 1, 1)

        elif self.configs.attn_mode == "xyconv":
            pooled = self.xy_reduce(h_t[self.num_layers - 1])  # (B, C, 1, 1)

        elif self.configs.attn_mode == "none":
            net = self.conv_last(h_t[self.num_layers - 1]) + net
            return net, net_temp, h_t_temp, decouple_loss, h_t, c_t, memory, delta_c_list, delta_m_list

        output = pooled.view(batch, 1, self.embed_dim)    # (B, 1, C)
        net_temp += [output]
        net_cat = torch.cat(net_temp, dim=1)              # (B, T, C)

        h_t_temp += [h_t[self.num_layers - 1]]            # (B, C, H, W)
        h_t_stack = torch.stack(h_t_temp, dim=1)          # (B, T, C, H, W)

        attn_output, attn_weights = self.attention(output, net_cat, net_cat)  # (B, 1, C)
        attn_weights = attn_weights.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # (B, T, 1, 1, 1)

        attn_applied = torch.sum(attn_weights * h_t_stack, dim=1)  # (B, C, H, W)
        net = self.scale * self.conv_last(attn_applied) + net

        return net, net_temp, h_t_temp, decouple_loss, h_t, c_t, memory, delta_c_list, delta_m_list
