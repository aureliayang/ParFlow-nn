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
        self.MSE_criterion = nn.MSELoss().cuda()

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

        # self.embed_dim = num_hidden[num_layers - 1]*configs.patch_size*configs.patch_size
        # self.attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads=8,
        #                                        batch_first=True, dropout=0.1)

        self.embed_dim = configs.img_channel*configs.patch_size*configs.patch_size
        self.attention = nn.MultiheadAttention(embed_dim = self.embed_dim, num_heads=8,
                                               batch_first=True, dropout=0.1)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, forcings, net, net_temp, h_t, c_t, memory, delta_c_list, delta_m_list):

        # batch, timesteps, channels, height, width = forcings.shape
        batch, channels, height, width = forcings.shape

        # next_frames = []
        # h_t = []
        # c_t = []
        # delta_c_list = []
        # delta_m_list = []
        decouple_loss = []

        # for i in range(self.num_layers):
        #     zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
        #     h_t.append(zeros)
        #     c_t.append(zeros)
        #     delta_c_list.append(zeros)
        #     delta_m_list.append(zeros)
        
        # memory = self.memory_encoder(init_cond[:, 0])
        # c_t = list(torch.split(self.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))
        
        # net = init_cond[:, 0]
        # net_temp = []
        # for t in range(timesteps):
        # action = forcings[:, t]
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
 
        # output = torch.reshape(h_t[self.num_layers - 1],(batch,1,self.embed_dim))
        # net_temp += [output]
        # net_cat = torch.cat(net_temp,1)
        # attn_output, attn_weights = self.attention(output, net_cat, net_cat)
        # attn_output = torch.reshape(attn_output,(batch, self.num_hidden[self.num_layers - 1], height, width))
        # net = self.scale * self.conv_last(attn_output) + net

        output = self.conv_last(h_t[self.num_layers - 1])
        output = torch.reshape(output,(batch,1,self.embed_dim))
        net_temp += [output]
        net_cat = torch.cat(net_temp,1)
        attn_output, attn_weights = self.attention(output, net_cat, net_cat)
        attn_output = torch.reshape(attn_output,(batch, self.configs.img_channel, height, width))
        net = self.scale * attn_output + net

            # # net = self.conv_last(h_t[self.num_layers - 1]) + net
            # next_frames.append(net)

        # decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        # next_frames = torch.stack(next_frames, dim=1)
        # loss = self.MSE_criterion(next_frames, targets) + self.beta * decouple_loss

        #return next_frames, loss
        return net, net_temp, decouple_loss, h_t, c_t, memory, delta_c_list, delta_m_list
