__author__ = 'chen yang'

import torch
import torch.nn as nn
from core.layers.LayerNorm_pf import LayerNorm2D


class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, action_channel, num_hidden, filter_size, stride):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 7)
        )
        self.conv_a = nn.Sequential(
            nn.Conv2d(action_channel, num_hidden * 4, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 4)
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 4)
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden * 3)
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, filter_size, stride, self.padding, padding_mode='reflect'),
            LayerNorm2D(num_hidden)
        )

        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x_t, a_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        a_concat = self.conv_a(a_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat * a_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        delta_c = i_t * g_t
        c_new = f_t * c_t + delta_c

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        delta_m = i_t_prime * g_t_prime
        m_new = f_t_prime * m_t + delta_m

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new, delta_c, delta_m
