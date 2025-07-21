__author__ = 'chen yang'

import os
import torch
from torch.optim import AdamW
from torch.optim import lr_scheduler
from core.models import predrnn_pf
import ctypes
# from core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn_pf': predrnn_pf.RNN,
            # 'predrnn': predrnn.RNN,
            # 'predrnn_v2': predrnn_v2.RNN,
            # 'action_cond_predrnn': action_cond_predrnn.RNN,
            # 'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = AdamW(self.network.parameters(), lr=configs.lr, betas=[0.8, 0.95], weight_decay=1e-2)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

    def train(self, forcings, init_cond, static_inputs, targets):
        # forcings_tensor = torch.FloatTensor(forcings).to(self.configs.device)
        # init_cond_tensor = torch.FloatTensor(init_cond).to(self.configs.device)
        # static_inputs_tensor = torch.FloatTensor(static_inputs).to(self.configs.device)
        # targets_tensor = torch.FloatTensor(targets).to(self.configs.device)
        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()

        batch, timesteps, channels, height, width = forcings.shape

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)

        memory = self.network.memory_encoder(init_cond[:, 0])
        c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

        net = init_cond[:, 0]
        net_temp = []

        for t in range(timesteps):
            net, net_temp, d_loss_step, h_t, c_t, memory, delta_c_list, delta_m_list \
                  = self.network(forcings, init_cond, static_inputs, targets, net, net_temp,
                                 h_t, c_t, memory, delta_c_list, delta_m_list, t)
            next_frames.append(net)
            # decouple_loss.append(torch.mean(torch.stack(d_loss_step)))
            decouple_loss += d_loss_step
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=1)
        loss = self.network.MSE_criterion(next_frames, targets) + self.configs.decouple_beta * decouple_loss

        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        # return loss.detach().cpu().numpy()
        return loss.item()

    def test(self, forcings, init_cond, static_inputs, targets):
        # frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        with torch.no_grad():
            batch, timesteps, channels, height, width = forcings.shape

            next_frames = []
            h_t = []
            c_t = []
            delta_c_list = []
            delta_m_list = []
            # decouple_loss = []

            for i in range(self.num_layers):
                zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
                h_t.append(zeros)
                c_t.append(zeros)
                delta_c_list.append(zeros)
                delta_m_list.append(zeros)

            memory = self.network.memory_encoder(init_cond[:, 0])
            c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

            net = init_cond[:, 0]
            net_temp = []

            # lib = ctypes.CDLL('./libclm_lsm.so')
            lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libclm_lsm.so'))
            lib = ctypes.CDLL(lib_path)
            print("CLM shared library loaded successfully.")

            for t in range(timesteps):

                net, net_temp, _, h_t, c_t, memory, delta_c_list, delta_m_list \
                    = self.network(forcings, init_cond, static_inputs, targets, net, net_temp,
                                    h_t, c_t, memory, delta_c_list, delta_m_list, t)
                next_frames.append(net)
                # decouple_loss.append(d_loss_step)
            # decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
            next_frames = torch.stack(next_frames, dim=1)
            # loss = self.network.MSE_criterion(next_frames, targets) + self.configs.beta * decouple_loss

            # next_frames, _ = self.network(forcings, init_cond, static_inputs, targets)
        # return next_frames.detach().cpu().numpy()
        return next_frames