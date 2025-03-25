__author__ = 'chen yang'

import os
import torch
from torch.optim import AdamW
from core.models import predrnn_pf
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

        self.optimizer = AdamW(self.network.parameters(), lr=configs.lr, betas=[0.8, 0.95])

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
        next_frames, loss = self.network(forcings, init_cond, static_inputs, targets)
        loss.backward()
        self.optimizer.step()
        return loss.detach().cpu().numpy()

    def test(self, forcings, init_cond, static_inputs, targets):
        # frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        next_frames, _ = self.network(forcings, init_cond, static_inputs, targets)
        return next_frames.detach().cpu().numpy()