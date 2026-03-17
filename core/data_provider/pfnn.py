__author__ = 'gaozhifeng'
import os
import numpy as np
import torch
import logging
from core.utils import preprocess
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, init_cond, static_inputs, forcings, targets, total_seq, configs, 
                 alpha, n_value, theta_r, theta_s, porosity, specific_storage, mask,
                 mode, coords_space=None, num_patch=None):
        self.configs = configs
        self.name = configs.pf_runname
        self.batch_size = configs.batch_size if mode == 'train' else total_seq
        self.init_cond = init_cond
        self.static_inputs = static_inputs
        self.forcings = forcings
        self.targets = targets
        self.total_seq = total_seq
        self.current_p = 0

        self.alpha = alpha
        self.n_value = n_value
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.porosity = porosity
        self.specific_storage = specific_storage
        self.mask = mask

        # 新增：保存 patch 元信息
        self.coords_space = coords_space
        self.num_patch = num_patch

    def total(self):
        return self.total_seq

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            rand_perm = torch.randperm(self.total_seq)
            self.init_cond = self.init_cond[rand_perm]  
            self.static_inputs = self.static_inputs[rand_perm]
            self.forcings = self.forcings[rand_perm]
            self.targets = self.targets[rand_perm]

            self.alpha = self.alpha[rand_perm]
            self.n_value = self.n_value[rand_perm]
            self.theta_r = self.theta_r[rand_perm]
            self.theta_s = self.theta_s[rand_perm]
            self.porosity = self.porosity[rand_perm]
            self.specific_storage = self.specific_storage[rand_perm]
            self.mask = self.mask[rand_perm]
        self.current_p = 0

    def next(self):
        self.current_p += self.batch_size
        if self.no_batch_left():
            return None

    def no_batch_left(self):
        if self.current_p + self.batch_size > self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + \
                ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None

        init_cond_batch = self.init_cond[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        static_inputs_batch = self.static_inputs[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        forcings_batch = self.forcings[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        targets_batch = self.targets[self.current_p:self.current_p + self.batch_size, :, :, :, :]

        alpha_batch = self.alpha[self.current_p:self.current_p + self.batch_size]
        n_batch = self.n_value[self.current_p:self.current_p + self.batch_size]
        theta_r_batch = self.theta_r[self.current_p:self.current_p + self.batch_size]
        theta_s_batch = self.theta_s[self.current_p:self.current_p + self.batch_size]
        porosity_batch = self.porosity[self.current_p:self.current_p + self.batch_size]
        specific_storage_batch = self.specific_storage[self.current_p:self.current_p + self.batch_size]
        mask_batch = self.mask[self.current_p:self.current_p + self.batch_size]

        return (
            forcings_batch.to(self.configs.device),
            init_cond_batch.to(self.configs.device),
            static_inputs_batch.to(self.configs.device),
            targets_batch.to(self.configs.device),
            alpha_batch.to(self.configs.device),
            n_batch.to(self.configs.device),
            theta_r_batch.to(self.configs.device),
            theta_s_batch.to(self.configs.device),
            porosity_batch.to(self.configs.device),
            specific_storage_batch.to(self.configs.device),
            mask_batch.to(self.configs.device),
        )

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_p))
        logger.info("    Batch Size: " + str(self.batch_size))
        logger.info("    total Size: " + str(self.total()))
        # logger.info("    input_length: " + str(self.input_length))
        # logger.info("    Input Data Type: " + str(self.input_data_type))

        
class DataProcess:
    def __init__(self, configs):
        
        self.input_param = configs
        
        # currently, please combine static parameters manually and provide it path
        self.static_inputs_path = os.path.join(
            configs.static_inputs_path, 
            configs.static_inputs_filename
        )
        
        forcings_filename = configs.pf_runname + ".out.evaptrans."
        self.forcings_path = os.path.join(configs.forcings_path, forcings_filename) 

        # unified ParFlow output prefix: xxx.out.<var>...
        self.output_prefix = os.path.join(
            configs.targets_path,
            configs.pf_runname + ".out."
        )

        self.target_mean_list = configs.target_mean
        self.target_std_list = configs.target_std
        self.force_mean_list = configs.force_mean
        self.force_std_list = configs.force_std
        
        # the RNN length
        self.input_length_train = configs.input_length_train
        self.input_length_test = configs.input_length_test
        
        # the origional size of the modeling domain and the subdomain size you want to 
        # insert into the CNN
        self.img_height = configs.img_height
        self.img_width  = configs.img_width
        self.patch_size = configs.patch_size
        
        self.init_cond_channel = configs.init_cond_channel
        self.static_channel = configs.static_channel
        self.act_channel = configs.act_channel
        self.img_channel = configs.img_channel
        
    def load_data(self, mode='train'):
        
        # model is no use, but keep it here and can be removed later
        if mode == 'train':
            start_step = self.input_param.training_start_step
            end_step = self.input_param.training_end_step
            ss_stride = self.input_param.ss_stride_train
            st_stride = self.input_param.st_stride_train
            input_length = self.input_length_train
        else:
            start_step = self.input_param.test_start_step
            end_step = self.input_param.test_end_step
            ss_stride = self.input_param.ss_stride_test
            st_stride = self.input_param.st_stride_test
            input_length = self.input_length_test

        ys = list(range(0, self.img_height - self.patch_size + 1, ss_stride))
        xs = list(range(0, self.img_width - self.patch_size + 1, ss_stride))

        # 补最后一行 patch
        if ys[-1] != self.img_height - self.patch_size:
            ys.append(self.img_height - self.patch_size)

        # 补最后一列 patch
        if xs[-1] != self.img_width - self.patch_size:
            xs.append(self.img_width - self.patch_size)

        coords_space = [(y, x) for y in ys for x in xs]    #y outer, x inner

        coords_time = [
            start_t for start_t in range(start_step, end_step - input_length + 2, st_stride)
        ]

        num_patch, num_seq = len(coords_space), len(coords_time)
        
        init_cond = torch.empty((num_patch*num_seq, 1, self.init_cond_channel, self.patch_size, self.patch_size),
                                 dtype=torch.float)  # np.float32
        
        forcings = torch.empty((num_patch*num_seq, input_length, self.act_channel, self.patch_size,
                                     self.patch_size), dtype=torch.float)  # np.float32
        
        targets = torch.empty((num_patch*num_seq, input_length, self.img_channel, self.patch_size,
                                    self.patch_size), dtype=torch.float)  # np.float32

        # static
        frame_np = read_pfb(get_absolute_path(self.static_inputs_path)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        mean = frame_im.mean(dim=(3,4), keepdim=True)
        std = frame_im.std(dim=(3,4), keepdim=True)+1e-8
        frame_im = (frame_im-mean)/std
        static_inputs_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)
            
        alpha_filename = self.output_prefix + 'alpha.pfb'
        n_filename = self.output_prefix + 'n.pfb'
        theta_s_filename = self.output_prefix + 'ssat.pfb'
        theta_r_filename = self.output_prefix + 'sres.pfb'
        porosity_filename = self.output_prefix + 'porosity.pfb'
        mask_filename = self.output_prefix + 'mask.pfb'
        specific_storage_filename = self.output_prefix + 'specific_storage.pfb' 
            

        # alpha
        frame_np = read_pfb(get_absolute_path(alpha_filename)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        alpha_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

        # n
        frame_np = read_pfb(get_absolute_path(n_filename)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        n_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

        # theta_r
        frame_np = read_pfb(get_absolute_path(theta_r_filename)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        theta_r_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

        # theta_s
        frame_np = read_pfb(get_absolute_path(theta_s_filename)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        theta_s_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

        # porosity
        frame_np = read_pfb(get_absolute_path(porosity_filename)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        porosity_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

        # specific_storage
        frame_np = read_pfb(get_absolute_path(specific_storage_filename)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        specific_storage_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

        # mask
        frame_np = read_pfb(get_absolute_path(mask_filename)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        mask_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

        mean_p = torch.tensor(self.target_mean_list).view(1, 1, -1, 1, 1)
        std_p = torch.tensor(self.target_std_list).view(1, 1, -1, 1, 1)
        mean_a = torch.tensor(self.force_mean_list).view(1, 1, -1, 1, 1)
        std_a = torch.tensor(self.force_std_list).view(1, 1, -1, 1, 1)

        for idx_t, start_t in enumerate(coords_time):
            for i in range(input_length):
                forcings_name = self.forcings_path + str(i+start_t).zfill(5) + ".pfb"
                frame_np = read_pfb(get_absolute_path(forcings_name)).astype(np.float32)
                frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
                frame_im = (frame_im-mean_a)/std_a
                for idx_s, (y, x) in enumerate(coords_space):
                    forcings[idx_t*num_patch+idx_s:idx_t*num_patch+idx_s+1, i:i+1, :, :, :] = \
                        frame_im[:, :, 1:11, y:y+self.patch_size, x:x+self.patch_size]

                targets_name = self.output_prefix + "press." + str(i+start_t).zfill(5) + ".pfb"
                frame_np = read_pfb(get_absolute_path(targets_name)).astype(np.float32)
                frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
                frame_im = (frame_im-mean_p)/std_p
                for idx_s, (y, x) in enumerate(coords_space):
                    targets[idx_t*num_patch+idx_s:idx_t*num_patch+idx_s+1, i:i+1, :, :, :] = \
                        frame_im[:, :, :, y:y+self.patch_size, x:x+self.patch_size]

            init_cond_name = self.output_prefix + "press." + str(start_t-1).zfill(5) + ".pfb"
            frame_np = read_pfb(get_absolute_path(init_cond_name)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            frame_im = (frame_im-mean_p)/std_p
            for idx_s, (y, x) in enumerate(coords_space):
                init_cond[idx_t*num_patch+idx_s:idx_t*num_patch+idx_s+1, 0, :, :, :] = \
                    frame_im[:, :, :, y:y+self.patch_size, x:x+self.patch_size]
                
        static_inputs = static_inputs_temp.repeat(num_seq,1,1,1,1) #.to(self.input_param.device)
        alpha = alpha_temp.repeat(num_seq, 1, 1, 1, 1)
        n_value = n_temp.repeat(num_seq, 1, 1, 1, 1)
        theta_r = theta_r_temp.repeat(num_seq, 1, 1, 1, 1)
        theta_s = theta_s_temp.repeat(num_seq, 1, 1, 1, 1)
        porosity = porosity_temp.repeat(num_seq, 1, 1, 1, 1)
        specific_storage = specific_storage_temp.repeat(num_seq, 1, 1, 1, 1)
        mask = mask_temp.repeat(num_seq, 1, 1, 1, 1)

        # === 统计信息输出（按当前精度FP32） ===
        bytes_per_element = 4
        total_elements = (
            init_cond.numel() +
            static_inputs.numel() +
            forcings.numel() +
            targets.numel() +
            alpha.numel() +
            n_value.numel() +
            theta_r.numel() +
            theta_s.numel() +
            porosity.numel() +
            specific_storage.numel() +
            mask.numel()
        )
        mem_fp32 = total_elements * bytes_per_element / (1024**3)

        print(f"Num patches: {num_patch}")
        print(f"Num sequences: {num_seq}")
        print(f"Total samples: {num_patch*num_seq}")
        print(f"Estimated memory: {mem_fp32:.2f} GB")
                 
        # return init_cond, static_inputs, forcings, targets, num_patch*num_seq
        return (init_cond, static_inputs, forcings, targets,
        num_patch * num_seq, coords_space, num_patch,
        alpha, n_value, theta_r, theta_s, porosity, specific_storage, mask)

    def get_train_input_handle(self):
        init_cond, static_inputs, forcings, targets, total_seq, coords_space, num_patch, \
        alpha, n_value, theta_r, theta_s, porosity, specific_storage, mask \
            = self.load_data(mode='train')
        return InputHandle(init_cond, static_inputs, forcings, targets, total_seq, self.input_param, 
                           alpha, n_value, theta_r, theta_s, porosity, specific_storage, mask,
                           mode='train', coords_space = coords_space, num_patch = num_patch)

    def get_test_input_handle(self):
        init_cond, static_inputs, forcings, targets, total_seq, coords_space, num_patch, \
        alpha, n_value, theta_r, theta_s, porosity, specific_storage, mask \
            = self.load_data(mode='test')
        return InputHandle(init_cond, static_inputs, forcings, targets, total_seq, self.input_param, 
                           alpha, n_value, theta_r, theta_s, porosity, specific_storage, mask,
                           mode='test', coords_space = coords_space, num_patch = num_patch)

