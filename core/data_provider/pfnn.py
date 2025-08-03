__author__ = 'gaozhifeng'
import numpy as np
import os
import torch
import logging
from core.utils import preprocess
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import write_pfb, read_pfb

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, init_cond, static_inputs, forcings, targets, total_seq, configs, mode):
        self.configs = configs
        self.name = configs.pf_runname
        if mode == 'train':
            self.batch_size = configs.batch_size
        else:
            self.batch_size = total_seq
        self.img_width = configs.patch_size
        self.init_cond = init_cond
        self.static_inputs = static_inputs
        self.forcings = forcings
        self.targets = targets
        self.total_seq = total_seq
        self.current_p = 0
        # self.input_length = configs.input_length
        self.init_cond_channel = configs.init_cond_channel
        self.static_channel = configs.static_channel
        self.act_channel = configs.act_channel
        self.img_channel = configs.img_channel

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
            
        return forcings_batch.to(self.configs.device), init_cond_batch.to(self.configs.device), \
                static_inputs_batch.to(self.configs.device), targets_batch.to(self.configs.device)

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
        
        # the root path and the full file name
        # self.init_cond_path = os.path.join(configs.init_cond_path,
        #                                    configs.init_cond_filename)
        # self.init_cond_test_path = os.path.join(configs.init_cond_test_path,
        #                                    configs.init_cond_test_filename)
        
        # currently, please combine static parameters manually and provide it path
        self.static_inputs_path = os.path.join(configs.static_inputs_path, 
                                               configs.static_inputs_filename)
        
        forcings_filename = configs.pf_runname + ".out.evaptrans."
        self.forcings_path = os.path.join(configs.forcings_path, forcings_filename) 
        
        targets_filename = configs.pf_runname + ".out.press."
        self.targets_path = os.path.join(configs.targets_path, targets_filename) 

        # self.target_norm_path = os.path.join(configs.targets_path,configs.target_norm_file)
        # self.force_norm_path = os.path.join(configs.forcings_path,configs.force_norm_file)
        self.target_mean_str = configs.target_mean
        self.target_std_str = configs.target_std
        self.force_mean_str = configs.force_mean
        self.force_std_str = configs.force_std
        
        # the files should be continuous in time
        # self.training_start_step = configs.training_start_step
        self.training_timesteps  = configs.training_end_step - configs.training_start_step + 1
        # self.test_start_step = configs.test_start_step
        self.test_timesteps  = configs.test_end_step - configs.test_start_step + 1
        
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

        # self.ss_stride_train = configs.ss_stride_train
        # self.st_stride_train = configs.st_stride_train
        # self.ss_stride_test = configs.ss_stride_test
        # self.st_stride_test = configs.st_stride_test
        
    def load_data(self, mode='train'):
        
        # model is no use, but keep it here and can be removed later
        if mode == 'train':
            start_step = self.input_param.training_start_step
            # timesteps  = self.training_timesteps
            end_step = self.input_param.training_end_step
            ss_stride = self.input_param.ss_stride_train
            st_stride = self.input_param.st_stride_train
            input_length = self.input_length_train
        else:
            start_step = self.input_param.test_start_step
            # timesteps  = self.test_timesteps
            end_step = self.input_param.test_end_step
            ss_stride = self.input_param.ss_stride_test
            st_stride = self.input_param.st_stride_test
            input_length = self.input_length_test
        
        coords_space = [
            (y, x)
            for y in range(0, self.img_height - self.patch_size + 1, ss_stride)
            for x in range(0, self.img_width - self.patch_size + 1, ss_stride)
        ]

        coords_time = [
            start_t
            for start_t in range(start_step, end_step - input_length + 2, st_stride)
        ]

        num_patch, num_seq = len(coords_space), len(coords_time)
        
        init_cond = torch.empty((num_patch*num_seq, 1, self.init_cond_channel, self.patch_size, self.patch_size),
                                 dtype=torch.float)  # np.float32
        
        static_inputs_temp = torch.empty((num_patch, 1, self.static_channel, self.patch_size, self.patch_size),
                                          dtype=torch.float)  # np.float32
        
        forcings = torch.empty((num_patch*num_seq, input_length, self.act_channel, self.patch_size,
                                     self.patch_size), dtype=torch.float)  # np.float32
        
        targets = torch.empty((num_patch*num_seq, input_length, self.img_channel, self.patch_size,
                                    self.patch_size), dtype=torch.float)  # np.float32

        # static
        static_inputs_name = self.static_inputs_path
        frame_np = read_pfb(get_absolute_path(static_inputs_name)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        mean = frame_im.mean(dim=(3,4), keepdim=True)
        std = frame_im.std(dim=(3,4), keepdim=True)
        frame_im = (frame_im-mean)/std
        for idx_s, (y, x) in enumerate(coords_space):
            static_inputs_temp[idx_s:idx_s+1, :, :, :, :] = \
                frame_im[:, :, :, y:y+self.patch_size, x:x+self.patch_size]

        # frame_np = read_pfb(get_absolute_path(self.target_norm_path)).astype(np.float32)
        # frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        # mean_p = frame_im.mean(dim=(3,4), keepdim=True)
        # std_p = frame_im.std(dim=(3,4), keepdim=True)

        # frame_np = read_pfb(get_absolute_path(self.force_norm_path)).astype(np.float32)
        # frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        # mean_a = frame_im.mean(dim=(3,4), keepdim=True)
        # std_a = frame_im.std(dim=(3,4), keepdim=True)

        # 转成浮点列表
        target_mean_list = [float(x) for x in self.target_mean_str.split(',')]
        target_std_list = [float(x) for x in self.target_std_str.split(',')]
        mean_p = torch.tensor(target_mean_list).view(1, 1, -1, 1, 1)
        std_p = torch.tensor(target_std_list).view(1, 1, -1, 1, 1)

        force_mean_list = [float(x) for x in self.force_mean_str.split(',')]
        force_std_list = [float(x) for x in self.force_std_str.split(',')]
        mean_a = torch.tensor(force_mean_list).view(1, 1, -1, 1, 1)
        std_a = torch.tensor(force_std_list).view(1, 1, -1, 1, 1)

        # # initial
        # if mode == 'train':
        #     init_cond_name = self.init_cond_path
        # else:
        #     init_cond_name = self.init_cond_test_path

        for idx_t, start_t in enumerate(coords_time):
            for i in range(input_length):
                forcings_name = self.forcings_path + str(i+start_t).zfill(5) + ".pfb"
                frame_np = read_pfb(get_absolute_path(forcings_name)).astype(np.float32)
                frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
                frame_im = (frame_im-mean_a)/std_a
                for idx_s, (y, x) in enumerate(coords_space):
                    forcings[idx_t*num_patch+idx_s:idx_t*num_patch+idx_s+1, i:i+1, :, :, :] = \
                        frame_im[:, :, 1:11, y:y+self.patch_size, x:x+self.patch_size]

                targets_name = self.targets_path + str(i+start_t).zfill(5) + ".pfb"
                frame_np = read_pfb(get_absolute_path(targets_name)).astype(np.float32)
                frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
                frame_im = (frame_im-mean_p)/std_p
                for idx_s, (y, x) in enumerate(coords_space):
                    targets[idx_t*num_patch+idx_s:idx_t*num_patch+idx_s+1, i:i+1, :, :, :] = \
                        frame_im[:, :, :, y:y+self.patch_size, x:x+self.patch_size]

            init_cond_name = self.targets_path + str(start_t-1).zfill(5) + ".pfb"
            frame_np = read_pfb(get_absolute_path(init_cond_name)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            frame_im = (frame_im-mean_p)/std_p
            for idx_s, (y, x) in enumerate(coords_space):
                init_cond[idx_t*num_patch+idx_s:idx_t*num_patch+idx_s+1, 0, :, :, :] = \
                    frame_im[:, :, :, y:y+self.patch_size, x:x+self.patch_size]
                
        static_inputs = static_inputs_temp.repeat(num_seq,1,1,1,1) #.to(self.input_param.device)

        # === 统计信息输出（按当前精度FP32） ===
        bytes_per_element = 4  # FP32

        total_elements = (
            init_cond.numel() +
            static_inputs.numel() +
            forcings.numel() +
            targets.numel()
        )

        mem_fp32 = total_elements * bytes_per_element / (1024**3)

        print(f"Num patches: {num_patch}")
        print(f"Num sequences: {num_seq}")
        print(f"Total samples: {num_patch*num_seq}")
        print(f"Estimated memory: {mem_fp32:.2f} GB")
                 
        return init_cond, static_inputs, forcings, targets, num_patch*num_seq

    def get_train_input_handle(self):
        init_cond, static_inputs, forcings, targets, total_seq \
            = self.load_data(mode='train')
        return InputHandle(init_cond, static_inputs, forcings, targets, 
                           total_seq, self.input_param, mode='train')

    def get_test_input_handle(self):
        init_cond, static_inputs, forcings, targets, total_seq \
            = self.load_data(mode='test')
        return InputHandle(init_cond, static_inputs, forcings, targets, 
                           total_seq, self.input_param, mode='test')

