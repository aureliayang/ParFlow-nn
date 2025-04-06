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
        self.input_length = configs.input_length
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

        init_cond_batch     = torch.zeros(self.batch_size, 1, self.init_cond_channel,
                                          self.img_width, self.img_width).to(self.configs.device)
        static_inputs_batch = torch.zeros(self.batch_size, 1, self.static_channel,
                                          self.img_width, self.img_width).to(self.configs.device)
        forcings_batch       = torch.zeros(self.batch_size, self.input_length, self.act_channel,
                                          self.img_width, self.img_width).to(self.configs.device)
        targets_batch       = torch.zeros(self.batch_size, self.input_length, self.img_channel,
                                          self.img_width, self.img_width).to(self.configs.device)

        init_cond_batch = self.init_cond[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        static_inputs_batch = self.static_inputs[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        forcings_batch = self.forcings[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        targets_batch = self.targets[self.current_p:self.current_p + self.batch_size, :, :, :, :]
            
        return forcings_batch, init_cond_batch, static_inputs_batch, targets_batch

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
        self.init_cond_path = os.path.join(configs.init_cond_path, 
                                           configs.init_cond_filename)   
        
        # currently, please combine static parameters manually and provide it path
        self.static_inputs_path = os.path.join(configs.static_inputs_path, 
                                               configs.static_inputs_filename)
        
        forcings_filename = configs.pf_runname + ".out.evaptrans."
        self.forcings_path = os.path.join(configs.forcings_path, forcings_filename) 
        
        targets_filename = configs.pf_runname + ".out.press."
        self.targets_path = os.path.join(configs.targets_path, targets_filename) 

        self.target_norm_path = os.path.join(configs.targets_path,configs.target_norm_file)
        self.force_norm_path = os.path.join(configs.forcings_path,configs.force_norm_file)
        
        # the files should be continuous in time
        self.training_start_step = configs.training_start_step
        self.training_timesteps  = configs.training_end_step - configs.training_start_step + 1
        self.test_start_step = configs.test_start_step
        self.test_timesteps  = configs.test_end_step - configs.test_start_step + 1
        
        # the RNN length
        self.input_length = configs.input_length
        
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
            start_step = self.training_start_step
            timesteps  = self.training_timesteps
        else:
            start_step = self.test_start_step
            timesteps  = self.test_timesteps
        
        # drop the residual cells
        num_patch_y = self.img_height // self.patch_size 
        num_patch_x = self.img_width // self.patch_size
        num_patch = num_patch_x * num_patch_y
        # drop the residual steps
        num_seq = timesteps // self.input_length
        framesteps = num_seq * self.input_length
        
        init_cond = torch.empty((num_patch*num_seq, 1, self.init_cond_channel, self.patch_size, self.patch_size),
                                 dtype=torch.float)  # np.float32
        
        static_inputs_temp = torch.empty((num_patch, 1, self.static_channel, self.patch_size, self.patch_size),
                                          dtype=torch.float)  # np.float32
        
        forcings_temp = torch.empty((num_patch, framesteps, self.act_channel, self.patch_size, self.patch_size),
                                     dtype=torch.float)  # np.float32
        
        targets_temp = torch.empty((num_patch, framesteps, self.img_channel, self.patch_size, self.patch_size),
                                    dtype=torch.float)  # np.float32

        # static
        static_inputs_name = self.static_inputs_path
        frame_np = read_pfb(get_absolute_path(static_inputs_name)).astype(np.float32)
        frame_np = frame_np[:, 0:num_patch_y*self.patch_size, 0:num_patch_x*self.patch_size] # drop off
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        mean = frame_im.mean(dim=(3,4), keepdim=True)
        std = frame_im.std(dim=(3,4), keepdim=True)
        frame_im = (frame_im-mean)/std
        static_inputs_temp[:,:,:,:,:] = preprocess.reshape_patch(frame_im, self.patch_size)
        
        # initial
        init_cond_name = self.init_cond_path
        frame_np = read_pfb(get_absolute_path(init_cond_name)).astype(np.float32)
        frame_np = frame_np[:, 0:num_patch_y*self.patch_size, 0:num_patch_x*self.patch_size]
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        mean = frame_im.mean(dim=(3,4), keepdim=True)
        std = frame_im.std(dim=(3,4), keepdim=True)
        frame_im = (frame_im-mean)/std
        init_cond[0:num_patch,:,:,:,:] = preprocess.reshape_patch(frame_im, self.patch_size)

        press_name = self.target_norm_path
        frame_np = read_pfb(get_absolute_path(press_name)).astype(np.float32)
        frame_np = frame_np[:, 0:num_patch_y*self.patch_size, 0:num_patch_x*self.patch_size]
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        mean_p = frame_im.mean(dim=(3,4), keepdim=True)
        std_p = frame_im.std(dim=(3,4), keepdim=True)

        forcings_name = self.force_norm_path
        frame_np = read_pfb(get_absolute_path(forcings_name)).astype(np.float32)
        frame_np = frame_np[6:10, 0:num_patch_y*self.patch_size, 0:num_patch_x*self.patch_size]
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        mean_a = frame_im.mean(dim=(3,4), keepdim=True)
        std_a = frame_im.std(dim=(3,4), keepdim=True)
                
        # read forcings and targets
        count = 0
        for i in range(framesteps):
            
            forcings_name = self.forcings_path + str(i+start_step).zfill(5) + ".pfb"
            frame_np = read_pfb(get_absolute_path(forcings_name)).astype(np.float32)
            frame_np = frame_np[6:10, 0:num_patch_y*self.patch_size, 0:num_patch_x*self.patch_size]
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            frame_im = (frame_im-mean_a)/std_a
            forcings_temp[:,i:i+1,:,:,:] = preprocess.reshape_patch(frame_im, self.patch_size)

            targets_name = self.targets_path + str(i+start_step).zfill(5) + ".pfb"
            frame_np = read_pfb(get_absolute_path(targets_name)).astype(np.float32)
            frame_np = frame_np[:, 0:num_patch_y*self.patch_size, 0:num_patch_x*self.patch_size]
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            frame_im = (frame_im-mean_p)/std_p
            targets_temp[:,i:i+1,:,:,:] = preprocess.reshape_patch(frame_im, self.patch_size)
            
            if ((i+1) % self.input_length == 0) and (i+1 != framesteps) :
                count = count + 1
                init_cond[num_patch*count:num_patch*(count+1),0,:,:,:] \
                    = targets_temp[:,i,:,:,:]
                
        # # reshape forcings and targets
        forcings = preprocess.reshape_patch_time(forcings_temp, self.input_length)
        forcings = forcings.to(self.input_param.device)    
        targets = preprocess.reshape_patch_time(targets_temp, self.input_length)
        targets = targets.to(self.input_param.device)
        
        # repeat static
        static_inputs = static_inputs_temp.repeat(num_seq,1,1,1,1).to(self.input_param.device)
        init_cond = init_cond.to(self.input_param.device)
                 
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

