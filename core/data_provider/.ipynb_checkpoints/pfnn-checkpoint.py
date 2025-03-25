__author__ = 'gaozhifeng'
import numpy as np
import os
# import cv2
# from PIL import Image
import logging
# import random
# from typing import Iterable, List
# from dataclasses import dataclass
from core.utils import preprocess
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import write_pfb, read_pfb

logger = logging.getLogger(__name__)

class InputHandle:
    def __init__(self, init_cond, static_inputs, forcings, targets, total_seq, configs):
        self.name = configs.pf_runname
        self.batch_size = configs.batch_size
        self.img_width = configs.patch_size
        self.init_cond = init_cond
        self.static_inputs = static_inputs
        self.forcings = forcings
        self.targets = targets
        self.total_seq = total_seq
        # self.current_p = 0
        # self.current_batch_indices = []
        self.input_length = configs.input_length

    def total(self):
        return self.total_seq

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data ")
        if do_shuffle:
            rand_perm = torch.randperm(self.total_seq)
            data = data[rand_perm]
        self.current_p = 0

    def next(self):
        self.current_p += self.batch_size
        if self.no_batch_left():
            return None

    def no_batch_left(self):
        if self.current_p + self.batch_size >= self.total():
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + \
                ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None

        init_cond_batch     = torch.zeros(self.batch_size, 1, 
                                          self.init_cond_channels, self.img_width, self.img_width)
        static_inputs_batch = torch.zeros(self.batch_size, 1, 
                                          self.static_inputs_channels, self.img_width, self.img_width)
        forcigs_batch       = torch.zeros(self.batch_size, self.input_length, 
                                          self.act_channels, self.img_width, self.img_width)
        targets_batch       = torch.zeros(self.batch_size, self.input_length, 
                                          self.img_channels, self.img_width, self.img_width)

        init_cond_batch = self.init_cond[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        static_inputs_batch = self.static_inputs[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        forcigs_batch = self.forcigs[self.current_p:self.current_p + self.batch_size, :, :, :, :]
        targets_batch = self.targets[self.current_p:self.current_p + self.batch_size, :, :, :, :]
            
        return init_cond_batch, static_inputs_batch, forcigs_batch, targets_batch

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
        
        # the files should be continuous in time
        self.start_step = configs.start_step
        self.end_step   = configs.end_step
        self.timesteps  = configs.end_step - configs.start_step + 1
        
        # the RNN length
        self.input_length = configs.input_length
        
        # the origional size of the modeling domain and the subdomain size you want to 
        # insert into the CNN
        self.img_height = configs.img_height
        self.img_width  = configs.img_width
        self.patch_size = configs.patch_size
        
        self.init_cond_channels = configs.init_cond_channels
        self.static_channels = configs.static_channels
        self.act_channels = configs.act_channels
        self.img_channels = configs.img_channels
        
    def load_data(self, mode='train'):
        
        # model is no use, but keep it here and can be removed later
        
        # drop the residual cells
        num_patch  = (self.img_height // self.patch_size) * (self.img_width // self.patch_size)
        # drop the residual steps
        num_seq = self.timesteps // self.input_length
        framesteps = num_seq * self.input_length
        
        init_cond_temp = np.empty((num_patch*num_seq, 1, self.init_cond_channels, self.patch_size, self.patch_size),
                                   dtype=np.float32)  # np.float32
        
        static_inputs_temp = np.empty((num_patch, 1, self.static_channels, self.patch_size, self.patch_size),
                                       dtype=np.float32)  # np.float32
        
        forcings_temp = np.empty((num_patch, framesteps, self.act_channels, self.patch_size, self.patch_size),
                                  dtype=np.float32)  # np.float32
        
        targets_temp = np.empty((num_patch, framesteps, self.img_channels, self.patch_size, self.patch_size),
                                 dtype=np.float32)  # np.float32

        # static
        static_inputs_name = self.static_inputs_path
        frame_im = read_pfb(get_absolute_path(static_inputs_name))
        frame_im = np.expand_dims(np.expand_dims(frame_im, axis=0), axis=0)
        frame_np = preprocess.reshape_patch(frame_im, self.patch_size)
        # static_inputs_temp[:,0,:,:,:] = frame_np.astype(np.float32)
        static_inputs_temp = frame_np.astype(np.float32)
        
        # initial
        init_cond_name = self.init_cond_path
        frame_im = read_pfb(get_absolute_path(init_cond_name))
        frame_im = np.expand_dims(np.expand_dims(frame_im, axis=0), axis=0)
        frame_np = preprocess.reshape_patch(frame_im, self.patch_size)
        # init_cond_temp[0,0,:,:,:] = frame_np.astype(np.float32)
        init_cond_temp[0:num_patch,:,:,:,:] = frame_np.astype(np.float32)
                
        # read forcings and targets
        count = 0
        for i in range(framesteps):
            
            forcings_name = self.forcings_path + str(i+start_step).zfill(5) + ".pfb"
            frame_im = read_pfb(get_absolute_path(forcings_name))
            frame_im = np.expand_dims(np.expand_dims(frame_im, axis=0), axis=0)
            frame_np = preprocess.reshape_patch(frame_im, self.patch_size)
            forcings_temp[:,i,:,:,:] = frame_np.astype(np.float32)

            targets_name = self.targets_path + str(i+start_step).zfill(5) + ".pfb"
            frame_im = read_pfb(get_absolute_path(targets_name))
            frame_im = np.expand_dims(np.expand_dims(frame_im, axis=0), axis=0)
            frame_np = preprocess.reshape_patch(frame_im, self.patch_size)
            targets_temp[:,i,:,:,:] = frame_np.astype(np.float32)
            
            if ((i+1) % self.input_length == 0) and (i+1 != framesteps) :
                count = count + 1
                init_cond_data[num_patch*count:num_patch*(count+1),0,:,:,:] \
                    = frame_np.astype(np.float32)
                
        # reshape forcings and targets
        frocings_data = np.reshape(forcings_temp, (num_patch*num_seq,
                                                    input_length, self.act_channels, 
                                                     self.patch_size, self.patch_size))
            
        targets_data = np.reshape(targets_temp, (num_patch*num_seq,
                                                  input_length, self.act_channels, 
                                                   self.patch_size, self.patch_size))
        
        init_cond = torch.FloatTensor(init_cond).to(self.configs.device)
        static_inputs_tensor = torch.FloatTensor(static_inputs).to(self.configs.device)
        forcings = torch.FloatTensor(forcings).to(self.configs.device)
        targets = torch.FloatTensor(targets).to(self.configs.device)
        
        # repeat static
        static_inputs = static_inputs_tensor.repeat(num_seq,1,1,1,1)
                
        return init_cond, static_inputs, forcings, targets, num_patch*num_seq

    def get_train_input_handle(self):
        init_cond, static_inputs, forcings, targets, total_seq \
            = self.load_data(mode='train')
        return InputHandle(init_cond, static_inputs, forcings, targets, 
                           total_seq, self.input_param)

    def get_test_input_handle(self):
        init_cond, static_inputs, forcings, targets, total_seq \
            = self.load_data(mode='test')
        return InputHandle(init_cond, static_inputs, forcings, targets, 
                           total_seq, self.input_param)

