__author__ = 'yunbo'

import numpy as np
import torch

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    # batch_size, seq_length, num_channels, img_height, img_width \
    #     = img_tensor.shape
    
    a = torch.split(img_tensor, patch_size, dim = 4) # in x direction
    b = torch.cat(a, dim = 0)
    c = torch.split(b, patch_size, dim = 3)
    patch_tensor = torch.cat(c, dim = 0)

    return patch_tensor

def reshape_patch_back(patch_tensor, num_x, num_y):
    assert 5 == patch_tensor.ndim
    # batch_size, seq_length, num_channels, img_height, img_width \
    #     = patch_tensor.shape
    
    a = torch.split(patch_tensor, num_x, dim = 0)
    b = torch.cat(a, dim = 3)
    c = torch.split(b, 1, dim = 0)
    img_tensor = torch.cat(c, dim = 4)

    return img_tensor

def reshape_patch_time(img_tensor, input_length):
    assert 5 == img_tensor.ndim
    # batch_size, seq_length, num_channels, img_height, img_width \
    #     = img_tensor.shape
    
    a = torch.split(img_tensor, input_length, dim = 1)
    patch_tensor = torch.cat(a, dim = 0)

    return patch_tensor

def reshape_patch_back_time(patch_tensor, num_patch):
    assert 5 == patch_tensor.ndim
    # batch_size, seq_length, num_channels, img_height, img_width \
    #     = patch_tensor.shape
    
    a = torch.split(patch_tensor, num_patch, dim = 0)
    img_tensor = torch.cat(a, dim = 1)

    return img_tensor


