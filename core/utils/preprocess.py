__author__ = 'yunbo'

import numpy as np
import torch

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    # batch_size, seq_length, num_channels, img_height, img_width \
    #     = img_tensor.shape
    
    a = torch.split(img_tensor, patch_size, dim = 4)
    b = torch.cat(a, dim = 0)
    c = torch.split(b, patch_size, dim = 3)
    patch_tensor = torch.cat(c, dim = 0)

    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size_x, patch_size_y):
    assert 5 == patch_tensor.ndim
    # batch_size, seq_length, num_channels, img_height, img_width \
    #     = patch_tensor.shape
    
    a = torch.split(patch_tensor, patch_size_y, dim = 0)
    b = torch.cat(a, dim = 3)
    c = torch.split(b, patch_size_x, dim = 0)
    img_tensor = torch.cat(c, dim = 4)

    return img_tensor

def reshape_patch_time(img_tensor, input_length):
    assert 5 == img_tensor.ndim
    # batch_size, seq_length, num_channels, img_height, img_width \
    #     = img_tensor.shape
    
    a = torch.split(img_tensor, input_length, dim = 1)
    patch_tensor = torch.cat(a, dim = 0)

    return patch_tensor

def reshape_patch_back_time(patch_tensor, num_seq):
    assert 5 == patch_tensor.ndim
    # batch_size, seq_length, num_channels, img_height, img_width \
    #     = patch_tensor.shape
    
    a = torch.split(patch_tensor, num_seq, dim = 0)
    img_tensor = torch.cat(a, dim = 1)

    return img_tensor

# def reshape_patch(img_tensor, patch_size):
#     assert 5 == img_tensor.ndim
#     batch_size = np.shape(img_tensor)[0]
#     seq_length = np.shape(img_tensor)[1]
#     num_channels = np.shape(img_tensor)[2]
#     img_height = np.shape(img_tensor)[3]
#     img_width = np.shape(img_tensor)[4]
#     
#     a = np.reshape(img_tensor, (batch_size, seq_length, num_channels,
#                                 patch_size, img_height//patch_size, 
#                                 patch_size, img_width//patch_size))
#     b = np.transpose(a, (0,4,6,1,2,3,5))
#     patch_tensor = np.reshape(b, (batch_size*img_height//patch_size*img_width//patch_size, 
#                                   seq_length, num_channels, patch_size, patch_size))
#     return patch_tensor
# 
# def reshape_patch_back(patch_tensor, patch_size):
#     assert 5 == patch_tensor.ndim
#     batch_size = np.shape(patch_tensor)[0]
#     seq_length = np.shape(patch_tensor)[1]
#     patch_height = np.shape(patch_tensor)[2]
#     patch_width = np.shape(patch_tensor)[3]
#     channels = np.shape(patch_tensor)[4]
#     img_channels = channels // (patch_size*patch_size)
#     a = np.reshape(patch_tensor, [batch_size, seq_length,
#                                   patch_height, patch_width,
#                                   patch_size, patch_size,
#                                   img_channels])
#     b = np.transpose(a, [0,1,2,4,3,5,6])
#     img_tensor = np.reshape(b, [batch_size, seq_length,
#                                 patch_height * patch_size,
#                                 patch_width * patch_size,
#                                 img_channels])
#     return img_tensor

