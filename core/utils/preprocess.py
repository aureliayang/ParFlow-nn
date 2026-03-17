__author__ = 'yunbo'

import numpy as np
import torch

# def reshape_patch(img_tensor, patch_size):
#     assert 5 == img_tensor.ndim
#     # batch_size, seq_length, num_channels, img_height, img_width \
#     #     = img_tensor.shape
    
#     a = torch.split(img_tensor, patch_size, dim = 4) # in x direction
#     b = torch.cat(a, dim = 0)
#     c = torch.split(b, patch_size, dim = 3)
#     patch_tensor = torch.cat(c, dim = 0)

#     return patch_tensor

def reshape_patch(img_tensor, coords_space, patch_size):
    assert img_tensor.ndim == 5
    # img_tensor: [batch_size, seq_length, num_channels, img_height, img_width]
    # 目前通常 batch_size=1, seq_length=1

    batch_size, seq_length, num_channels, _, _ = img_tensor.shape
    assert batch_size == 1, f"Only batch_size=1 is supported now, got {batch_size}"
    assert seq_length == 1, f"Only seq_length=1 is supported now, got {seq_length}"

    num_patch = len(coords_space)

    patch_tensor = torch.empty(
        (num_patch, seq_length, num_channels, patch_size, patch_size),
        dtype=img_tensor.dtype,
        device=img_tensor.device
    )

    for idx_s, (y, x) in enumerate(coords_space):
        patch_tensor[idx_s:idx_s+1, :, :, :, :] = \
            img_tensor[:, :, :, y:y+patch_size, x:x+patch_size]

    return patch_tensor

# def reshape_patch_back(patch_tensor, num_x, num_y):
#     assert 5 == patch_tensor.ndim
#     # batch_size, seq_length, num_channels, img_height, img_width \
#     #     = patch_tensor.shape
    
#     a = torch.split(patch_tensor, num_x, dim = 0)
#     b = torch.cat(a, dim = 3)
#     c = torch.split(b, 1, dim = 0)
#     img_tensor = torch.cat(c, dim = 4)

#     return img_tensor

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

def reshape_patch_back(patch_tensor, coords_space, img_height, img_width, patch_size):
    assert patch_tensor.ndim == 5
    # patch_tensor: [num_patch, seq_length, num_channels, patch_size, patch_size]

    device = patch_tensor.device
    dtype = patch_tensor.dtype

    num_patch, seq_length, num_channels, _, _ = patch_tensor.shape

    # 重建图像
    img_tensor = torch.zeros(
        (seq_length, num_channels, img_height, img_width),
        dtype=dtype,
        device=device
    )

    # 计数器，用来对 overlap 区域求平均
    count_tensor = torch.zeros(
        (seq_length, num_channels, img_height, img_width),
        dtype=dtype,
        device=device
    )

    for idx, (y, x) in enumerate(coords_space):
        img_tensor[:, :, y:y+patch_size, x:x+patch_size] += patch_tensor[idx]
        count_tensor[:, :, y:y+patch_size, x:x+patch_size] += 1.0

    # 防止除零
    img_tensor = img_tensor / torch.clamp(count_tensor, min=1.0)

    return img_tensor


