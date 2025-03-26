import os.path
import datetime
# import cv2
import numpy as np
# from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
# import lpips
import torch
from parflow.tools.io import write_pfb, read_pfb

# loss_fn_alex = lpips.LPIPS(net='alex')


def train(model, forcings, init_cond, static_inputs, targets, configs, itr):
    cost = model.train(forcings, init_cond, static_inputs, targets)
    # if configs.reverse_input:
    #     ims_rev = np.flip(ims, axis=1).copy()
    #     cost += model.train(ims_rev, real_input_flag)
    #     cost = cost / 2

    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr))
        print('training loss: ' + str(cost))

def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    batch_id = 0  # actually, only one batch when test

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        forcings, init_cond, static_inputs, targets = test_input_handle.get_batch()
        img_gen = model.test(forcings, init_cond, static_inputs, targets)
 
        # save prediction examples
        path = os.path.join(res_path, str(batch_id))
        os.mkdir(path)
        num_seq = (configs.test_end_step - configs.test_start_step + 1) // configs.input_length
        num_patch_y = configs.img_height // configs.patch_size 
        num_patch_x = configs.img_width // configs.patch_size
        img_gen = preprocess.reshape_patch_back_time(img_gen, num_seq)
        img_gen = preprocess.reshape_patch_back(img_gen, num_patch_x, num_patch_y)
        img_gen = torch.squeeze(img_gen).numpy()
        for i in range(num_seq*configs.input_length):
            file_name = 'nn_gen.press.' + str(i+configs.test_start_step).zfill(5) + '.pfb'
            write_pfb(file_name, img_gen[i,:,:,:], dist=False)
        test_input_handle.next()

