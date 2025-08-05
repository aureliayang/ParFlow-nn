import os.path
import datetime
# import cv2
import numpy as np
# from skimage.metrics import structural_similarity as compare_ssim
from core.utils import preprocess, metrics
# import lpips
import torch
from parflow.tools.io import write_pfb, read_pfb
from parflow.tools.fs import get_absolute_path

# loss_fn_alex = lpips.LPIPS(net='alex')


def train(model, forcings, init_cond, static_inputs, targets, configs, itr):
    cost, cost1, cost2, cost3 = model.train(forcings, init_cond, static_inputs, targets)
    # if configs.reverse_input:
    #     ims_rev = np.flip(ims, axis=1).copy()
    #     cost += model.train(ims_rev, real_input_flag)
    #     cost = cost / 2

    # if itr % configs.display_interval == 0:
    #     print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'itr: ' + str(itr),'training loss: ' \
    #           + str(cost))
        # print('training loss: ' + str(cost))
    if itr % configs.display_interval == 0:
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f"itr: {itr} "
            f"total: {cost:.6f} "
            f"MSE: {cost1:.6f} "
            f"grad: {cost2:.6f} "
            f"decouple: {cost3:.6f}")

def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')
    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)
    batch_id = 0  # actually, only one batch when test

    while (test_input_handle.no_batch_left() == False):
        batch_id = batch_id + 1
        forcings, init_cond, static_inputs, targets = test_input_handle.get_batch()
        img_gen = model.test(forcings, init_cond, static_inputs)
 
        # save prediction examples
        path = os.path.join(res_path, str(batch_id))
        os.mkdir(path)
        num_seq = (configs.test_end_step - configs.test_start_step + 1) // configs.input_length_test
        num_patch_y = configs.img_height // configs.patch_size 
        num_patch_x = configs.img_width // configs.patch_size

        target_mean_list = [float(x) for x in configs.target_mean.split(',')]
        target_std_list = [float(x) for x in configs.target_std.split(',')]
        mean_p = torch.tensor(target_mean_list).view(1, 1, -1, 1, 1)
        std_p = torch.tensor(target_std_list).view(1, 1, -1, 1, 1)

        img_gen = preprocess.reshape_patch_back_time(img_gen, num_patch_x*num_patch_y)
        img_gen = preprocess.reshape_patch_back(img_gen, num_patch_x, num_patch_y)
        img_gen = torch.squeeze((img_gen.detach().cpu())*std_p+mean_p).numpy().astype(np.float64)

        img_tar = preprocess.reshape_patch_back_time(targets, num_patch_x*num_patch_y)
        img_tar = preprocess.reshape_patch_back(img_tar, num_patch_x, num_patch_y)
        img_tar = torch.squeeze((img_tar.detach().cpu())*std_p+mean_p).numpy().astype(np.float64)

        print ('RMSE_TOTAL =', np.sqrt(np.mean((img_gen - img_tar) ** 2)))

        for i in range(num_seq*configs.input_length_test):
            file_name = 'nn_gen.press.' + str(i+configs.test_start_step).zfill(5) + '.pfb'
            file_name = os.path.join(path, file_name)
            write_pfb(file_name, img_gen[i,:,:,:], dist=False)
            file_name = 'nn_tar.press.' + str(i+configs.test_start_step).zfill(5) + '.pfb'
            file_name = os.path.join(path, file_name)
            write_pfb(file_name, img_tar[i,:,:,:], dist=False)
        test_input_handle.next()

def test_lsm(model, configs, itr):
    ###return and save files
    num_patch_y = configs.img_height // configs.patch_size
    num_patch_x = configs.img_width // configs.patch_size

    img_gen, mean_p, std_p = model.test_lsm()
    img_gen = preprocess.reshape_patch_back(img_gen, num_patch_x, num_patch_y)
    img_gen = torch.squeeze((img_gen.detach().cpu())*std_p+mean_p).numpy().astype(np.float64)

    path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(path)

    for i in range(configs.test_start_step, configs.test_end_step + 1):
        file_name = 'nn_gen.press.' + str(i).zfill(5) + '.pfb'
        file_name = os.path.join(path, file_name)
        write_pfb(file_name, img_gen[i-configs.test_start_step,:,:,:], dist=False)

