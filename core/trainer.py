import os
import datetime
import numpy as np
import torch
from core.utils import preprocess
from parflow.tools.io import write_pfb


def train(model, forcings, init_cond, static_inputs, targets,
          alpha, n, theta_r, theta_s, porosity, specific_storage, mask,
          configs, itr):
    total, mse, grad, decouple, storage = model.train(
        forcings, init_cond, static_inputs, targets,
        alpha, n, theta_r, theta_s, porosity, specific_storage, mask
    )

    if itr % configs.display_interval == 0:
        print(
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            f"itr: {itr} "
            f"total: {total:.6f} "
            f"MSE: {mse:.6f} "
            f"grad: {grad:.6f} "
            f"decouple: {decouple:.6f} "
            f"storage: {storage:.6f}"
        )

def test(model, test_input_handle, configs, itr):
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'test...')

    test_input_handle.begin(do_shuffle=False)
    res_path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(res_path)

    batch_id = 0  # actually, only one batch when test
    mean_p = torch.tensor(configs.target_mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
    std_p = torch.tensor(configs.target_std, dtype=torch.float32).view(1, 1, -1, 1, 1)

    while (test_input_handle.no_batch_left() == False):

        batch_id = batch_id + 1
        # forcings, init_cond, static_inputs, targets = test_input_handle.get_batch()
        (   forcings, init_cond, static_inputs, targets,
            alpha, n, theta_r, theta_s, porosity, specific_storage, mask
        ) = test_input_handle.get_batch()


        img_gen = model.test(forcings, init_cond, static_inputs)
 
        # save prediction examples
        path = os.path.join(res_path, str(batch_id))
        os.mkdir(path)

        num_patch = test_input_handle.num_patch
        coords_space = test_input_handle.coords_space

        img_gen = preprocess.reshape_patch_back_time(img_gen, num_patch)
        img_gen = preprocess.reshape_patch_back(
            img_gen,
            coords_space,
            configs.img_height,
            configs.img_width,
            configs.patch_size
        )
        img_gen = torch.squeeze((img_gen.detach().cpu())*std_p+mean_p).numpy().astype(np.float64)
        img_tar = preprocess.reshape_patch_back_time(targets, num_patch)
        img_tar = preprocess.reshape_patch_back(
            img_tar,
            coords_space,
            configs.img_height,
            configs.img_width,
            configs.patch_size
        )
        img_tar = torch.squeeze((img_tar.detach().cpu())*std_p+mean_p).numpy().astype(np.float64)

        assert img_gen.shape == img_tar.shape

        print ('RMSE_TOTAL =', np.sqrt(np.mean((img_gen - img_tar) ** 2)))

        for i in range(img_gen.shape[0]):
            file_name = 'nn_gen.press.' + str(i+configs.test_start_step).zfill(5) + '.pfb'
            file_name = os.path.join(path, file_name)
            write_pfb(file_name, img_gen[i,:,:,:], dist=False)
            file_name = 'nn_tar.press.' + str(i+configs.test_start_step).zfill(5) + '.pfb'
            file_name = os.path.join(path, file_name)
            write_pfb(file_name, img_tar[i,:,:,:], dist=False)
        test_input_handle.next()

def test_lsm(model, configs, itr):
    img_gen, mean_p, std_p = model.test_lsm()
    img_gen = torch.squeeze((img_gen.detach().cpu())*std_p+mean_p).numpy().astype(np.float64)

    path = os.path.join(configs.gen_frm_dir, str(itr))
    os.mkdir(path)

    for i in range(configs.test_start_step, configs.test_end_step + 1):
        file_name = 'nn_gen.press.' + str(i).zfill(5) + '.pfb'
        file_name = os.path.join(path, file_name)
        write_pfb(file_name, img_gen[i-configs.test_start_step,:,:,:], dist=False)