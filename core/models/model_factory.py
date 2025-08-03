__author__ = 'chen yang'

import os
import torch
from torch.optim import AdamW
from torch.optim import lr_scheduler
from core.models import predrnn_pf
import ctypes
from ctypes import c_double, c_int, c_char_p, POINTER, byref, CDLL
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import write_pfb, read_pfb
import numpy as np
from core.utils import preprocess
# from core.models import predrnn, predrnn_v2, action_cond_predrnn, action_cond_predrnn_v2

class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        networks_map = {
            'predrnn_pf': predrnn_pf.RNN,
            # 'predrnn': predrnn.RNN,
            # 'predrnn_v2': predrnn_v2.RNN,
            # 'action_cond_predrnn': action_cond_predrnn.RNN,
            # 'action_cond_predrnn_v2': action_cond_predrnn_v2.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError('Name of network unknown %s' % configs.model_name)

        self.optimizer = AdamW(self.network.parameters(), lr=configs.lr, betas=[0.8, 0.95], weight_decay=1e-2)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)

    def save(self, itr):
        stats = {}
        stats['net_param'] = self.network.state_dict()
        checkpoint_path = os.path.join(self.configs.save_dir, 'model.ckpt'+'-'+str(itr))
        torch.save(stats, checkpoint_path)
        print("save model to %s" % checkpoint_path)

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path)
        self.network.load_state_dict(stats['net_param'])

        # stats = torch.load(checkpoint_path, map_location='cpu')  # 强制先加载到CPU
        # self.network.load_state_dict(stats['net_param'])
        # self.network.to(self.configs.device)  # 再整体转到GPU

    def train(self, forcings, init_cond, static_inputs, targets):
        # forcings_tensor = torch.FloatTensor(forcings).to(self.configs.device)
        # init_cond_tensor = torch.FloatTensor(init_cond).to(self.configs.device)
        # static_inputs_tensor = torch.FloatTensor(static_inputs).to(self.configs.device)
        # targets_tensor = torch.FloatTensor(targets).to(self.configs.device)
        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        self.optimizer.zero_grad()

        batch, timesteps, channels, height, width = forcings.shape

        next_frames = []
        h_t = []
        c_t = []
        delta_c_list = []
        delta_m_list = []
        decouple_loss = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        memory = self.network.memory_encoder(init_cond[:, 0])
        c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

        net = init_cond[:, 0]
        net_temp = []

        for t in range(timesteps):
            net, net_temp, d_loss_step, h_t, c_t, memory, delta_c_list, delta_m_list \
                  = self.network(forcings[:, t], net, net_temp,
                                 h_t, c_t, memory, delta_c_list, delta_m_list)
            next_frames.append(net)
            # decouple_loss.append(torch.mean(torch.stack(d_loss_step)))
            decouple_loss += d_loss_step
        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=1)

        # 计算预测梯度
        dy_pred = next_frames[:, :, :, 1:, :] - next_frames[:, :, :, :-1, :]
        dx_pred = next_frames[:, :, :, :, 1:] - next_frames[:, :, :, :, :-1]

        # 计算真实梯度
        dy_true = targets[:, :, :, 1:, :] - targets[:, :, :, :-1, :]
        dx_true = targets[:, :, :, :, 1:] - targets[:, :, :, :, :-1]

        # 梯度差异 loss
        grad_loss = torch.mean(torch.abs(dy_pred - dy_true)) + \
                    torch.mean(torch.abs(dx_pred - dx_true))

        loss = self.network.MSE_criterion(next_frames, targets) + grad_loss + \
            self.configs.decouple_beta * decouple_loss

        loss.backward()
        self.optimizer.step()
        # self.scheduler.step()
        # return loss.detach().cpu().numpy()
        return loss.item()

    def test(self, forcings, init_cond, static_inputs):
        # frames_tensor = torch.FloatTensor(frames).to(self.configs.device)
        # mask_tensor = torch.FloatTensor(mask).to(self.configs.device)
        with torch.no_grad():
            batch, timesteps, channels, height, width = forcings.shape

            next_frames = []
            h_t = []
            c_t = []
            delta_c_list = []
            delta_m_list = []
            # decouple_loss = []

            for i in range(self.num_layers):
                zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
                h_t.append(zeros)
                c_t.append(zeros)
                delta_c_list.append(zeros)
                delta_m_list.append(zeros)

            memory = self.network.memory_encoder(init_cond[:, 0])
            c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

            net = init_cond[:, 0]
            net_temp = []

            for t in range(timesteps):

                net, net_temp, _, h_t, c_t, memory, delta_c_list, delta_m_list \
                    = self.network(forcings[:, t], net, net_temp,
                                    h_t, c_t, memory, delta_c_list, delta_m_list)
                next_frames.append(net)
                # decouple_loss.append(d_loss_step)
            # decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
            next_frames = torch.stack(next_frames, dim=1)
            # loss = self.network.MSE_criterion(next_frames, targets) + self.configs.beta * decouple_loss

            # next_frames, _ = self.network(forcings, init_cond, static_inputs, targets)
        # return next_frames.detach().cpu().numpy()
        return next_frames

    def test_lsm(self):

        with torch.no_grad():

            #get init and static, reshape
            num_patch_y = self.configs.img_height // self.configs.patch_size 
            num_patch_x = self.configs.img_width // self.configs.patch_size
            length_x = num_patch_x*self.configs.patch_size
            length_y = num_patch_y*self.configs.patch_size
            num_patch = num_patch_x*num_patch_y

            static_inputs = torch.empty((num_patch, 1, self.configs.static_channel, self.configs.patch_size,
                                         self.configs.patch_size), dtype=torch.float)  # np.float32
            init_cond = torch.empty((num_patch, 1, self.configs.init_cond_channel, self.configs.patch_size, 
                                     self.configs.patch_size), dtype=torch.float)  # np.float32
            forcings_temp = torch.empty((num_patch, 1, self.configs.act_channel, self.configs.patch_size, 
                                         self.configs.patch_size), dtype=torch.float)  # np.float32
            # static
            static_inputs_name = os.path.join(self.configs.static_inputs_path, 
                                              self.configs.static_inputs_filename) 
            frame_np = read_pfb(get_absolute_path(static_inputs_name)).astype(np.float32)
            frame_np = frame_np[:, :length_y, :length_x] # drop off
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            mean = frame_im.mean(dim=(3,4), keepdim=True)
            std = frame_im.std(dim=(3,4), keepdim=True)
            frame_im = (frame_im-mean)/std
            static_inputs[:,:,:,:,:] = preprocess.reshape_patch(frame_im, self.configs.patch_size)
            static_inputs = static_inputs.to(self.configs.device)

            target_norm_path = os.path.join(self.configs.targets_path,self.configs.target_norm_file)
            frame_np = read_pfb(get_absolute_path(target_norm_path)).astype(np.float32)
            frame_np = frame_np[:, :length_y, :length_x]
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            mean_p = frame_im.mean(dim=(3,4), keepdim=True)
            std_p = frame_im.std(dim=(3,4), keepdim=True)

            force_norm_path = os.path.join(self.configs.forcings_path,self.configs.force_norm_file)
            frame_np = read_pfb(get_absolute_path(force_norm_path)).astype(np.float32)
            frame_np = frame_np[1:11, :length_y, :length_x]  #10 layers
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            mean_a = frame_im.mean(dim=(3,4), keepdim=True)
            std_a = frame_im.std(dim=(3,4), keepdim=True)
            
            init_cond_name = os.path.join(self.configs.init_cond_test_path,
                                          self.configs.init_cond_test_filename)
            frame_np = read_pfb(get_absolute_path(init_cond_name)).astype(np.float32)
            frame_np = frame_np[:, :length_y, :length_x]
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            frame_im = (frame_im-mean_p)/std_p
            init_cond[:,:,:,:,:] = preprocess.reshape_patch(frame_im, self.configs.patch_size)
            init_cond = init_cond.to(self.configs.device)

            #change forcings to static to get the shape?
            batch, _, channels, height, width = static_inputs.shape

            next_frames = []
            h_t = []
            c_t = []
            delta_c_list = []
            delta_m_list = []

            for i in range(self.num_layers):
                zeros = torch.zeros([batch, self.num_hidden[i], height, width]).cuda()
                h_t.append(zeros)
                c_t.append(zeros)
                delta_c_list.append(zeros)
                delta_m_list.append(zeros)

            memory = self.network.memory_encoder(init_cond[:, 0])
            c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

            net = init_cond[:, 0]
            net_temp = []

            lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libclm_lsm.so'))
            lib = ctypes.CDLL(lib_path)
            print("CLM shared library loaded successfully.")

            self._set_clm_lsm_c_argtypes(lib)

            # 构造虚拟参数（用实际数据替换）
            nz, clm_nz = 11, 10
            nx, ny = self.configs.img_width, self.configs.img_height

            size_2d  = (nx+2) * (ny+2) * 3
            size_3d  = (nx+2) * (ny+2) * (nz+2)
            size_clm = (nx+2) * (ny+2) * (clm_nz+2)

            temp_arr_2d = np.ones(size_2d, dtype=np.float64)
            temp_arr_3d = np.ones(size_3d, dtype=np.float64)
            temp_arr_clm = np.ones(size_clm, dtype=np.float64)

            targets_filename = self.configs.pf_runname + ".out."
            targets_path = os.path.join(self.configs.targets_path, targets_filename)
            alpha_filename = targets_path+'alpha.pfb'
            n_filename = targets_path+'n.pfb'
            theta_s_filename = targets_path+'ssat.pfb'
            theta_r_filename = targets_path+'sres.pfb'

            mask_filename = targets_path+'mask.pfb'
            porosity_filename = targets_path+'porosity.pfb'
            pf_dz_mult_filename = targets_path+'dz_mult.pfb'

            alpha = read_pfb(alpha_filename)
            n_value = read_pfb(n_filename)
            theta_s = read_pfb(theta_s_filename)
            theta_r = read_pfb(theta_r_filename)

            press_temp = read_pfb(init_cond_name)
            topo = read_pfb(mask_filename)
            topo[:,length_y:,length_x:] = -9999.
            porosity = read_pfb(porosity_filename)
            pf_dz_mult = read_pfb(pf_dz_mult_filename)

            topo = np.pad(topo, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
            topo = topo.flatten()
            porosity = np.pad(porosity, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
            porosity = porosity.flatten()
            pf_dz_mult = np.pad(pf_dz_mult, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
            pf_dz_mult = pf_dz_mult.flatten()

            for t in range(self.configs.test_start_step, self.configs.test_end_step + 1):

                # read 8 forcings, cal saturation
                hour = t % 24
                time1 = str(t // 24 * 24 + 1).zfill(6) 
                time2 = str(t // 24 * 24 + 24).zfill(6) 

                general_path = os.path.join(self.configs.lsm_forcings_path,self.configs.lsm_forcings_name)
                sw_pf_filename = general_path +'.DSWR.' + time1 + '_to_' + time2 + '.pfb'
                lw_pf_filename = general_path +'.DLWR.' + time1 + '_to_' + time2 + '.pfb'
                prcp_pf_filename = general_path +'.APCP.' + time1 + '_to_' + time2 + '.pfb'
                tas_pf_filename = general_path +'.Temp.' + time1 + '_to_' + time2 + '.pfb'
                u_pf_filename = general_path +'.UGRD.' + time1 + '_to_' + time2 + '.pfb'
                v_pf_filename = general_path +'.VGRD.' + time1 + '_to_' + time2 + '.pfb'
                patm_pf_filename = general_path +'.Press.' + time1 + '_to_' + time2 + '.pfb'
                qatm_pf_filename = general_path +'.SPFH.' + time1 + '_to_' + time2 + '.pfb'

                # read forcings and pad dim
                pf_vars = {'sw_pf': sw_pf_filename, 'lw_pf': lw_pf_filename,
                           'prcp_pf': prcp_pf_filename, 'tas_pf': tas_pf_filename,
                           'u_pf': u_pf_filename, 'v_pf': v_pf_filename,
                           'patm_pf': patm_pf_filename, 'qatm_pf': qatm_pf_filename}
                variables = {}
                for var_name, filename in pf_vars.items():
                    data = read_pfb(get_absolute_path(filename))[hour, :, :]
                    data = np.expand_dims(data, axis=0)
                    variables[var_name] = data
                sw_pf, lw_pf, prcp_pf, tas_pf = variables['sw_pf'], variables['lw_pf'], variables['prcp_pf'], variables['tas_pf']
                u_pf, v_pf, patm_pf, qatm_pf = variables['u_pf'], variables['v_pf'], variables['patm_pf'], variables['qatm_pf']

                # cal saturation
                satur = self._vg_saturation(press_temp, alpha, n_value, theta_r, theta_s)
                press = np.pad(press_temp, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
                press = press.flatten()

                # pad ghost cells and flatten
                variables = {'satur': satur, 'sw_pf': sw_pf, 'lw_pf': lw_pf, 'prcp_pf': prcp_pf,
                             'tas_pf': tas_pf, 'u_pf': u_pf, 'v_pf': v_pf, 'patm_pf': patm_pf, 'qatm_pf': qatm_pf}
                for name, arr in variables.items():
                    arr = np.pad(arr, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
                    arr = arr.flatten()
                    variables[name] = arr
                satur = variables['satur']
                sw_pf, lw_pf, prcp_pf, tas_pf = variables['sw_pf'], variables['lw_pf'], variables['prcp_pf'], variables['tas_pf']
                u_pf, v_pf, patm_pf, qatm_pf = variables['u_pf'], variables['v_pf'], variables['patm_pf'], variables['qatm_pf']
                
                # call lsm
                lib.clm_lsm_c(
                    press.ctypes.data_as(POINTER(c_double)),        # pressure
                    satur.ctypes.data_as(POINTER(c_double)),        # saturation
                    temp_arr_3d.ctypes.data_as(POINTER(c_double)),  # evap_trans
                    topo.ctypes.data_as(POINTER(c_double)),         # topo
                    porosity.ctypes.data_as(POINTER(c_double)),     # porosity
                    pf_dz_mult.ctypes.data_as(POINTER(c_double)),   # pf_dz_mult
                    t,                                  # istep_pf
                    1.0,                                # dt
                    t - 1.,                             # time ？？？
                    self.configs.test_start_step - 1.,  # start_time_pf
                    961.72,                   # pdx
                    961.72,                   # pdy
                    200.,                     # pdz
                    0, 0, nx, ny, nz, nx+2, ny+2, nz+2, 0,         # ix, iy, nx, ny, nz, nx_f, ny_f, nz_f, nz_rz
                    0, 0, 0, 0, nx, ny, 0,                         # ip,npp,npq,npr,gnx,gny,rank
                    sw_pf.ctypes.data_as(POINTER(c_double)),
                    lw_pf.ctypes.data_as(POINTER(c_double)),
                    prcp_pf.ctypes.data_as(POINTER(c_double)),
                    tas_pf.ctypes.data_as(POINTER(c_double)),
                    u_pf.ctypes.data_as(POINTER(c_double)),
                    v_pf.ctypes.data_as(POINTER(c_double)),
                    patm_pf.ctypes.data_as(POINTER(c_double)),
                    qatm_pf.ctypes.data_as(POINTER(c_double)),
                    *(temp_arr_2d.ctypes.data_as(POINTER(c_double)) for _ in range(18)),  # 所有 double* 参数
                    temp_arr_clm.ctypes.data_as(POINTER(c_double)),
                    1, 0, 0,                                 # clm_dump_interval, clm_1d_out, clm_forc_veg
                    b"./output/", 9,                         # clm_output_dir, clm_output_dir_length
                    1,                                       # clm_bin_output_dir
                    1, 0, 1,                                 # write_CLM_binary, slope_accounting_CLM, beta_typepf
                    2,                                       # veg_water_stress_typepf
                    0.2,                                     # wilting_pointpf
                    1.0,                                     # field_capacitypf
                    0.2,                                     # res_satpf
                    0, 0,                                    # irr_typepf, irr_cyclepf
                    0.,                                      # irr_ratepf
                    12.,                                     # irr_startpf
                    20.,                                     # irr_stoppf
                    0.5,                                     # irr_thresholdpf
                    temp_arr_2d.ctypes.data_as(POINTER(c_double)),    # qirr_pf
                    temp_arr_clm.ctypes.data_as(POINTER(c_double)),   # qirr_inst_pf
                    temp_arr_2d.ctypes.data_as(POINTER(c_double)),    # irr_flag_pf
                    2,                                       # irr_thresholdtypepf
                    10,                                      # soi_z
                    1, 0, 1, 1,                              # clm_next, clm_write_logs, clm_last_rst, clm_daily_rst
                    10, 10                                   # pf_nlevsoi, pf_nlevlak
                )
                #reshape evaptrans to get the forcing to network
                evap_trans = np.reshape(temp_arr_3d,(nz+2,ny+2,nx+2)).astype(np.float32)
                evap_trans = torch.from_numpy(evap_trans[2:nz+1,1:length_y+1,
                                                         1:length_x+1]).unsqueeze(0).unsqueeze(0)
                evap_trans = (evap_trans-mean_a)/std_a
                forcings_temp[:,:,:,:,:] = preprocess.reshape_patch(evap_trans, self.configs.patch_size)
                forcings = forcings_temp.to(self.configs.device)

                net, net_temp, _, h_t, c_t, memory, delta_c_list, delta_m_list \
                    = self.network(forcings[:,0], net, net_temp, h_t, c_t, memory, delta_c_list, delta_m_list)
                next_frames.append(net)

                #reshape back net to get the init_cond for next step
                press_ = preprocess.reshape_patch_back(net.unsqueeze(1), num_patch_x, num_patch_y)
                press_ = torch.squeeze((press_.detach().cpu())*std_p+mean_p).numpy().astype(np.float64)
                # padding
                press_temp[:,:length_y,:length_x] = press_

            next_frames = torch.stack(next_frames, dim=1)
        # return next_frames.detach().cpu().numpy()
        return next_frames, mean_p, std_p
    
    def _vg_saturation(self, h, alpha, n, theta_r, theta_s):
        m = 1 - 1 / n
        Se = np.ones_like(h)  # 初始化为全1
        h_neg = np.abs(h)
        
        # 在负压头处更新 Se，其他保持1
        vg_expr = (1 + (alpha * h_neg) ** n) ** (-m)
        Se = np.where(h < 0, vg_expr, Se)

        theta = theta_r + (theta_s - theta_r) * Se
        return theta
    
    def _set_clm_lsm_c_argtypes(self, lib):
        # 定义 clm_lsm_c 函数原型
        lib.clm_lsm_c.argtypes = [
            POINTER(c_double),  # pressure
            POINTER(c_double),  # saturation
            POINTER(c_double),  # evap_trans
            POINTER(c_double),  # topo
            POINTER(c_double),  # porosity
            POINTER(c_double),  # pf_dz_mult
            c_int,              # istep_pf
            c_double,           # dt
            c_double,           # time
            c_double,           # start_time_pf
            c_double,           # pdx
            c_double,           # pdy
            c_double,           # pdz
            c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, # ix~nz_rz
            c_int, c_int, c_int, c_int, c_int, c_int, c_int,               # ip~rank
            POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),  # sw_pf~tas_pf
            POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),  # u_pf~qatm_pf
            POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),  # lai_pf~slope_y_pf
            POINTER(c_double), POINTER(c_double),
            POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),  # eflx_lh_pf~eflx_grnd_pf
            POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),  # qflx系列
            POINTER(c_double), POINTER(c_double),                                        
            POINTER(c_double), POINTER(c_double), POINTER(c_double),                     # swe_pf, t_g_pf, t_soil_pf                  
            c_int, c_int, c_int,  # clm_dump_interval, clm_1d_out, clm_forc_veg
            c_char_p,             # clm_output_dir
            c_int,                # clm_output_dir_length
            c_int,                # clm_bin_output_dir
            c_int, c_int, c_int,  # write_CLM_binary, slope_accounting_CLM, beta_typepf
            c_int,                # veg_water_stress_typepf
            c_double, c_double, c_double,  # wilting_pointpf, field_capacitypf, res_satpf
            c_int, c_int,                  # irr_typepf, irr_cyclepf
            c_double, c_double, c_double, c_double,  # irr_ratepf~irr_thresholdpf
            POINTER(c_double), POINTER(c_double),  # qirr_pf, qirr_inst_pf
            POINTER(c_double), # irr_flag_pf
            c_int,  # irr_thresholdtypepf
            c_int,  # soi_z
            c_int, c_int, c_int, c_int,  # clm_next~clm_daily_rst
            c_int, c_int  # pf_nlevsoi, pf_nlevlak
        ]

        # 返回值类型为 None
        lib.clm_lsm_c.restype = None