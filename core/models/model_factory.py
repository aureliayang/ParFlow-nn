__author__ = 'chen yang'

import os
import ctypes
import torch
import numpy as np
from torch.optim import AdamW
from ctypes import c_double, c_int, c_char_p, POINTER
from core.models import predrnn_pf
from parflow.tools.io import write_pfb, read_pfb
from core.utils import preprocess
from parflow.tools.fs import get_absolute_path


class Model(object):
    def __init__(self, configs):
        self.configs = configs
        self.num_hidden = [int(x) for x in configs.num_hidden.split(',')]
        self.num_layers = len(self.num_hidden)
        self.use_storage_loss = abs(float(configs.storage_beta)) > 1e-12

        networks_map = {
            'predrnn_pf': predrnn_pf.RNN,
        }

        if configs.model_name in networks_map:
            Network = networks_map[configs.model_name]
            self.network = Network(self.num_layers, self.num_hidden, configs).to(configs.device)
        else:
            raise ValueError(f'Name of network unknown {configs.model_name}')

        self.optimizer = AdamW(self.network.parameters(), lr=configs.lr, betas=(0.8, 0.95))

        if configs.lr_mode == "onecycle":
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=configs.max_lr,
                total_steps=configs.max_iterations,
                pct_start=configs.onecycle_pct_start,
                anneal_strategy='cos',
                div_factor=configs.onecycle_div_factor,
                final_div_factor=configs.onecycle_final_div_factor
            )
        elif configs.lr_mode == "constant":
            self.scheduler = None
        else:
            raise ValueError(f"Unknown lr_mode: {configs.lr_mode}")

        self.mean_p_t = torch.tensor(
            configs.target_mean, dtype=torch.float32, device=configs.device
        ).view(1, 1, -1, 1, 1)
        self.std_p_t = torch.tensor(
            configs.target_std, dtype=torch.float32, device=configs.device
        ).view(1, 1, -1, 1, 1)

        self.dx = configs.dx
        self.dy = configs.dy
        self.dz_t = torch.tensor(
            configs.dz, dtype=torch.float32, device=configs.device
        ).view(1, 1, -1, 1, 1)

    def save(self, itr):
        stats = {'net_param': self.network.state_dict()}
        checkpoint_path = os.path.join(self.configs.save_dir, f'model.ckpt-{itr}')
        torch.save(stats, checkpoint_path)
        print(f"save model to {checkpoint_path}")

    def load(self, checkpoint_path):
        print('load model:', checkpoint_path)
        stats = torch.load(checkpoint_path, map_location=torch.device(self.configs.device))
        self.network.load_state_dict(stats['net_param'])

    def _init_states(self, batch, height, width):
        h_t, c_t, delta_c_list, delta_m_list = [], [], [], []
        for hidden in self.num_hidden:
            zeros = torch.zeros([batch, hidden, height, width], device=self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)
            delta_c_list.append(zeros)
            delta_m_list.append(zeros)
        return h_t, c_t, delta_c_list, delta_m_list

    def _compute_storage_component(
        self,
        next_frames,
        targets,
        alpha,
        n,
        theta_r,
        theta_s,
        porosity,
        specific_storage,
        mask
    ):
        if not self.use_storage_loss:
            return torch.zeros((), dtype=next_frames.dtype, device=next_frames.device)

        required = [alpha, n, theta_r, theta_s, porosity, specific_storage, mask]
        if any(x is None for x in required):
            raise ValueError(
                "Storage loss is enabled, but one or more storage-related tensors are None."
            )

        next_frames_phys = next_frames * self.std_p_t + self.mean_p_t
        targets_phys = targets * self.std_p_t + self.mean_p_t

        sat_pred = self.vg_saturation_torch(next_frames_phys, alpha, n, theta_r, theta_s)
        sat_true = self.vg_saturation_torch(targets_phys, alpha, n, theta_r, theta_s)

        subsurface_pred = self.calculate_subsurface_storage_torch(
            porosity,
            next_frames_phys,
            sat_pred,
            specific_storage,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz_t,
            mask=mask
        )

        subsurface_true = self.calculate_subsurface_storage_torch(
            porosity,
            targets_phys,
            sat_true,
            specific_storage,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz_t,
            mask=mask
        )

        surface_pred = self.calculate_surface_storage_torch(
            next_frames_phys, dx=self.dx, dy=self.dy, mask=mask
        )

        surface_true = self.calculate_surface_storage_torch(
            targets_phys, dx=self.dx, dy=self.dy, mask=mask
        )

        storage_pred = subsurface_pred.sum(dim=2) + surface_pred
        storage_true = subsurface_true.sum(dim=2) + surface_true

        storage_loss = torch.mean(torch.abs(storage_pred - storage_true))
        return self.configs.storage_beta * storage_loss

    def train(
        self,
        forcings,
        init_cond,
        static_inputs,
        targets,
        alpha,
        n,
        theta_r,
        theta_s,
        porosity,
        specific_storage,
        mask
    ):
        if self.use_storage_loss:
            if alpha is None or n is None or theta_r is None or theta_s is None \
               or porosity is None or specific_storage is None or mask is None:
                raise ValueError(
                    "Storage loss is enabled, but storage-related inputs contain None."
                )

            if alpha.ndim == 4:
                alpha = alpha.unsqueeze(1)
                n = n.unsqueeze(1)
                theta_r = theta_r.unsqueeze(1)
                theta_s = theta_s.unsqueeze(1)
                porosity = porosity.unsqueeze(1)
                specific_storage = specific_storage.unsqueeze(1)
                mask = mask.unsqueeze(1)

        self.network.train()
        self.optimizer.zero_grad()

        batch, timesteps, channels, height, width = forcings.shape

        next_frames = []
        decouple_loss = []

        h_t, c_t, delta_c_list, delta_m_list = self._init_states(batch, height, width)

        memory = self.network.memory_encoder(init_cond[:, 0])
        c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

        net = init_cond[:, 0]
        net_temp = []
        h_t_temp = []

        for t in range(timesteps):
            net, net_temp, h_t_temp, d_loss_step, h_t, c_t, memory, delta_c_list, delta_m_list = \
                self.network(
                    forcings[:, t],
                    net,
                    net_temp,
                    h_t_temp,
                    h_t,
                    c_t,
                    memory,
                    delta_c_list,
                    delta_m_list
                )
            next_frames.append(net)
            decouple_loss += d_loss_step

        decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
        next_frames = torch.stack(next_frames, dim=1)

        dy_pred = next_frames[:, :, :, 1:, :] - next_frames[:, :, :, :-1, :]
        dx_pred = next_frames[:, :, :, :, 1:] - next_frames[:, :, :, :, :-1]
        dy_true = targets[:, :, :, 1:, :] - targets[:, :, :, :-1, :]
        dx_true = targets[:, :, :, :, 1:] - targets[:, :, :, :, :-1]
        grad_loss = torch.mean(torch.abs(dy_pred - dy_true)) + torch.mean(torch.abs(dx_pred - dx_true))

        storage_component = self._compute_storage_component(
            next_frames,
            targets,
            alpha,
            n,
            theta_r,
            theta_s,
            porosity,
            specific_storage,
            mask
        )

        mse_loss = self.network.MSE_criterion(next_frames, targets)
        grad_component = self.configs.grad_beta * grad_loss
        decouple_component = self.configs.decouple_beta * decouple_loss

        total_loss = mse_loss + grad_component + decouple_component + storage_component

        total_loss.backward()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        current_lr = self.optimizer.param_groups[0]["lr"]

        return (
            total_loss.item(),
            mse_loss.item(),
            grad_component.item(),
            decouple_component.item(),
            storage_component.item(),
            current_lr,
        )

    def vg_saturation_torch(self, pressure, alpha, n, theta_r, theta_s):
        m = 1.0 - 1.0 / n
        h_neg = torch.abs(pressure)

        vg_expr = (1.0 + (alpha * h_neg) ** n) ** (-m)
        Se = torch.where(pressure < 0, vg_expr, torch.ones_like(pressure))

        theta = theta_r + (theta_s - theta_r) * Se
        return theta

    def calculate_subsurface_storage_torch(
        self, porosity, pressure, saturation, specific_storage, dx, dy, dz, mask=None
    ):
        if mask is None:
            mask = torch.ones_like(porosity)

        mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))

        if dz.ndim == 1:
            dz = dz.view(1, 1, -1, 1, 1)

        incompressible = porosity * saturation * dz * dx * dy
        compressible = pressure * saturation * specific_storage * dz * dx * dy
        compressible = torch.where(pressure < 0, torch.zeros_like(compressible), compressible)

        total = incompressible + compressible
        total = torch.where(mask == 0, torch.zeros_like(total), total)
        return total

    def calculate_surface_storage_torch(self, pressure, dx, dy, mask=None):
        if mask is None:
            mask = torch.ones_like(pressure)

        mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
        surface_mask = mask[:, :, -1, :, :]

        total = pressure[:, :, -1, :, :] * dx * dy
        total = torch.where(total < 0, torch.zeros_like(total), total)
        total = torch.where(surface_mask == 0, torch.zeros_like(total), total)
        return total

    #### test
    def test(self, forcings, init_cond, static_inputs):
        self.network.eval()
        with torch.no_grad():
            batch, timesteps, channels, height, width = forcings.shape

            next_frames = []
            h_t, c_t, delta_c_list, delta_m_list = self._init_states(batch, height, width)

            memory = self.network.memory_encoder(init_cond[:, 0])
            c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

            net = init_cond[:, 0]
            net_temp = []
            h_t_temp = []

            for t in range(timesteps):
                net, net_temp, h_t_temp, _, h_t, c_t, memory, delta_c_list, delta_m_list = \
                    self.network(
                        forcings[:, t],
                        net,
                        net_temp,
                        h_t_temp,
                        h_t,
                        c_t,
                        memory,
                        delta_c_list,
                        delta_m_list
                    )
                next_frames.append(net)

            next_frames = torch.stack(next_frames, dim=1)

        return next_frames
    
###test_lsm
    def test_lsm(self):

        self.network.eval()

        with torch.no_grad():

            res_path = os.path.join(self.configs.gen_frm_dir, 'lsm_result')
            os.makedirs(res_path, exist_ok=True)

            nx, ny = self.configs.img_width, self.configs.img_height
            patch_size = self.configs.patch_size
            ss_stride = self.configs.ss_stride_test

            ys = list(range(0, ny - patch_size + 1, ss_stride))
            xs = list(range(0, nx - patch_size + 1, ss_stride))

            if ys[-1] != ny - patch_size:
                ys.append(ny - patch_size)

            if xs[-1] != nx - patch_size:
                xs.append(nx - patch_size)

            coords_space = [(y, x) for y in ys for x in xs]

            # static
            static_inputs_name = os.path.join(self.configs.static_inputs_path, self.configs.static_inputs_filename)
            frame_np = read_pfb(get_absolute_path(static_inputs_name)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            mean = frame_im.mean(dim=(3,4), keepdim=True)
            std = frame_im.std(dim=(3,4), keepdim=True)+1e-8
            frame_im = (frame_im-mean)/std
            static_inputs = preprocess.reshape_patch(frame_im, coords_space, patch_size).to(self.configs.device)

            mean_p = torch.tensor(self.configs.target_mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
            std_p = torch.tensor(self.configs.target_std, dtype=torch.float32).view(1, 1, -1, 1, 1)
            mean_a = torch.tensor(self.configs.force_mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
            std_a = torch.tensor(self.configs.force_std, dtype=torch.float32).view(1, 1, -1, 1, 1)

            targets_prefix = os.path.join(self.configs.targets_path, self.configs.pf_runname + ".out.")

            init_cond_name = targets_prefix + 'press.' + str(self.configs.test_start_step - 1).zfill(5) + ".pfb"
            frame_np = read_pfb(get_absolute_path(init_cond_name)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            frame_im = (frame_im-mean_p)/std_p
            init_cond = preprocess.reshape_patch(frame_im, coords_space, patch_size).to(self.configs.device)

            #change forcings to static to get the shape?
            batch, _, channels, height, width = static_inputs.shape

            next_frames = []
            h_t, c_t, delta_c_list, delta_m_list = self._init_states(batch, height, width)

            memory = self.network.memory_encoder(init_cond[:, 0])
            c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))

            net = init_cond[:, 0]
            net_temp = []
            h_t_temp = []

            lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libclm_lsm.so'))
            lib = ctypes.CDLL(lib_path)
            print("CLM shared library loaded successfully.")
            self._set_clm_lsm_c_argtypes(lib)

            # 构造虚拟参数（用实际数据替换）
            nz, clm_nz = 11, 10
            size_2d  = (nx+2) * (ny+2) * 3
            size_3d  = (nx+2) * (ny+2) * (nz+2)
            size_clm = (nx+2) * (ny+2) * (clm_nz+2)

            temp_arr_2d = np.ones(size_2d, dtype=np.float64)
            temp_arr_lh = np.ones(size_2d, dtype=np.float64)
            temp_arr_sh = np.ones(size_2d, dtype=np.float64)
            temp_arr_3d = np.ones(size_3d, dtype=np.float64)
            temp_arr_clm = np.ones(size_clm, dtype=np.float64)

            alpha_filename, n_filename = targets_prefix+'alpha.pfb', targets_prefix+'n.pfb'
            theta_s_filename, theta_r_filename = targets_prefix+'ssat.pfb', targets_prefix+'sres.pfb'
            mask_filename, porosity_filename = targets_prefix+'mask.pfb', targets_prefix+'porosity.pfb'
            pf_dz_mult_filename = targets_prefix+'dz_mult.pfb'

            alpha = read_pfb(alpha_filename)
            n_value = read_pfb(n_filename)
            theta_s = read_pfb(theta_s_filename)
            theta_r = read_pfb(theta_r_filename)

            press_temp = read_pfb(init_cond_name)
            topo = read_pfb(mask_filename)
            porosity = read_pfb(porosity_filename)
            pf_dz_mult = read_pfb(pf_dz_mult_filename)

            topo = np.pad(topo, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0).flatten()
            porosity = np.pad(porosity, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0).flatten()
            pf_dz_mult = np.pad(pf_dz_mult, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0).flatten()

            for t in range(self.configs.test_start_step - 1, self.configs.test_end_step):

                #this t is that in parflow, t+1 is that in colm and it is the real time
                # read 8 forcings, cal saturation
                hour = t % 24
                time1 = str(t // 24 * 24 + 1).zfill(6) 
                time2 = str(t // 24 * 24 + 24).zfill(6) 

                general_path = os.path.join(self.configs.lsm_forcings_path, self.configs.lsm_forcings_name)
                sw_pf_filename = general_path +'.DSWR.' + time1 + '_to_' + time2 + '.pfb'
                lw_pf_filename = general_path +'.DLWR.' + time1 + '_to_' + time2 + '.pfb'
                prcp_pf_filename = general_path +'.APCP.' + time1 + '_to_' + time2 + '.pfb'
                tas_pf_filename = general_path +'.Temp.' + time1 + '_to_' + time2 + '.pfb'
                u_pf_filename = general_path +'.UGRD.' + time1 + '_to_' + time2 + '.pfb'
                v_pf_filename = general_path +'.VGRD.' + time1 + '_to_' + time2 + '.pfb'
                patm_pf_filename = general_path +'.Press.' + time1 + '_to_' + time2 + '.pfb'
                qatm_pf_filename = general_path +'.SPFH.' + time1 + '_to_' + time2 + '.pfb'

                # read forcings and pad dim
                pf_vars = {'sw_pf': sw_pf_filename, 'lw_pf': lw_pf_filename, 'prcp_pf': prcp_pf_filename, 'tas_pf': tas_pf_filename,
                           'u_pf': u_pf_filename, 'v_pf': v_pf_filename, 'patm_pf': patm_pf_filename, 'qatm_pf': qatm_pf_filename}
                variables = {}
                for var_name, filename in pf_vars.items():
                    data = read_pfb(get_absolute_path(filename))[hour, :, :]
                    data = np.expand_dims(data, axis=0)
                    variables[var_name] = data
                sw_pf, lw_pf, prcp_pf, tas_pf = variables['sw_pf'], variables['lw_pf'], variables['prcp_pf'], variables['tas_pf']
                u_pf, v_pf, patm_pf, qatm_pf = variables['u_pf'], variables['v_pf'], variables['patm_pf'], variables['qatm_pf']

                # cal saturation
                satur = self._vg_saturation(press_temp, alpha, n_value, theta_r, theta_s)
                press = np.pad(press_temp, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0).flatten()

                # pad ghost cells and flatten
                variables = {'satur': satur, 'sw_pf': sw_pf, 'lw_pf': lw_pf, 'prcp_pf': prcp_pf,
                             'tas_pf': tas_pf, 'u_pf': u_pf, 'v_pf': v_pf, 'patm_pf': patm_pf, 'qatm_pf': qatm_pf}
                for name, arr in variables.items():
                    arr = np.pad(arr, pad_width=((1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0).flatten()
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
                    t + 1,                                  # istep_pf
                    1.0,                                    # dt
                    t,                                      # time ？？？
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
                    *(temp_arr_2d.ctypes.data_as(POINTER(c_double)) for _ in range(6)),  # 所有 double* 参数
                    temp_arr_lh.ctypes.data_as(POINTER(c_double)),
                    temp_arr_2d.ctypes.data_as(POINTER(c_double)),
                    temp_arr_sh.ctypes.data_as(POINTER(c_double)),
                    *(temp_arr_2d.ctypes.data_as(POINTER(c_double)) for _ in range(9)),
                    temp_arr_clm.ctypes.data_as(POINTER(c_double)),
                    1, 0, 0,                                 # clm_dump_interval####, clm_1d_out, clm_forc_veg
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
                    10,                                      # soi_z ####
                    1, 0, 1, 1,                              # clm_next, clm_write_logs, clm_last_rst, clm_daily_rst
                    10, 10                                   # pf_nlevsoi, pf_nlevlak ####
                )

                evap_trans = np.reshape(temp_arr_3d,(nz+2,ny+2,nx+2)).astype(np.float32)[1:nz+1,1:ny+1,1:nx+1]
                evap_trans = torch.from_numpy(evap_trans).unsqueeze(0).unsqueeze(0)
                evap_trans = ((evap_trans - mean_a) / std_a)[:, :, 1:nz, :, :]
                forcings = preprocess.reshape_patch(evap_trans, coords_space, patch_size).to(self.configs.device)

                heat_lh = np.reshape(temp_arr_lh, (3, ny+2, nx+2))[1, 1:ny+1, 1:nx+1]
                heat_sh = np.reshape(temp_arr_sh, (3, ny+2, nx+2))[1, 1:ny+1, 1:nx+1]
                temp_gt = np.reshape(temp_arr_2d, (3, ny+2, nx+2))[1, 1:ny+1, 1:nx+1]
                trans = np.reshape(temp_arr_3d, (nz+2, ny+2, nx+2))[1:nz+1, 1:ny+1, 1:nx+1]

                write_pfb(os.path.join(res_path, 'heat_lh.' + str(t+1).zfill(5) + '.pfb'), heat_lh, dist=False)
                write_pfb(os.path.join(res_path, 'heat_sh.' + str(t+1).zfill(5) + '.pfb'), heat_sh, dist=False)
                write_pfb(os.path.join(res_path, 'temp_gt.' + str(t+1).zfill(5) + '.pfb'), temp_gt, dist=False)
                write_pfb(os.path.join(res_path, 'evap.' + str(t+1).zfill(5) + '.pfb'), trans, dist=False)

                net, net_temp, h_t_temp, _, h_t, c_t, memory, delta_c_list, delta_m_list = \
                self.network(forcings[:,0], net, net_temp, h_t_temp, h_t, c_t, memory, delta_c_list, delta_m_list)
                next_frames.append(net)

                press_ = preprocess.reshape_patch_back(
                    net.unsqueeze(1),   # [num_patch, 1, C, patch, patch]
                    coords_space,
                    ny,
                    nx,
                    patch_size
                )
                press_ = torch.squeeze((press_.detach().cpu()) * std_p + mean_p).numpy().astype(np.float64)
                assert press_.ndim == 3, f"press_ shape wrong: {press_.shape}"
                assert press_.shape[1] == ny and press_.shape[2] == nx, f"press_ shape wrong: {press_.shape}"
                press_temp = press_

            next_frames = torch.stack(next_frames, dim=1)
            next_frames = preprocess.reshape_patch_back(
                next_frames,
                coords_space,
                ny,
                nx,
                patch_size
            )
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
            c_int, c_int, c_int,            # clm_dump_interval, clm_1d_out, clm_forc_veg
            c_char_p,                       # clm_output_dir
            c_int,                          # clm_output_dir_length
            c_int,                          # clm_bin_output_dir
            c_int, c_int, c_int,            # write_CLM_binary, slope_accounting_CLM, beta_typepf
            c_int,                          # veg_water_stress_typepf
            c_double, c_double, c_double,   # wilting_pointpf, field_capacitypf, res_satpf
            c_int, c_int,                   # irr_typepf, irr_cyclepf
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