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
import shutil
import multiprocessing as mp
import sys
from contextlib import contextmanager

_worker_lib_cache = {}

@contextmanager
def redirect_fd_to_file(log_path, also_stderr=True):
    """
    把当前进程底层 stdout/stderr 临时重定向到文件。
    这对 ctypes 调进去的 C/Fortran printf / write(*,*) 也通常有效。
    """
    sys.stdout.flush()
    sys.stderr.flush()

    stdout_fd = sys.stdout.fileno()
    stderr_fd = sys.stderr.fileno()

    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd) if also_stderr else None

    with open(log_path, "a", buffering=1) as f:
        try:
            os.dup2(f.fileno(), stdout_fd)
            if also_stderr:
                os.dup2(f.fileno(), stderr_fd)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()

            os.dup2(saved_stdout_fd, stdout_fd)
            os.close(saved_stdout_fd)

            if also_stderr and saved_stderr_fd is not None:
                os.dup2(saved_stderr_fd, stderr_fd)
                os.close(saved_stderr_fd)

def get_worker_lib(lib_path):
    global _worker_lib_cache
    lib = _worker_lib_cache.get(lib_path)
    if lib is None:
        lib = ctypes.CDLL(lib_path)
        _set_clm_lsm_c_argtypes_local(lib)
        _worker_lib_cache[lib_path] = lib
    return lib
    
def _set_clm_lsm_c_argtypes_local(lib):
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
        c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        c_int, c_int, c_int, c_int, c_int, c_int, c_int,
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double),
        POINTER(c_double), POINTER(c_double), POINTER(c_double),
        c_int, c_int, c_int,
        c_char_p,
        c_int,
        c_int,
        c_int, c_int, c_int,
        c_int,
        c_double, c_double, c_double,
        c_int, c_int,
        c_double, c_double, c_double, c_double,
        POINTER(c_double), POINTER(c_double), POINTER(c_double),
        c_int,
        c_int,
        c_int, c_int, c_int, c_int,
        c_int, c_int
    ]
    lib.clm_lsm_c.restype = None

def run_block_worker_fixed(block_info, task_queue, result_queue):
    """
    固定 worker：只负责一个 block，常驻进程。
    """
    (
        lib_path, ix, iy, nx_b, ny_b, nz, clm_nz, nx, ny, general_path, worker_id,
        alpha_block, n_block, theta_r_block, theta_s_block,
        topo_block, porosity_block, pf_dz_mult_block
    ) = block_info

    lib_i = get_worker_lib(lib_path)

    size_2d  = (nx_b + 2) * (ny_b + 2) * 3
    size_3d  = (nx_b + 2) * (ny_b + 2) * (nz + 2)
    size_clm = (nx_b + 2) * (ny_b + 2) * (clm_nz + 2)
    temp_arr_2d  = np.ones(size_2d,  dtype=np.float64)
    temp_arr_lh  = np.ones(size_2d,  dtype=np.float64)
    temp_arr_sh  = np.ones(size_2d,  dtype=np.float64)
    temp_arr_3d  = np.ones(size_3d,  dtype=np.float64)
    temp_arr_clm = np.ones(size_clm, dtype=np.float64)


    press_buf = np.zeros((nz + 2, ny_b + 2, nx_b + 2), dtype=np.float64)
    satur_buf = np.zeros((nz + 2, ny_b + 2, nx_b + 2), dtype=np.float64)

    sw_buf    = np.zeros((3, ny_b + 2, nx_b + 2), dtype=np.float64)
    lw_buf    = np.zeros((3, ny_b + 2, nx_b + 2), dtype=np.float64)
    prcp_buf  = np.zeros((3, ny_b + 2, nx_b + 2), dtype=np.float64)
    tas_buf   = np.zeros((3, ny_b + 2, nx_b + 2), dtype=np.float64)
    u_buf     = np.zeros((3, ny_b + 2, nx_b + 2), dtype=np.float64)
    v_buf     = np.zeros((3, ny_b + 2, nx_b + 2), dtype=np.float64)
    patm_buf  = np.zeros((3, ny_b + 2, nx_b + 2), dtype=np.float64)
    qatm_buf  = np.zeros((3, ny_b + 2, nx_b + 2), dtype=np.float64)

    outdir = f"./output_block_{ix}_{iy}/"
    os.makedirs(outdir, exist_ok=True)
    outdir_b = outdir.encode()

    fortran_log = os.path.join(outdir, f"fortran_block_{ix}_{iy}.log")
    open(fortran_log, "w").close()

    cached_day_key = None
    cached_forcing = None

    topo = np.pad(
        topo_block,
        pad_width=((1, 1), (1, 1), (1, 1)),
        mode='constant',
        constant_values=0
    ).astype(np.float64, copy=False).ravel()

    porosity = np.pad(
        porosity_block,
        pad_width=((1, 1), (1, 1), (1, 1)),
        mode='constant',
        constant_values=0
    ).astype(np.float64, copy=False).ravel()

    pf_dz_mult = np.pad(
        pf_dz_mult_block,
        pad_width=((1, 1), (1, 1), (1, 1)),
        mode='constant',
        constant_values=0
    ).astype(np.float64, copy=False).ravel()

    with redirect_fd_to_file(fortran_log, also_stderr=True):
        while True:
            task = task_queue.get()
            if task is None:
                break
                
            temp_arr_2d.fill(1.0)
            temp_arr_lh.fill(1.0)
            temp_arr_sh.fill(1.0)
            temp_arr_3d.fill(1.0)
            temp_arr_clm.fill(1.0)
    
            (
                t, hour, start_time_pf,
                press_block
            ) = task
    
            day_key = t // 24
    
            if cached_day_key != day_key:
                time1 = str(day_key * 24 + 1).zfill(6)
                time2 = str(day_key * 24 + 24).zfill(6)
    
                sw_pf_filename   = general_path + '.DSWR.'  + time1 + '_to_' + time2 + '.pfb'
                lw_pf_filename   = general_path + '.DLWR.'  + time1 + '_to_' + time2 + '.pfb'
                prcp_pf_filename = general_path + '.APCP.'  + time1 + '_to_' + time2 + '.pfb'
                tas_pf_filename  = general_path + '.Temp.'  + time1 + '_to_' + time2 + '.pfb'
                u_pf_filename    = general_path + '.UGRD.'  + time1 + '_to_' + time2 + '.pfb'
                v_pf_filename    = general_path + '.VGRD.'  + time1 + '_to_' + time2 + '.pfb'
                patm_pf_filename = general_path + '.Press.' + time1 + '_to_' + time2 + '.pfb'
                qatm_pf_filename = general_path + '.SPFH.'  + time1 + '_to_' + time2 + '.pfb'
    
                keys_day = {
                    "x": {"start": ix, "stop": ix + nx_b},
                    "y": {"start": iy, "stop": iy + ny_b},
                    "z": {"start": 0, "stop": 24},
                }
    
                cached_forcing = {
                    "sw": np.ascontiguousarray(
                        read_pfb(get_absolute_path(sw_pf_filename), keys=keys_day), dtype=np.float64
                    ),
                    "lw": np.ascontiguousarray(
                        read_pfb(get_absolute_path(lw_pf_filename), keys=keys_day), dtype=np.float64
                    ),
                    "prcp": np.ascontiguousarray(
                        read_pfb(get_absolute_path(prcp_pf_filename), keys=keys_day), dtype=np.float64
                    ),
                    "tas": np.ascontiguousarray(
                        read_pfb(get_absolute_path(tas_pf_filename), keys=keys_day), dtype=np.float64
                    ),
                    "u": np.ascontiguousarray(
                        read_pfb(get_absolute_path(u_pf_filename), keys=keys_day), dtype=np.float64
                    ),
                    "v": np.ascontiguousarray(
                        read_pfb(get_absolute_path(v_pf_filename), keys=keys_day), dtype=np.float64
                    ),
                    "patm": np.ascontiguousarray(
                        read_pfb(get_absolute_path(patm_pf_filename), keys=keys_day), dtype=np.float64
                    ),
                    "qatm": np.ascontiguousarray(
                        read_pfb(get_absolute_path(qatm_pf_filename), keys=keys_day), dtype=np.float64
                    ),
                }
    
                cached_day_key = day_key

            # 按当前 hour 取出一个小时
            sw_pf_block   = cached_forcing["sw"][hour:hour+1, :, :]
            lw_pf_block   = cached_forcing["lw"][hour:hour+1, :, :]
            prcp_pf_block = cached_forcing["prcp"][hour:hour+1, :, :]
            tas_pf_block  = cached_forcing["tas"][hour:hour+1, :, :]
            u_pf_block    = cached_forcing["u"][hour:hour+1, :, :]
            v_pf_block    = cached_forcing["v"][hour:hour+1, :, :]
            patm_pf_block = cached_forcing["patm"][hour:hour+1, :, :]
            qatm_pf_block = cached_forcing["qatm"][hour:hour+1, :, :]
            
            m = 1.0 - 1.0 / n_block
            h_neg = np.abs(press_block)
            vg_expr = (1.0 + (alpha_block * h_neg) ** n_block) ** (-m)
            Se = np.where(press_block < 0.0, vg_expr, 1.0)
            satur_block = theta_r_block + (theta_s_block - theta_r_block) * Se
    
            press_buf.fill(0.0)
            press_buf[1:nz+1, 1:ny_b+1, 1:nx_b+1] = press_block
            press = press_buf.ravel()

            satur_buf.fill(0.0)
            satur_buf[1:nz+1, 1:ny_b+1, 1:nx_b+1] = satur_block
            satur = satur_buf.ravel()

            sw_buf.fill(0.0)
            sw_buf[1:2, 1:ny_b+1, 1:nx_b+1] = sw_pf_block
            sw_pf = sw_buf.ravel()

            lw_buf.fill(0.0)
            lw_buf[1:2, 1:ny_b+1, 1:nx_b+1] = lw_pf_block
            lw_pf = lw_buf.ravel()

            prcp_buf.fill(0.0)
            prcp_buf[1:2, 1:ny_b+1, 1:nx_b+1] = prcp_pf_block
            prcp_pf = prcp_buf.ravel()

            tas_buf.fill(0.0)
            tas_buf[1:2, 1:ny_b+1, 1:nx_b+1] = tas_pf_block
            tas_pf = tas_buf.ravel()

            u_buf.fill(0.0)
            u_buf[1:2, 1:ny_b+1, 1:nx_b+1] = u_pf_block
            u_pf = u_buf.ravel()

            v_buf.fill(0.0)
            v_buf[1:2, 1:ny_b+1, 1:nx_b+1] = v_pf_block
            v_pf = v_buf.ravel()

            patm_buf.fill(0.0)
            patm_buf[1:2, 1:ny_b+1, 1:nx_b+1] = patm_pf_block
            patm_pf = patm_buf.ravel()

            qatm_buf.fill(0.0)
            qatm_buf[1:2, 1:ny_b+1, 1:nx_b+1] = qatm_pf_block
            qatm_pf = qatm_buf.ravel()
    
            try:
                lib_i.clm_lsm_c(
                    press.ctypes.data_as(POINTER(c_double)),
                    satur.ctypes.data_as(POINTER(c_double)),
                    temp_arr_3d.ctypes.data_as(POINTER(c_double)),
                    topo.ctypes.data_as(POINTER(c_double)),
                    porosity.ctypes.data_as(POINTER(c_double)),
                    pf_dz_mult.ctypes.data_as(POINTER(c_double)),
                    t + 1,
                    1.0,
                    float(t),
                    start_time_pf,
                    961.72,
                    961.72,
                    200.0,
                    ix, iy, nx_b, ny_b, nz, nx_b + 2, ny_b + 2, nz + 2, 0,
                    0, 0, 0, 0, nx, ny, worker_id,
                    sw_pf.ctypes.data_as(POINTER(c_double)),
                    lw_pf.ctypes.data_as(POINTER(c_double)),
                    prcp_pf.ctypes.data_as(POINTER(c_double)),
                    tas_pf.ctypes.data_as(POINTER(c_double)),
                    u_pf.ctypes.data_as(POINTER(c_double)),
                    v_pf.ctypes.data_as(POINTER(c_double)),
                    patm_pf.ctypes.data_as(POINTER(c_double)),
                    qatm_pf.ctypes.data_as(POINTER(c_double)),
                    *(temp_arr_2d.ctypes.data_as(POINTER(c_double)) for _ in range(6)),
                    temp_arr_lh.ctypes.data_as(POINTER(c_double)),
                    temp_arr_2d.ctypes.data_as(POINTER(c_double)),
                    temp_arr_sh.ctypes.data_as(POINTER(c_double)),
                    *(temp_arr_2d.ctypes.data_as(POINTER(c_double)) for _ in range(9)),
                    temp_arr_clm.ctypes.data_as(POINTER(c_double)),
                    1, 0, 0,
                    outdir_b, len(outdir_b),
                    1,
                    1, 0, 1,
                    2,
                    0.2,
                    1.0,
                    0.2,
                    0, 0,
                    0.0,
                    12.0,
                    20.0,
                    0.5,
                    temp_arr_2d.ctypes.data_as(POINTER(c_double)),
                    temp_arr_clm.ctypes.data_as(POINTER(c_double)),
                    temp_arr_2d.ctypes.data_as(POINTER(c_double)),
                    2,
                    10,
                    1, 0, 1, 1,
                    10, 10
                )
    
                evap_trans = np.reshape(
                    temp_arr_3d, (nz + 2, ny_b + 2, nx_b + 2)
                ).astype(np.float32)[1:nz+1, 1:ny_b+1, 1:nx_b+1]
    
                heat_lh = np.reshape(
                    temp_arr_lh, (3, ny_b + 2, nx_b + 2)
                )[1, 1:ny_b+1, 1:nx_b+1]
    
                heat_sh = np.reshape(
                    temp_arr_sh, (3, ny_b + 2, nx_b + 2)
                )[1, 1:ny_b+1, 1:nx_b+1]
    
                temp_gt = np.reshape(
                    temp_arr_2d, (3, ny_b + 2, nx_b + 2)
                )[1, 1:ny_b+1, 1:nx_b+1]
    
                trans = np.reshape(
                    temp_arr_3d, (nz + 2, ny_b + 2, nx_b + 2)
                )[1:nz+1, 1:ny_b+1, 1:nx_b+1]
    
                result_queue.put((
                    ix, iy, nx_b, ny_b,
                    evap_trans, heat_lh, heat_sh, temp_gt, trans
                ))
    
            except Exception as e:
                result_queue.put(("ERROR", ix, iy, t, repr(e)))
                break

def start_fixed_block_workers(
    blocks, lib_paths, nz, clm_nz, nx, ny, general_path, mp_ctx,
    alpha_global, n_value_global, theta_r_global, theta_s_global,
    topo_global, porosity_global, pf_dz_mult_global
):
    workers = []
    task_queues = []
    result_queues = []

    for worker_id, ((ix, iy, nx_b, ny_b), lib_path) in enumerate(zip(blocks, lib_paths)):
        task_q = mp_ctx.Queue(maxsize=1)
        result_q = mp_ctx.Queue(maxsize=1)

        block_info = (
            lib_path, ix, iy, nx_b, ny_b, nz, clm_nz, nx, ny, general_path, worker_id,
            np.ascontiguousarray(alpha_global[:, iy:iy+ny_b, ix:ix+nx_b], dtype=np.float64),
            np.ascontiguousarray(n_value_global[:, iy:iy+ny_b, ix:ix+nx_b], dtype=np.float64),
            np.ascontiguousarray(theta_r_global[:, iy:iy+ny_b, ix:ix+nx_b], dtype=np.float64),
            np.ascontiguousarray(theta_s_global[:, iy:iy+ny_b, ix:ix+nx_b], dtype=np.float64),
            np.ascontiguousarray(topo_global[:, iy:iy+ny_b, ix:ix+nx_b], dtype=np.float64),
            np.ascontiguousarray(porosity_global[:, iy:iy+ny_b, ix:ix+nx_b], dtype=np.float64),
            np.ascontiguousarray(pf_dz_mult_global[:, iy:iy+ny_b, ix:ix+nx_b], dtype=np.float64),
        )

        p = mp_ctx.Process(
            target=run_block_worker_fixed,
            args=(block_info, task_q, result_q)
        )
        p.start()

        workers.append(p)
        task_queues.append(task_q)
        result_queues.append(result_q)

    return workers, task_queues, result_queues

def stop_fixed_block_workers(workers, task_queues):
    for q in task_queues:
        q.put(None)

    for p in workers:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
            p.join()

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
            std = frame_im.std(dim=(3,4), keepdim=True) + 1e-8
            frame_im = (frame_im - mean) / std
            static_inputs = preprocess.reshape_patch(frame_im, coords_space, patch_size).to(self.configs.device)
    
            mean_p = torch.tensor(self.configs.target_mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
            std_p = torch.tensor(self.configs.target_std, dtype=torch.float32).view(1, 1, -1, 1, 1)
            mean_a = torch.tensor(self.configs.force_mean, dtype=torch.float32).view(1, 1, -1, 1, 1)
            std_a = torch.tensor(self.configs.force_std, dtype=torch.float32).view(1, 1, -1, 1, 1)
    
            targets_prefix = os.path.join(self.configs.targets_path, self.configs.pf_runname + ".out.")
    
            init_cond_name = targets_prefix + 'press.' + str(self.configs.test_start_step - 1).zfill(5) + ".pfb"
            frame_np = read_pfb(get_absolute_path(init_cond_name)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            frame_im = (frame_im - mean_p) / std_p
            init_cond = preprocess.reshape_patch(frame_im, coords_space, patch_size).to(self.configs.device)
    
            batch, _, channels, height, width = static_inputs.shape
    
            next_frames = []
            h_t, c_t, delta_c_list, delta_m_list = self._init_states(batch, height, width)
    
            memory = self.network.memory_encoder(init_cond[:, 0])
            c_t = list(torch.split(self.network.cell_encoder(static_inputs[:, 0]), self.num_hidden, dim=1))
    
            net = init_cond[:, 0]
            net_temp = []
            h_t_temp = []
    
            blocks = [
                (0,   0, 63, 49),
                (63,  0, 63, 49),
                (126, 0, 63, 49),
                (189, 0, 63, 49),
    
                (0,   49, 63, 49),
                (63,  49, 63, 49),
                (126, 49, 63, 49),
                (189, 49, 63, 49),
    
                (0,   98, 63, 48),
                (63,  98, 63, 48),
                (126, 98, 63, 48),
                (189, 98, 63, 48),
            ]

            base_lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libclm_lsm.so'))
    
            lib_paths = []
            for i in range(len(blocks)):
                new_path = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), f'libclm_lsm_block_{i:02d}.so')
                )
                if not os.path.exists(new_path):
                    shutil.copyfile(base_lib_path, new_path)
                lib_paths.append(new_path)
    
            print(f"CLM shared libraries prepared successfully: {len(lib_paths)} instances.")
    
            alpha_filename = targets_prefix + 'alpha.pfb'
            n_filename = targets_prefix + 'n.pfb'
            theta_s_filename = targets_prefix + 'ssat.pfb'
            theta_r_filename = targets_prefix + 'sres.pfb'
            mask_filename = targets_prefix + 'mask.pfb'
            porosity_filename = targets_prefix + 'porosity.pfb'
            pf_dz_mult_filename = targets_prefix + 'dz_mult.pfb'
    
            alpha_global = np.ascontiguousarray(read_pfb(alpha_filename), dtype=np.float64)
            n_value_global = np.ascontiguousarray(read_pfb(n_filename), dtype=np.float64)
            theta_s_global = np.ascontiguousarray(read_pfb(theta_s_filename), dtype=np.float64)
            theta_r_global = np.ascontiguousarray(read_pfb(theta_r_filename), dtype=np.float64)
    
            press_temp_global = np.ascontiguousarray(read_pfb(init_cond_name), dtype=np.float64)
            topo_global = np.ascontiguousarray(read_pfb(mask_filename), dtype=np.float64)
            porosity_global = np.ascontiguousarray(read_pfb(porosity_filename), dtype=np.float64)
            pf_dz_mult_global = np.ascontiguousarray(read_pfb(pf_dz_mult_filename), dtype=np.float64)
    
            nz, clm_nz = 11, 10
            evap_trans_global = np.zeros((1, 1, nz - 1, ny, nx), dtype=np.float32)
            heat_lh = np.zeros((ny, nx), dtype=np.float64)
            heat_sh = np.zeros((ny, nx), dtype=np.float64)
            temp_gt = np.zeros((ny, nx), dtype=np.float64)
            trans = np.zeros((nz, ny, nx), dtype=np.float64)
    
            mp_ctx = mp.get_context("spawn")

            general_path = os.path.join(self.configs.lsm_forcings_path, self.configs.lsm_forcings_name)

            workers, task_queues, result_queues = start_fixed_block_workers(
                blocks, lib_paths, nz, clm_nz, nx, ny, general_path, mp_ctx,
                alpha_global, n_value_global, theta_r_global, theta_s_global,
                topo_global, porosity_global, pf_dz_mult_global
            )
            
            try:
                for t in range(self.configs.test_start_step - 1, self.configs.test_end_step):
                    hour = t % 24
            
                    evap_trans_global.fill(0.0)
                    heat_lh.fill(0.0)
                    heat_sh.fill(0.0)
                    temp_gt.fill(0.0)
                    trans.fill(0.0)
            
                    # 发任务：每个 block 发给它固定的 worker
                    for q, (ix, iy, nx_b, ny_b) in zip(task_queues, blocks):
                        q.put((
                            t,
                            hour,
                            float(self.configs.test_start_step - 1),
                            np.ascontiguousarray(press_temp_global[:, iy:iy+ny_b, ix:ix+nx_b], dtype=np.float64),
                        ))
            
                    # 收结果
                    results = []
                    for rq in result_queues:
                        item = rq.get()
                        if isinstance(item, tuple) and len(item) > 0 and item[0] == "ERROR":
                            _, ix, iy, tt, msg = item
                            raise RuntimeError(f"worker failed at block=({ix},{iy}) t={tt}: {msg}")
                        results.append(item)
            
                    for ix, iy, nx_b, ny_b, evap_trans, lh_blk, sh_blk, tg_blk, trans_blk in results:
                        evap_trans_torch = torch.from_numpy(evap_trans).unsqueeze(0).unsqueeze(0)
                        evap_trans_torch = ((evap_trans_torch - mean_a) / std_a)[:, :, 1:nz, :, :]
                        evap_trans_global[:, :, :, iy:iy+ny_b, ix:ix+nx_b] = evap_trans_torch.numpy()
            
                        heat_lh[iy:iy+ny_b, ix:ix+nx_b] = lh_blk
                        heat_sh[iy:iy+ny_b, ix:ix+nx_b] = sh_blk
                        temp_gt[iy:iy+ny_b, ix:ix+nx_b] = tg_blk
                        trans[:, iy:iy+ny_b, ix:ix+nx_b] = trans_blk
            
                    evap_trans_tensor = torch.from_numpy(evap_trans_global)
                    forcings = preprocess.reshape_patch(evap_trans_tensor, coords_space, patch_size).to(self.configs.device)
            
                    # write_pfb(os.path.join(res_path, 'heat_lh.' + str(t+1).zfill(5) + '.pfb'), heat_lh, dist=False)
                    # write_pfb(os.path.join(res_path, 'heat_sh.' + str(t+1).zfill(5) + '.pfb'), heat_sh, dist=False)
                    # write_pfb(os.path.join(res_path, 'temp_gt.' + str(t+1).zfill(5) + '.pfb'), temp_gt, dist=False)
                    # write_pfb(os.path.join(res_path, 'evap.' + str(t+1).zfill(5) + '.pfb'), trans, dist=False)
            
                    net, net_temp, h_t_temp, _, h_t, c_t, memory, delta_c_list, delta_m_list = \
                        self.network(forcings[:, 0], net, net_temp, h_t_temp, h_t, c_t, memory, delta_c_list, delta_m_list)
                    next_frames.append(net)
            
                    press_ = preprocess.reshape_patch_back(
                        net.unsqueeze(1),
                        coords_space,
                        ny,
                        nx,
                        patch_size
                    )
                    press_ = torch.squeeze((press_.detach().cpu()) * std_p + mean_p).numpy().astype(np.float64)
                    assert press_.ndim == 3, f"press_ shape wrong: {press_.shape}"
                    assert press_.shape[1] == ny and press_.shape[2] == nx, f"press_ shape wrong: {press_.shape}"
                    press_temp_global = np.ascontiguousarray(press_, dtype=np.float64)

                    if ((t + 1) % 24 == 0) or (t == self.configs.test_start_step - 1):
                        print(f"[LSM] finished timestep {t+1}/{self.configs.test_end_step}", flush=True)
            
            finally:
                stop_fixed_block_workers(workers, task_queues)
    
            next_frames = torch.stack(next_frames, dim=1)
            next_frames = preprocess.reshape_patch_back(
                next_frames,
                coords_space,
                ny,
                nx,
                patch_size
            )
    
        return next_frames, mean_p, std_p