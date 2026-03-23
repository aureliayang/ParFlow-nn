__author__ = 'chen yang'
import os
import numpy as np
import torch
import logging
from core.utils import preprocess
from parflow.tools.fs import get_absolute_path
from parflow.tools.io import read_pfb

logger = logging.getLogger(__name__)

def maybe_to_device(x, device):
    return x.to(device, non_blocking=True) if x is not None else None

def numel_or_zero(x):
    return x.numel() if x is not None else 0

class InputHandle:
    def __init__(
        self,
        init_cond,
        static_inputs,
        forcings,
        targets,
        total_seq,
        configs,
        alpha,
        n_value,
        theta_r,
        theta_s,
        porosity,
        specific_storage,
        mask,
        mode,
        coords_space=None,
        num_patch=None,
    ):
        self.configs = configs
        self.name = configs.pf_runname
        self.batch_size = configs.batch_size if mode == 'train' else total_seq
        self.init_cond = init_cond
        self.static_inputs = static_inputs
        self.forcings = forcings
        self.targets = targets
        self.total_seq = total_seq
        self.current_p = 0

        self.alpha = alpha
        self.n_value = n_value
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.porosity = porosity
        self.specific_storage = specific_storage
        self.mask = mask

        self.coords_space = coords_space
        self.num_patch = num_patch

        self.use_storage_terms = self.alpha is not None

    def total(self):
        return self.total_seq

    def begin(self, do_shuffle=True):
        logger.info("Initialization for read data")
        if do_shuffle:
            rand_perm = torch.randperm(self.total_seq)

            self.init_cond = self.init_cond[rand_perm]
            # self.static_inputs = self.static_inputs[rand_perm]
            self.forcings = self.forcings[rand_perm]
            self.targets = self.targets[rand_perm]

            rand_perm_static = rand_perm.to(self.static_inputs.device)
            self.static_inputs = self.static_inputs[rand_perm_static]

            if self.use_storage_terms:
                self.alpha = self.alpha[rand_perm]
                self.n_value = self.n_value[rand_perm]
                self.theta_r = self.theta_r[rand_perm]
                self.theta_s = self.theta_s[rand_perm]
                self.porosity = self.porosity[rand_perm]
                self.specific_storage = self.specific_storage[rand_perm]
                self.mask = self.mask[rand_perm]

        self.current_p = 0

    def next(self):
        self.current_p += self.batch_size
        if self.no_batch_left():
            return None

    def no_batch_left(self):
        return self.current_p + self.batch_size > self.total()

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in "
                + self.name
                + ". Consider to use iterators.begin() to rescan from the beginning of the iterators"
            )
            return None

        slc = slice(self.current_p, self.current_p + self.batch_size)

        init_cond_batch = self.init_cond[slc, :, :, :, :]
        static_inputs_batch = self.static_inputs[slc, :, :, :, :]
        forcings_batch = self.forcings[slc, :, :, :, :]
        targets_batch = self.targets[slc, :, :, :, :]

        if not self.use_storage_terms:
            return (
                forcings_batch.to(self.configs.device, non_blocking=True),
                init_cond_batch.to(self.configs.device, non_blocking=True),
                static_inputs_batch,
                targets_batch.to(self.configs.device, non_blocking=True),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        alpha_batch = self.alpha[slc]
        n_batch = self.n_value[slc]
        theta_r_batch = self.theta_r[slc]
        theta_s_batch = self.theta_s[slc]
        porosity_batch = self.porosity[slc]
        specific_storage_batch = self.specific_storage[slc]
        mask_batch = self.mask[slc]

        return (
            forcings_batch.to(self.configs.device, non_blocking=True),
            init_cond_batch.to(self.configs.device, non_blocking=True),
            static_inputs_batch,
            targets_batch.to(self.configs.device, non_blocking=True),
            alpha_batch.to(self.configs.device, non_blocking=True),
            n_batch.to(self.configs.device, non_blocking=True),
            theta_r_batch.to(self.configs.device, non_blocking=True),
            theta_s_batch.to(self.configs.device, non_blocking=True),
            porosity_batch.to(self.configs.device, non_blocking=True),
            specific_storage_batch.to(self.configs.device, non_blocking=True),
            mask_batch.to(self.configs.device, non_blocking=True),
        )

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_p))
        logger.info("    Batch Size: " + str(self.batch_size))
        logger.info("    total Size: " + str(self.total()))
        

class DataProcess:
    def __init__(self, configs):
        self.input_param = configs

        # self.static_inputs_path = os.path.join(
        #     configs.static_inputs_path,
        #     configs.static_inputs_filename
        # )

        # forcings_filename = configs.pf_runname + ".out.evaptrans."
        # self.forcings_path = os.path.join(configs.forcings_path, forcings_filename)

        # self.output_prefix = os.path.join(
        #     configs.targets_path,
        #     configs.pf_runname + ".out."
        # )

        self.static_inputs_path = os.path.join(
            configs.static_inputs_path,
            configs.static_inputs_filename
        )

        forcings_filename = configs.pf_runname + ".out.evaptrans."

        forcing_dirs = configs.forcings_paths if len(configs.forcings_paths) > 0 else [configs.forcings_path]
        target_dirs  = configs.targets_paths  if len(configs.targets_paths)  > 0 else [configs.targets_path]

        self.forcings_prefixes = [
            os.path.join(p, forcings_filename) for p in forcing_dirs
        ]

        self.output_prefixes = [
            os.path.join(p, configs.pf_runname + ".out.") for p in target_dirs
        ]

        self.output_prefix_ref = self.output_prefixes[0]

        self.target_mean_list = configs.target_mean
        self.target_std_list = configs.target_std
        self.force_mean_list = configs.force_mean
        self.force_std_list = configs.force_std

        self.input_length_train = configs.input_length_train
        self.input_length_test = configs.input_length_test

        self.img_height = configs.img_height
        self.img_width = configs.img_width
        self.patch_size = configs.patch_size

        self.init_cond_channel = configs.init_cond_channel
        self.static_channel = configs.static_channel
        self.act_channel = configs.act_channel
        self.img_channel = configs.img_channel

    def collect_time_files(self, prefixes, varname="", start_idx=0, skip_first_of_later_dirs=False):
        """
        从多个目录收集时间序列文件，并按目录顺序拼接。

        参数
        ----
        prefixes : list[str]
            每个目录对应的文件前缀
        varname : str
            target 用 "press."，forcing 用 ""
        start_idx : int
            每个目录默认从哪个编号开始读
            forcing 应该从 1 开始
            press 应该从 0 开始
        skip_first_of_later_dirs : bool
            是否在后续目录中跳过第一个时间步
            press=True，因为第二年 0 与上一年最后一帧重复
            forcing=False，因为第二年 1 不重复
        """
        files = []

        for i_prefix, prefix in enumerate(prefixes):
            t = start_idx

            if i_prefix > 0 and skip_first_of_later_dirs:
                t += 1

            while True:
                fname = prefix + varname + str(t).zfill(5) + ".pfb"
                fname_abs = get_absolute_path(fname)

                if not os.path.exists(fname_abs):
                    break

                files.append(fname_abs)
                t += 1

        return files



    def load_data(self, mode='train'):
        if mode == 'train':
            start_step = self.input_param.training_start_step
            end_step = self.input_param.training_end_step
            ss_stride = self.input_param.ss_stride_train
            st_stride = self.input_param.st_stride_train
            input_length = self.input_length_train
        else:
            start_step = self.input_param.test_start_step
            end_step = self.input_param.test_end_step
            ss_stride = self.input_param.ss_stride_test
            st_stride = self.input_param.st_stride_test
            input_length = self.input_length_test

        use_storage_terms = abs(float(self.input_param.storage_beta)) > 1e-12

        # forcing_files = self.collect_time_files(self.forcings_prefixes, varname="")
        # target_files = self.collect_time_files(self.output_prefixes, varname="press.")

        forcing_files = self.collect_time_files(
            self.forcings_prefixes,
            varname="",
            start_idx=1,
            skip_first_of_later_dirs=False,
        )

        target_files = self.collect_time_files(
            self.output_prefixes,
            varname="press.",
            start_idx=0,
            skip_first_of_later_dirs=True,
        )

        if len(forcing_files) == 0:
            raise ValueError("No forcing files found in forcings_paths / forcings_path")

        if len(target_files) == 0:
            raise ValueError("No target files found in targets_paths / targets_path")

        total_steps = min(len(forcing_files), len(target_files))

        print(f"[load_data] total forcing files = {len(forcing_files)}")
        print(f"[load_data] total target files  = {len(target_files)}")
        print(f"[load_data] usable total steps  = {total_steps}")

        ys = list(range(0, self.img_height - self.patch_size + 1, ss_stride))
        xs = list(range(0, self.img_width - self.patch_size + 1, ss_stride))

        if ys[-1] != self.img_height - self.patch_size:
            ys.append(self.img_height - self.patch_size)

        if xs[-1] != self.img_width - self.patch_size:
            xs.append(self.img_width - self.patch_size)

        coords_space = [(y, x) for y in ys for x in xs]

        if end_step >= total_steps:
            raise ValueError(
                f"end_step={end_step} exceeds available total_steps={total_steps}. "
                f"Please reduce training_end_step/test_end_step."
            )

        coords_time = [
            start_t for start_t in range(start_step, end_step - input_length + 2, st_stride)
        ]

        num_patch, num_seq = len(coords_space), len(coords_time)

        init_cond = torch.empty(
            (num_patch * num_seq, 1, self.init_cond_channel, self.patch_size, self.patch_size),
            dtype=torch.float32
        )

        forcings = torch.empty(
            (num_patch * num_seq, input_length, self.act_channel, self.patch_size, self.patch_size),
            dtype=torch.float32
        )

        targets = torch.empty(
            (num_patch * num_seq, input_length, self.img_channel, self.patch_size, self.patch_size),
            dtype=torch.float32
        )

        # static inputs
        frame_np = read_pfb(get_absolute_path(self.static_inputs_path)).astype(np.float32)
        frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
        mean = frame_im.mean(dim=(3, 4), keepdim=True)
        std = frame_im.std(dim=(3, 4), keepdim=True) + 1e-8
        frame_im = (frame_im - mean) / std
        static_inputs_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

        # storage-related static parameters
        if use_storage_terms:
            alpha_filename = self.output_prefix_ref + 'alpha.pfb'
            n_filename = self.output_prefix_ref + 'n.pfb'
            theta_s_filename = self.output_prefix_ref + 'ssat.pfb'
            theta_r_filename = self.output_prefix_ref + 'sres.pfb'
            porosity_filename = self.output_prefix_ref + 'porosity.pfb'
            mask_filename = self.output_prefix_ref + 'mask.pfb'
            specific_storage_filename = self.output_prefix_ref + 'specific_storage.pfb'

            frame_np = read_pfb(get_absolute_path(alpha_filename)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            alpha_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

            frame_np = read_pfb(get_absolute_path(n_filename)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            n_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

            frame_np = read_pfb(get_absolute_path(theta_r_filename)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            theta_r_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

            frame_np = read_pfb(get_absolute_path(theta_s_filename)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            theta_s_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

            frame_np = read_pfb(get_absolute_path(porosity_filename)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            porosity_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

            frame_np = read_pfb(get_absolute_path(specific_storage_filename)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            specific_storage_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)

            frame_np = read_pfb(get_absolute_path(mask_filename)).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            mask_temp = preprocess.reshape_patch(frame_im, coords_space, self.patch_size)
        else:
            alpha_temp = None
            n_temp = None
            theta_r_temp = None
            theta_s_temp = None
            porosity_temp = None
            specific_storage_temp = None
            mask_temp = None

        mean_p = torch.tensor(self.target_mean_list, dtype=torch.float32).view(1, 1, -1, 1, 1)
        std_p = torch.tensor(self.target_std_list, dtype=torch.float32).view(1, 1, -1, 1, 1)
        mean_a = torch.tensor(self.force_mean_list, dtype=torch.float32).view(1, 1, -1, 1, 1)
        std_a = torch.tensor(self.force_std_list, dtype=torch.float32).view(1, 1, -1, 1, 1)

        for idx_t, start_t in enumerate(coords_time):
            # print(f"[load_data] seq {idx_t+1}/{num_seq}, start_t={start_t}", flush=True)
            forcing_start = start_t
            forcing_end   = start_t + input_length - 1
            init_idx      = start_t - 1

            print(
                f"[seq {idx_t+1}/{num_seq}] "
                f"forcing: {forcing_start}->{forcing_end}, "
                f"target: {forcing_start}->{forcing_end}, "
                f"init: {init_idx}",
                flush=True
            )
            for i in range(input_length):
                # forcings_name = self.forcings_path + str(i + start_t).zfill(5) + ".pfb"
                # frame_np = read_pfb(get_absolute_path(forcings_name)).astype(np.float32)
                forcing_idx = i + start_t
                forcings_name = forcing_files[forcing_idx]
                frame_np = read_pfb(forcings_name).astype(np.float32)
                frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
                frame_im = (frame_im - mean_a) / std_a

                for idx_s, (y, x) in enumerate(coords_space):
                    forcings[idx_t * num_patch + idx_s: idx_t * num_patch + idx_s + 1, i:i + 1, :, :, :] = \
                        frame_im[:, :, 1:11, y:y + self.patch_size, x:x + self.patch_size]

                # targets_name = self.output_prefix + "press." + str(i + start_t).zfill(5) + ".pfb"
                # frame_np = read_pfb(get_absolute_path(targets_name)).astype(np.float32)
                target_idx = i + start_t
                targets_name = target_files[target_idx]
                frame_np = read_pfb(targets_name).astype(np.float32)
                frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
                frame_im = (frame_im - mean_p) / std_p

                for idx_s, (y, x) in enumerate(coords_space):
                    targets[idx_t * num_patch + idx_s: idx_t * num_patch + idx_s + 1, i:i + 1, :, :, :] = \
                        frame_im[:, :, :, y:y + self.patch_size, x:x + self.patch_size]

            # init_cond_name = self.output_prefix + "press." + str(start_t - 1).zfill(5) +
            init_idx = start_t - 1
            if init_idx < 0:
                raise ValueError("start_t must be >= 1 because init_cond uses start_t - 1")

            init_cond_name = target_files[init_idx]
            frame_np = read_pfb(init_cond_name).astype(np.float32)
            frame_im = torch.from_numpy(frame_np).unsqueeze(0).unsqueeze(0)
            frame_im = (frame_im - mean_p) / std_p

            for idx_s, (y, x) in enumerate(coords_space):
                init_cond[idx_t * num_patch + idx_s: idx_t * num_patch + idx_s + 1, 0, :, :, :] = \
                    frame_im[:, :, :, y:y + self.patch_size, x:x + self.patch_size]

        static_inputs = static_inputs_temp.repeat(num_seq, 1, 1, 1, 1)
        alpha = alpha_temp.repeat(num_seq, 1, 1, 1, 1) if alpha_temp is not None else None
        n_value = n_temp.repeat(num_seq, 1, 1, 1, 1) if n_temp is not None else None
        theta_r = theta_r_temp.repeat(num_seq, 1, 1, 1, 1) if theta_r_temp is not None else None
        theta_s = theta_s_temp.repeat(num_seq, 1, 1, 1, 1) if theta_s_temp is not None else None
        porosity = porosity_temp.repeat(num_seq, 1, 1, 1, 1) if porosity_temp is not None else None
        specific_storage = (
            specific_storage_temp.repeat(num_seq, 1, 1, 1, 1)
            if specific_storage_temp is not None else None
        )
        mask = mask_temp.repeat(num_seq, 1, 1, 1, 1) if mask_temp is not None else None

        if str(self.input_param.device).startswith("cuda"):
            print("[load_data] static_inputs moved to GPU; init_cond/forcings/targets pinned in CPU", flush=True)
            static_inputs = static_inputs.to(self.input_param.device)
            init_cond = init_cond.pin_memory()
            forcings = forcings.pin_memory()
            targets = targets.pin_memory()

            if alpha is not None:
                alpha = alpha.pin_memory()
            if n_value is not None:
                n_value = n_value.pin_memory()
            if theta_r is not None:
                theta_r = theta_r.pin_memory()
            if theta_s is not None:
                theta_s = theta_s.pin_memory()
            if porosity is not None:
                porosity = porosity.pin_memory()
            if specific_storage is not None:
                specific_storage = specific_storage.pin_memory()
            if mask is not None:
                mask = mask.pin_memory()

        bytes_per_element = 4
        total_elements = (
            numel_or_zero(init_cond) +
            numel_or_zero(static_inputs) +
            numel_or_zero(forcings) +
            numel_or_zero(targets) +
            numel_or_zero(alpha) +
            numel_or_zero(n_value) +
            numel_or_zero(theta_r) +
            numel_or_zero(theta_s) +
            numel_or_zero(porosity) +
            numel_or_zero(specific_storage) +
            numel_or_zero(mask)
        )
        mem_fp32 = total_elements * bytes_per_element / (1024 ** 3)

        print(f"Num patches: {num_patch}")
        print(f"Num sequences: {num_seq}")
        print(f"Total samples: {num_patch * num_seq}")
        print(f"Estimated memory: {mem_fp32:.2f} GB")
        print(f"Use storage terms: {use_storage_terms}")

        return (
            init_cond,
            static_inputs,
            forcings,
            targets,
            num_patch * num_seq,
            coords_space,
            num_patch,
            alpha,
            n_value,
            theta_r,
            theta_s,
            porosity,
            specific_storage,
            mask,
        )

    def get_train_input_handle(self):
        (
            init_cond,
            static_inputs,
            forcings,
            targets,
            total_seq,
            coords_space,
            num_patch,
            alpha,
            n_value,
            theta_r,
            theta_s,
            porosity,
            specific_storage,
            mask,
        ) = self.load_data(mode='train')

        return InputHandle(
            init_cond,
            static_inputs,
            forcings,
            targets,
            total_seq,
            self.input_param,
            alpha,
            n_value,
            theta_r,
            theta_s,
            porosity,
            specific_storage,
            mask,
            mode='train',
            coords_space=coords_space,
            num_patch=num_patch,
        )

    def get_test_input_handle(self):
        (
            init_cond,
            static_inputs,
            forcings,
            targets,
            total_seq,
            coords_space,
            num_patch,
            alpha,
            n_value,
            theta_r,
            theta_s,
            porosity,
            specific_storage,
            mask,
        ) = self.load_data(mode='test')

        return InputHandle(
            init_cond,
            static_inputs,
            forcings,
            targets,
            total_seq,
            self.input_param,
            alpha,
            n_value,
            theta_r,
            theta_s,
            porosity,
            specific_storage,
            mask,
            mode='test',
            coords_space=coords_space,
            num_patch=num_patch,
        )
