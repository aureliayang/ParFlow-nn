__author__ = 'chen yang'

import os
import argparse
import random
import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from core.data_provider import datasets_factory
from core.models.model_factory import Model
import core.trainer as trainer

# -----------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="CONCN surrogate model - ParFlow-nn")

    # training/test
    parser.add_argument("--is_training", type=int, default=1)
    parser.add_argument("--is_test_lsm", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--attn_mode", type=str, default="none")

    # data
    parser.add_argument("--dataset_name", type=str, default="pfnn")
    parser.add_argument("--pf_runname", type=str, default="Little_Washita")

    # model
    parser.add_argument("--model_name", type=str, default="predrnn_pf")
    parser.add_argument("--pretrained_model", type=str, default="")

    parser.add_argument("--save_dir", type=str, default="checkpoints/mnist_predrnn")
    parser.add_argument("--gen_frm_dir", type=str, default="results/mnist_predrnn")

    parser.add_argument("--training_start_step", type=int, default=0)
    parser.add_argument("--training_end_step", type=int, default=20)
    parser.add_argument("--test_start_step", type=int, default=0)
    parser.add_argument("--test_end_step", type=int, default=20)
    parser.add_argument("--img_height", type=int, default=64)
    parser.add_argument("--img_width", type=int, default=64)

    # paths
    parser.add_argument("--static_inputs_path", type=str, default="")
    parser.add_argument("--static_inputs_filename", type=str, default="")
    parser.add_argument("--forcings_path", type=str, default="")
    parser.add_argument("--targets_path", type=str, default="")
    parser.add_argument("--forcings_paths", type=str, nargs="+", default=[])
    parser.add_argument("--targets_paths", type=str, nargs="+", default=[])
    parser.add_argument("--norm_file", type=str, default="normalize.yaml")
    parser.add_argument("--target_mean", default=[])
    parser.add_argument("--target_std", default=[])
    parser.add_argument("--force_mean", default=[])
    parser.add_argument("--force_std", default=[])

    parser.add_argument("--lsm_forcings_path", type=str, default="")
    parser.add_argument("--lsm_forcings_name", type=str, default="")

    # channel
    parser.add_argument("--init_cond_channel", type=int, default=10)
    parser.add_argument("--static_channel", type=int, default=15)
    parser.add_argument("--act_channel", type=int, default=4)
    parser.add_argument("--img_channel", type=int, default=10)

    # CNN
    parser.add_argument("--num_hidden", type=str, default="16,16")
    parser.add_argument("--filter_size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--decouple_beta", type=float, default=1.0)
    parser.add_argument("--grad_beta", type=float, default=0.5)
    parser.add_argument("--input_length_train", type=int, default=10)
    parser.add_argument("--input_length_test", type=int, default=10)

    parser.add_argument("--ss_stride_train", type=int, default=1)
    parser.add_argument("--st_stride_train", type=int, default=1)
    parser.add_argument("--ss_stride_test", type=int, default=1)
    parser.add_argument("--st_stride_test", type=int, default=1)

    # optimization
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--lr_mode", type=str, default="constant",
                    choices=["constant", "onecycle"])
    parser.add_argument("--max_lr", type=float, default=0.002)
    parser.add_argument("--onecycle_pct_start", type=float, default=0.3)
    parser.add_argument("--onecycle_div_factor", type=float, default=5.0)
    parser.add_argument("--onecycle_final_div_factor", type=float, default=5.0)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_iterations", type=int, default=80000)
    parser.add_argument("--display_interval", type=int, default=100)
    parser.add_argument("--test_interval", type=int, default=5000)
    parser.add_argument("--snapshot_interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=420)

    # physics / grid
    parser.add_argument("--dx", type=float, default=961.72)
    parser.add_argument("--dy", type=float, default=961.72)
    parser.add_argument("--storage_beta", type=float, default=0.0)
    parser.add_argument(
        "--dz",
        type=float,
        nargs="+",
        default=[
            100.0, 1.1370, 0.9133, 0.5539, 0.3360, 0.2038,
            0.1236, 0.0749, 0.0455, 0.0276, 0.0175
        ],
    )
    return parser

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def load_norm_stats(args):
    norm_path = Path(args.norm_file)
    if not norm_path.exists():
        raise FileNotFoundError(f"Normalization file not found: {norm_path}")

    with norm_path.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f) or {}

    if "target_mean" in y:
        args.target_mean = y["target_mean"]
    if "target_std" in y:
        args.target_std = y["target_std"]
    if "force_mean" in y:
        args.force_mean = y["force_mean"]
    if "force_std" in y:
        args.force_std = y["force_std"]

def validate_args(args):
    if args.device not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {args.device}. Choose from ['cpu', 'cuda'].")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise EnvironmentError("CUDA was requested, but no CUDA device is available.")

    if args.is_training not in {0, 1}:
        raise ValueError("--is_training must be 0 or 1")

    if args.is_test_lsm not in {0, 1}:
        raise ValueError("--is_test_lsm must be 0 or 1")

    if args.max_iterations <= 0:
        raise ValueError("--max_iterations must be > 0")

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be > 0")

    if args.snapshot_interval <= 0:
        raise ValueError("--snapshot_interval must be > 0")

    if args.test_interval <= 0:
        raise ValueError("--test_interval must be > 0")

    if args.img_height <= 0 or args.img_width <= 0:
        raise ValueError("--img_height and --img_width must be > 0")

    if args.is_training == 0 or args.is_test_lsm == 1:
        if not args.pretrained_model:
            raise ValueError("Testing mode requires --pretrained_model")
        if not Path(args.pretrained_model).exists():
            raise FileNotFoundError(f"Pretrained model not found: {args.pretrained_model}")

def print_config(args):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("\n===== Config =====")
    for k, v in sorted(vars(args).items()):
        print(f"{k:25s}: {v}")
    print("==================\n")

def prepare_dirs(args):
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.gen_frm_dir).mkdir(parents=True, exist_ok=True)

def debug_random_state(device: str):
    print("Random:", random.random())
    print("Numpy:", np.random.rand(5))
    print("Torch CPU:", torch.rand(3))

    if device == "cuda":
        print("Torch CUDA:", torch.rand(3, device="cuda"))
    else:
        print("Torch CUDA: skipped (device=cpu)")

def train_wrapper(model, args):
    start_itr = 0
    if args.pretrained_model:
        start_itr = model.load(
            args.pretrained_model,
            model.optimizer,
            model.scheduler
        )
        print(f"Resume training from iteration {start_itr}")

    train_input_handle, test_input_handle = datasets_factory.data_provider(args)

    for itr in range(start_itr + 1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)

        (
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
            mask,
        ) = train_input_handle.get_batch()

        trainer.train(
            model,
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
            mask,
            args,
            itr,
        )

        if itr % args.snapshot_interval == 0:
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'save model start')
            model.save(itr, model.optimizer, model.scheduler)
            print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'save model end')

        if itr % args.test_interval == 0:
            trainer.test(
                model,
                test_input_handle,
                args,
                itr,
                save_results=(itr % args.snapshot_interval == 0)
            )

        train_input_handle.next()

def test_wrapper(model, args):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(args)
    trainer.test(model, test_input_handle, args, 'test_result')

def test_lsm_wrapper(model, args):
    model.load(args.pretrained_model)
    trainer.test_lsm(model, args, 'test_result')

def main():
    parser = build_parser()
    args = parser.parse_args()

    load_norm_stats(args)
    validate_args(args)
    print_config(args)

    set_seed(args.seed)
    debug_random_state(args.device)
    set_seed(args.seed)
    prepare_dirs(args)

    print("Initializing models")
    model = Model(args)

    num_params = sum(p.numel() for p in model.network.parameters())
    print(f"Total parameters: {num_params:,}")

    if args.is_test_lsm:
        args.is_training = 0
        test_lsm_wrapper(model, args)
    elif args.is_training:
        train_wrapper(model, args)
    else:
        test_wrapper(model, args)

    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    main()
