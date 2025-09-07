__author__ = 'chen yang'

import os
import shutil
import argparse
import numpy as np
import random
# import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
# from core.utils import preprocess
import core.trainer as trainer
import torch
import yaml

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='CONCN surrogate model - ParFlow-nn')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--is_test_lsm', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu:0')
parser.add_argument('--attn_mode', type=str, default='none')

# data
parser.add_argument('--dataset_name', type=str, default='pfnn') # different preprocessing method
parser.add_argument('--pf_runname', type=str, default='Little_Washita')

# model
parser.add_argument('--model_name', type=str, default='predrnn_pf') # deep learning model
parser.add_argument('--pretrained_model', type=str, default='') # full path of the pretrained model?

parser.add_argument('--save_dir', type=str, default='checkpoints/mnist_predrnn') # model save dir
parser.add_argument('--gen_frm_dir', type=str, default='results/mnist_predrnn') # results save dir

parser.add_argument('--training_start_step', type=int, default=0)
parser.add_argument('--training_end_step', type=int, default=20)
parser.add_argument('--test_start_step', type=int, default=0)
parser.add_argument('--test_end_step', type=int, default=20)
parser.add_argument('--img_height', type=int, default=64)
parser.add_argument('--img_width', type=int, default=64)

#paths
parser.add_argument('--static_inputs_path', type=str, default='')
parser.add_argument('--static_inputs_filename', type=str, default='') #combined and put in a new dir
parser.add_argument('--forcings_path', type=str, default='')
parser.add_argument('--targets_path', type=str, default='') # forcings and targets may be in the same dir
parser.add_argument("--norm_file", type=str, default="normalize.yaml")
parser.add_argument('--target_mean', type=list, default=[])
parser.add_argument('--target_std', type=list, default=[])
parser.add_argument('--force_mean', type=list, default=[])
parser.add_argument('--force_std', type=list, default=[])

parser.add_argument('--lsm_forcings_path', type=str, default='')
parser.add_argument('--lsm_forcings_name', type=str, default='')

#channel
parser.add_argument('--init_cond_channel', type=int, default=10)
parser.add_argument('--static_channel', type=int, default=15)
parser.add_argument('--act_channel', type=int, default=4)
parser.add_argument('--img_channel', type=int, default=10)

#CNN
parser.add_argument('--num_hidden', type=str, default='40,40')
parser.add_argument('--filter_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--decouple_beta', type=float, default=1)
parser.add_argument('--grad_beta', type=float, default=0.5)
parser.add_argument('--input_length_train', type=int, default=10)
parser.add_argument('--input_length_test', type=int, default=10)

parser.add_argument('--ss_stride_train', type=int, default=1)
parser.add_argument('--st_stride_train', type=int, default=1)
parser.add_argument('--ss_stride_test', type=int, default=1)
parser.add_argument('--st_stride_test', type=int, default=1)

# optimization
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=8) # place holder when test
parser.add_argument('--max_iterations', type=int, default=80000) # how many batches
parser.add_argument('--display_interval', type=int, default=100) # print loss
parser.add_argument('--test_interval', type=int, default=5000) # run test in training
parser.add_argument('--snapshot_interval', type=int, default=5000) # save model

args = parser.parse_args()
with open(args.norm_file, "r", encoding="utf-8") as f:
    y = yaml.safe_load(f) or {}

if "target_mean" in y: args.target_mean = y["target_mean"]
if "target_std"  in y: args.target_std  = y["target_std"]
if "force_mean" in y: args.force_mean = y["force_mean"]
if "force_std"  in y: args.force_std  = y["force_std"]
print(args)

def train_wrapper(model):
    if args.pretrained_model:
        model.load(args.pretrained_model)
    # load data
    train_input_handle, test_input_handle = datasets_factory.data_provider(args)

    for itr in range(1, args.max_iterations + 1):
        if train_input_handle.no_batch_left():
            train_input_handle.begin(do_shuffle=True)

        forcings, init_cond, static_inputs, targets = train_input_handle.get_batch()

        trainer.train(model, forcings, init_cond, static_inputs, targets, args, itr)

        if itr % args.snapshot_interval == 0:
            model.save(itr)

        # we will not test during training currently, will do in the future
        if itr % args.test_interval == 0:
            trainer.test(model, test_input_handle, args, itr)

        train_input_handle.next()

def test_wrapper(model):
    model.load(args.pretrained_model)
    test_input_handle = datasets_factory.data_provider(args)
    trainer.test(model, test_input_handle, args, 'test_result')

def test_lsm_wrapper(model):
    model.load(args.pretrained_model)
    # test_input_handle = datasets_factory.data_provider(args)
    trainer.test_lsm(model, args, 'test_result')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

print("Random:", random.random())
print("Numpy:", np.random.rand(5))
print("Torch CPU:", torch.rand(3))
print("Torch CUDA:", torch.rand(3).cuda())

if os.path.exists(args.save_dir):
    if not args.pretrained_model:
        shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
else:
    os.makedirs(args.save_dir)

if os.path.exists(args.gen_frm_dir):
    shutil.rmtree(args.gen_frm_dir)
os.makedirs(args.gen_frm_dir)

print('Initializing models')


#torch.cuda.empty_cache()
#torch.cuda.reset_peak_memory_stats()

model = Model(args)

if args.is_test_lsm:
    args.is_training = 0
    test_lsm_wrapper(model)
elif args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
