__author__ = 'chen yang'

import os
import shutil
import argparse
# import numpy as np
# import math
from core.data_provider import datasets_factory
from core.models.model_factory import Model
# from core.utils import preprocess
import core.trainer as trainer
import torch

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='CONCN surrogate model - ParFlow-nn')

# training/test
parser.add_argument('--is_training', type=int, default=1)
parser.add_argument('--is_test_lsm', type=int, default=0)
parser.add_argument('--device', type=str, default='cpu:0')

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
# parser.add_argument('--init_cond_path', type=str, default='')
# parser.add_argument('--init_cond_filename', type=str, default='')  #associated with training_start_step (minus -1)
parser.add_argument('--init_cond_test_path', type=str, default='')
parser.add_argument('--init_cond_test_filename', type=str, default='')  #associated with test_start_step (minus -1)
parser.add_argument('--static_inputs_path', type=str, default='')
parser.add_argument('--static_inputs_filename', type=str, default='') #combined and put in a new dir
parser.add_argument('--forcings_path', type=str, default='')
parser.add_argument('--targets_path', type=str, default='') # forcings and targets may be in the same dir
# parser.add_argument('--target_norm_file', type=str, default='')
# parser.add_argument('--force_norm_file', type=str, default='')
parser.add_argument('--target_mean', type=str, default='50.6647919,2.18094949,1.19974567,0.5132593,\
                    0.11379735,-0.12730431,-0.27250824,-0.36000992,-0.41267873,-0.44452377,-0.46438877')
parser.add_argument('--target_std', type=str, default='11.32939271,0.93890168,0.81459955,0.69557101,\
                    0.59929806,0.54758706,0.52029829,0.50644763,0.49912652,0.49520303,0.49344641')
parser.add_argument('--force_mean', type=str, default='0.00000000e+00,-1.65678623e-06,-2.55343334e-06,-9.06575188e-06,\
                    -2.91817793e-05,-7.71748930e-05,-1.55229783e-04,-2.26101332e-04,-2.38931359e-04,-1.84304437e-04,9.87172230e-03')
parser.add_argument('--force_std', type=str, default='0.00000000e+00,2.97740304e-06,4.71635086e-06,1.69609972e-05,\
                    5.46345691e-05,1.41404926e-04,2.77901646e-04,4.11366363e-04,4.57953708e-04,3.71477025e-04,3.55238828e-02')

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
parser.add_argument('--decouple_beta', type=float, default=0.1)
parser.add_argument('--input_length_train', type=int, default=10)
parser.add_argument('--input_length_test', type=int, default=10)
# parser.add_argument('--lsm_timesteps', type=int, default=10)
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
# parser.add_argument('--n_gpu', type=int, default=1)
# parser.add_argument('--reverse_input', type=int, default=1)

# # visualization of memory decoupling
# parser.add_argument('--visual', type=int, default=0)
# parser.add_argument('--visual_path', type=str, default='./decoupling_visual')

args = parser.parse_args()
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


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

model = Model(args)

if args.is_test_lsm:
    args.is_training = 0
    test_lsm_wrapper(model)
elif args.is_training:
    train_wrapper(model)
else:
    test_wrapper(model)
