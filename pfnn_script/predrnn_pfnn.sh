export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --is_test_lsm 0 \
    --device cuda \
    --dataset_name pfnn \
    --pf_runname unname_test \
    --model_name predrnn_pf \
    --save_dir checkpoints/pfnn_predrnn \
    --gen_frm_dir results/pfnn_predrnn \
    --norm_file /home/cy15/clm_call/normalize_481_7000.yaml \
    --attn_mode none \
    --training_start_step 601 \
    --training_end_step 3000 \
    --test_start_step 1 \
    --test_end_step 120 \
    --img_height 146 \
    --img_width 252 \
    --patch_size 16 \
    --input_length_train 120 \
    --input_length_test 120 \
    --ss_stride_train 16 \
    --st_stride_train 120 \
    --ss_stride_test 16 \
    --st_stride_test 120 \
    --static_inputs_path /home/cy15/clm_call \
    --static_inputs_filename static_inputs_combined.pfb \
    --forcings_path /home/cy15/unname/pfb_shallow_2nd \
    --targets_path /home/cy15/unname/pfb_shallow_2nd \
    --init_cond_channel 11 \
    --static_channel 40 \
    --act_channel 10 \
    --img_channel 11 \
    --lr 0.0003 \
    --grad_beta 0.5 \
    --batch_size 16 \
    --max_iterations 50000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 5000 \
    # --pretrained_model /home/cy15/ParFlow-nn/checkpoints_1001_3400/pfnn_predrnn/model.ckpt-50000 \
    # --lsm_forcings_path /home/cy15/E5L \
    # --lsm_forcings_name E5L 
   