# export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --is_test_lsm 0 \
    --seed 420 \
    --pf_runname unname_test \
    --save_dir checkpoints/onecycle_kernel4 \
    --gen_frm_dir results/onecycle_kernel4 \
    --norm_file /data/aurelia-data/workspace/clm_call/stats_press_evap.yaml \
    --static_inputs_path /data/aurelia-data/workspace/clm_call \
    --static_inputs_filename static_inputs_combined46.pfb \
    --forcings_paths /data/aurelia-data/workspace/beijiang/outputs2 /data/aurelia-data/workspace/beijiang/outputs2020 \
    --targets_paths /data/aurelia-data/workspace/beijiang/outputs2 /data/aurelia-data/workspace/beijiang/outputs2020 \
    --init_cond_channel 11 \
    --static_channel 46 \
    --act_channel 10 \
    --img_channel 11 \
    --attn_mode none \
    --training_start_step 1 \
    --training_end_step 17520 \
    --test_start_step 15001 \
    --test_end_step 17520 \
    --img_height 146 \
    --img_width 252 \
    --patch_size 16 \
    --input_length_train 120 \
    --input_length_test 120 \
    --ss_stride_train 16 \
    --st_stride_train 120 \
    --ss_stride_test 16 \
    --st_stride_test 120 \
    --lr_mode onecycle \
    --lr 0.003 \
    --grad_beta 0.1 \
    --batch_size 16 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 5000
    # --pretrained_model /data/aurelia-data/workspace/ParFlow-nn/checkpoints_none/pfnn_predrnn/model.ckpt-80000 \
    # --lsm_forcings_path /data/aurelia-data/workspace/E5L \
    # --lsm_forcings_name E5L 
   