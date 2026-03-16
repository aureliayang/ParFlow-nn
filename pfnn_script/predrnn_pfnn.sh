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
    --norm_file /data/aurelia-data/workspace/clm_call/normalize_1_6000.yaml \
    --attn_mode none \
    --training_start_step 1 \
    --training_end_step 6000 \
    --test_start_step 6961 \
    --test_end_step 8760 \
    --img_height 146 \
    --img_width 252 \
    --patch_size 16 \
    --input_length_train 120 \
    --input_length_test 120 \
    --ss_stride_train 16 \
    --st_stride_train 120 \
    --ss_stride_test 16 \
    --st_stride_test 120 \
    --static_inputs_path /data/aurelia-data/workspace/clm_call \
    --static_inputs_filename static_inputs_combined.pfb \
    --forcings_path /data/aurelia-data/workspace/beijiang/outputs \
    --targets_path /data/aurelia-data/workspace/beijiang/outputs \
    --init_cond_channel 11 \
    --static_channel 40 \
    --act_channel 10 \
    --img_channel 11 \
    --lr 0.0003 \
    --grad_beta 0.1 \
    --batch_size 16 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 5000 \
    # --pretrained_model /data/aurelia-data/workspace/ParFlow-nn/checkpoints_none/pfnn_predrnn/model.ckpt-80000 \
    # --lsm_forcings_path /data/aurelia-data/workspace/E5L \
    # --lsm_forcings_name E5L 
   