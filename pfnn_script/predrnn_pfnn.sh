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
    --training_start_step 2401 \
    --training_end_step 4800 \
    --test_start_step 4801 \
    --test_end_step 5040 \
    --img_height 146 \
    --img_width 252 \
    --ss_stride_train 12 \
    --st_stride_train 90 \
    --ss_stride_test 16 \
    --st_stride_test 240 \
    --static_inputs_path /home/cy15/clm_call \
    --static_inputs_filename static_inputs_combined.pfb \
    --forcings_path /home/cy15/unname/pfb_shallow_2nd \
    --targets_path /home/cy15/unname/pfb_shallow_2nd \
    --init_cond_channel 11 \
    --static_channel 40 \
    --act_channel 10 \
    --img_channel 11 \
    --input_length_train 120 \
    --input_length_test  240 \
    --patch_size 16 \
    --lr 0.0003 \
    --batch_size 16 \
    --max_iterations 20000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 5000 \
    --lsm_forcings_path /home/cy15/E5L \
    --lsm_forcings_name E5L \
    --pretrained_model /home/cy15/ParFlow-nn/checkpoints/pfnn_predrnn/model.ckpt-20000

    # --init_cond_test_path /home/cy15/unname/pfb_shallow_2nd \
    # --init_cond_test_filename unname_test.out.press.01200.pfb \
    # necessary for lsm test


    # --reverse_input 1 \
    # --init_cond_path /home/cy15/unname/pfb_shallow_2nd \
    # --init_cond_filename unname_test.out.press.00000.pfb \
    # --force_norm_file unname_test.out.evaptrans.00200.pfb \
    # --target_norm_file unname_test.out.press.00200.pfb \