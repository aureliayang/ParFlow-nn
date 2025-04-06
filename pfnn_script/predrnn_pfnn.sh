export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name pfnn \
    --pf_runname a1_run_5 \
    --model_name predrnn_pf \
    --save_dir checkpoints/pfnn_predrnn \
    --gen_frm_dir results/pfnn_predrnn \
    --training_start_step 201 \
    --training_end_step 800 \
    --test_start_step 1001 \
    --test_end_step 1200 \
    --img_height 146 \
    --img_width 252 \
    --init_cond_path /home/aurelia/parflow-nn/standard_2018/output_press \
    --init_cond_filename a1_run_5.out.press.00200.pfb \
    --static_inputs_path /home/aurelia/parflow-nn/standard_2018/subsurface_data \
    --static_inputs_filename static_inputs_combined.pfb \
    --forcings_path /home/aurelia/parflow-nn/standard_2018/output_evapotrans \
    --targets_path /home/aurelia/parflow-nn/standard_2018/output_press \
    --force_norm_file a1_run_5.out.evaptrans.00200.pfb \
    --target_norm_file a1_run_5.out.press.00200.pfb \
    --init_cond_channel 10 \
    --static_channel 16 \
    --act_channel 4 \
    --img_channel 10 \
    --input_length 10 \
    --patch_size 16 \
    --lr 0.0003 \
    --batch_size 16 \
    --max_iterations 10000 \
    --display_interval 100 \
    --test_interval 1000 \
    --snapshot_interval 10000 \
    # --pretrained_model /home/aurelia/parflow-nn/predrnn-pytorch-master/checkpoints_1/pfnn_predrnn/model.ckpt-10000
    # --reverse_input 1 \