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
    --training_start_step 1 \
    --training_end_step 24 \
    --test_start_step 1 \
    --test_end_step 24 \
    --img_height 146 \
    --img_width 252 \
    --init_cond_path /home/aurelia/parflow-nn/standard_2018/output_press \
    --init_cond_filename a1_run_5.out.press.00000.pfb \
    --static_inputs_path /home/aurelia/parflow-nn/standard_2018/output_press \
    --static_inputs_filename a1_run_5.out.press.00000.pfb \
    --forcings_path /home/aurelia/parflow-nn/standard_2018/output_evapotrans \
    --targets_path /home/aurelia/parflow-nn/standard_2018/output_press \
    --init_cond_channel 10 \
    --static_channel 10 \
    --act_channel 4 \
    --img_channel 10 \
    --input_length 6 \
    --patch_size 64 \
    --lr 0.0003 \
    --batch_size 4 \
    --max_iterations 10 \
    --display_interval 1 \
    --test_interval 5000 \
    --snapshot_interval 5000 \
    # --pretrained_model /home/aurelia/parflow-nn/predrnn-pytorch-master/checkpoints_1/pfnn_predrnn/model.ckpt-10000
    # --reverse_input 1 \