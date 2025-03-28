export CUDA_VISIBLE_DEVICES=0
cd ..
python -u run.py \
    --is_training 1 \
    --device cuda \
    --dataset_name pfnn \
    --train_data_paths /home/aurelia/parflow-nn/WTD_2018 \
    --valid_data_paths /home/aurelia/parflow-nn/WTD_2018 \
    --save_dir checkpoints/pfnn_predrnn \
    --gen_frm_dir results/pfnn_predrnn \
    --model_name predrnn \
    --reverse_input 1 \
    --img_width 128 \
    --img_channel 1 \
    --input_length 10 \
    --total_length 20 \
    --num_hidden 128,128,128,128 \
    --filter_size 5 \
    --stride 1 \
    --patch_size 4 \
    --layer_norm 0 \
    --scheduled_sampling 1 \
    --sampling_stop_iter 50000 \
    --sampling_start_value 1.0 \
    --sampling_changing_rate 0.00002 \
    --lr 0.0003 \
    --batch_size 4 \
    --max_iterations 80000 \
    --display_interval 100 \
    --test_interval 100 \
    --snapshot_interval 5000 \
    --pretrained_model /home/aurelia/parflow-nn/predrnn-pytorch-master/checkpoints_1/pfnn_predrnn/model.ckpt-10000