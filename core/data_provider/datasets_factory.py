from core.data_provider import pfnn
# , bair

datasets_map = {
    'pfnn': pfnn
    # 'bair': bair,
}


def data_provider(configs):
    dataset_name = configs.dataset_name
    if dataset_name not in datasets_map:
        raise ValueError('Name of dataset unknown %s' % dataset_name)
    
    if dataset_name == 'pfnn':
        # input_param = {'paths': valid_data_list,
        #                'image_width': img_width,
        #                'minibatch_size': batch_size,
        #                'seq_length': seq_length,
        #                'input_data_type': 'float32',
        #                'name': dataset_name + ' iterator'}
        input_handle = datasets_map[dataset_name].DataProcess(configs)
        if configs.is_training:
            train_input_handle = input_handle.get_train_input_handle()
            train_input_handle.begin(do_shuffle=True)
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return train_input_handle, test_input_handle
        else:
            test_input_handle = input_handle.get_test_input_handle()
            test_input_handle.begin(do_shuffle=False)
            return test_input_handle