from core.data_provider import pfnn

datasets_map = {
    'pfnn': pfnn,
}

def data_provider(configs):
    dataset_name = configs.dataset_name
    if dataset_name not in datasets_map:
        raise ValueError(
            f"Unknown dataset_name: {dataset_name}. "
            f"Available datasets: {list(datasets_map.keys())}"
        )

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