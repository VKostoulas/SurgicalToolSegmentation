import os
import glob
import tempfile
import argparse
from torch.utils.data import DataLoader
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from stseg.utils import get_config, check_and_convert_user_config_args
from stseg.data_processing import create_split_files, get_data_ids, SegTestDataset
from stseg.model import SegModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on a data.")
    parser.add_argument("test_dataset_path", type=str, help="Path to the dataset we want to infer on.")
    parser.add_argument("results_path", type=str, help="Path to save results.")

    main_args, additional_args = parser.parse_known_args()

    additional_args = check_and_convert_user_config_args(additional_args, mode='testing')

    if not os.path.exists(main_args.results_path):
        raise FileExistsError(f"Results path {main_args.results_path} should already exist. First, train a model.")

    return main_args, additional_args


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")

        args, config_args = parse_arguments()
        test_dataset_path = args.test_dataset_path
        results_path = args.results_path

        config = get_config(test_dataset_path, results_path, mode='testing', config_args=config_args)

        train_path = config['dataset_path']
        if train_path != test_dataset_path:
            print("Training data path is different from dataset path. Assuming we are in 'testing' mode and we are testing on a new dataset.")
            file_paths = glob.glob(os.path.join(test_dataset_path, 'data', "*.zarr"))
            data_ids = [os.path.basename(fp).replace('.zarr', '') for fp in file_paths]
            print(f"{len(data_ids)} samples for testing")
        else:
            print("Testing on the test partition of the training dataset.")
            split_file_path = create_split_files(test_dataset_path, 'train-val-test', seed=12345)
            data_ids = get_data_ids(split_file_path)['test']

        preprocess_func = get_preprocessing_fn(config['model']['encoder_name'], pretrained=config['model']['encoder_weights'])
        test_ds = SegTestDataset(data_path=test_dataset_path, data_ids=data_ids, batch_size=config['infer_batch_size'], preprocess_func=preprocess_func)
        test_loader = DataLoader(test_ds, batch_size=None, shuffle=False, num_workers=config['infer_num_workers'],
                                 pin_memory=True, persistent_workers=True, prefetch_factor=1)

        model = SegModel(config=config)
        model.run_inference(test_loader=test_loader)
