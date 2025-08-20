import os
import ast
import argparse
import tempfile
from stseg.data_processing import get_data_loaders
from stseg.utils import get_config, check_and_convert_user_config_args
from stseg.model import SegModel


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a model to segment surgical instruments.")
    parser.add_argument("dataset_path", type=str, help="Path to the dataset.")
    parser.add_argument("results_path", type=str, help="Path to save results.")
    parser.add_argument("splitting", choices=["train-val-test", "5-fold"],
                        help="Choose either 'train-val-test' for a standard split or '5-fold' for cross-validation.")
    parser.add_argument("n_classes", type=int, help="Number of classes.")
    parser.add_argument("patch_size", type=ast.literal_eval, help="Patch size in format [W,H]")
    parser.add_argument("-f", "--fold", type=int, choices=[0, 1, 2, 3, 4], required=False, default=None,
                        help="Specify the fold index (0-4) when using 5-fold cross-validation.")
    parser.add_argument("-p", "--progress_bar", action="store_true", help="Enable progress bar (default: False)")
    parser.add_argument("-c", "--continue_training", action="store_true",
                        help="Continue training from the last checkpoint (default: False)")

    main_args, additional_args = parser.parse_known_args()

    additional_args = check_and_convert_user_config_args(additional_args, mode='training')

    # Ensure --fold is provided only when --splitting is "5-fold"
    if main_args.splitting == "5-fold" and main_args.fold is None:
        parser.error("--fold is required when --splitting is set to '5-fold'")

    # Ensure --fold is None when --splitting is "train-val-test"
    if main_args.splitting == "train-val-test" and main_args.fold is not None:
        parser.error("--fold should not be provided when --splitting is set to 'train-val-test'")

    if os.path.exists(main_args.results_path) and not main_args.continue_training:
        raise FileExistsError(f"Results path {main_args.results_path} already exists.")

    return main_args, additional_args


def main():
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temp directory: {temp_dir}")

        args, config_args = parse_arguments()
        dataset_path = args.dataset_path
        results_path = args.results_path
        splitting = args.splitting
        n_classes = args.n_classes
        patch_size = args.patch_size
        fold = args.fold
        progress_bar = args.progress_bar
        continue_training = args.continue_training

        mode = 'continue_training' if continue_training else 'training'
        config = get_config(dataset_path, results_path, mode, config_args=config_args, n_classes=n_classes,
                            patch_size=patch_size, progress_bar=progress_bar)

        train_loader, val_loader = get_data_loaders(config, splitting, fold)

        model = SegModel(config=config)
        model.train(train_loader=train_loader, val_loader=val_loader)
