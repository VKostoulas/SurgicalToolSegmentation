import os
import ast
import yaml
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


def save_losses(loss_dict, save_path, log_scale=False):
    os.makedirs(save_path, exist_ok=True)
    save_plot_path = os.path.join(save_path, f"loss.png")

    epochs = range(1, len(loss_dict['train_loss']) + 1)

    mapping_names_dict = {'train_loss': 'Train DiceCELoss', 'val_loss': 'Val DiceCELoss'}

    plt.figure(figsize=(10, 8))

    for key in mapping_names_dict:
        if key in loss_dict.keys():
            plt.plot(epochs, loss_dict[key], label=mapping_names_dict[key], linestyle='-')

    if log_scale:
        plt.yscale('log')
        plt.ylabel("log(loss)")
    else:
        plt.ylabel("loss")
    plt.xlabel("Epoch")
    plt.title("Losses per Epoch")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=300, bbox_inches='tight')
    plt.close()


def clean_numpy_scalars(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Convert np.float64, np.int64, etc. to Python scalars
    elif isinstance(obj, tuple):
        return tuple(clean_numpy_scalars(x) for x in obj)
    elif isinstance(obj, list):
        return [clean_numpy_scalars(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_numpy_scalars(v) for k, v in obj.items()}
    else:
        return obj


def get_default_config(n_classes, patch_size):
    # create a default configuration file
    n_epochs = 200
    transformations = {"patch_size": patch_size, "scaling": True, "rotation": True, "gaussian_noise": True,
                       "gaussian_blur": True, "low_resolution": False, "brightness": True, "contrast": True,
                       "gamma": True, "mirror": True, "dummy_2d": False}
    config = {"n_classes": n_classes,
              "transformations": transformations,
              "oversample_ratio": 0.33,
              "batch_size": 8,
              "num_workers": 8,
              "infer_batch_size": 6,
              "infer_num_workers": 4,
              "sw_batch_size": 24,
              "sw_overlap": 0.5,
              "n_epochs": n_epochs,
              "val_plot_interval": 10,
              "grad_clip_max_norm": 12,
              "grad_accumulate_step": 1,
              "lr_scheduler": {"name": "PolynomialLR", "total_iters": n_epochs, "power": 0.9},
              "optimizer": {"name": 'AdamW', "lr": 1e-4},
              "model": {"arch": 'UnetPlusPlus', "encoder_name": 'tu-mobilenetv3_small_100', "encoder_weights": 'imagenet',
                        "in_channels": 3, "classes": n_classes + 1}}

    return config


def create_config(config, results_path):
    config_save_path = os.path.join(results_path, 'config.yaml')
    os.makedirs(results_path, exist_ok=True)

    # Custom Dumper to avoid anchors and enforce list formatting
    class CustomDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True  # Removes YAML anchors (&id001)

    # Ensure lists stay in flow style
    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    CustomDumper.add_representer(list, represent_list)
    # Save to YAML with all fixes
    with open(config_save_path, "w") as file:
        yaml.dump(config, file, sort_keys=False, Dumper=CustomDumper)
    print(f"Saved configuration at {config_save_path}")


def add_user_config_args(config, config_args):
    if config_args is not None:
        for key in config_args:
            if not isinstance(config_args[key], dict):
                config[key] = config_args[key]
            else:
                for inner_key in config_args[key]:
                    config[key][inner_key] = config_args[key][inner_key]
    return config


def check_and_convert_user_config_args(args, mode):

    args = [a.lstrip("-") for a in args]  # strip leading "-" or "--"
    args = {args[i]: ast.literal_eval(args[i + 1]) for i in range(0, len(args), 2)}

    if args:
        allowed_training_args = ["transformations_scaling", "transformations_rotation", "transformations_gaussian_noise",
                                 "transformations_gaussian_blur", "transformations_low_resolution", "transformations_brightness",
                                 "transformations_contrast", "transformations_gamma", "transformations_mirror", "transformations_dummy_2d",
                                 "oversample_ratio", "batch_size", "num_workers", "n_epochs", "val_plot_interval", "grad_clip_max_norm",
                                 "grad_accumulate_step"]
        allowed_testing_args = ["infer_batch_size", "infer_num_workers", "sw_batch_size", "sw_overlap"]
        flexible_args = ["lr_scheduler", "optimizer", "model"]
        args_with_prefix = ["transformations", "lr_scheduler", "optimizer", "model"]

        args_dict = {}
        for arg in args:
            arg_wrong_for_training = (arg not in allowed_training_args and mode =='training') and not any([item in arg for item in flexible_args])
            arg_wrong_for_inference = arg not in allowed_testing_args and mode =='testing'
            if arg_wrong_for_training or arg_wrong_for_inference:
                raise KeyError(f"Argument '{arg}' is not an allowed argument. Valid training arguments: {allowed_training_args}. "
                               f"Valid inference arguments: {allowed_testing_args}. "
                               f"Valid flexible arguments: {[item + '_*' for item in flexible_args]}. ")

            if not any([item in arg for item in args_with_prefix]):
                args_dict[arg] = args[arg]
            else:
                for item in args_with_prefix:
                    if item in arg:
                        if not item in args_dict.keys():
                            args_dict[item] = {}
                        key_name = arg.replace(item + '_', '')
                        args_dict[item][key_name] = args[arg]

        return args_dict

    else:
        return None


def get_config(dataset_path, results_path, n_classes, patch_size, config_args=None, progress_bar=False, mode=None):
    config_path = os.path.join(results_path, 'config.yaml')

    if mode == 'training':
        config = get_default_config(n_classes, patch_size)
        config["dataset_path"] = dataset_path
        config = add_user_config_args(config, config_args)
        create_config(config, results_path)
    else:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        print(f"Using configuration file: {config_path}")
        user_config = add_user_config_args(deepcopy(config), config_args)

        if mode == 'continue_training':
            different_args = [(key, config[key], user_config[key]) for key in user_config if config[key] != user_config[key]]
            assertion_print = ",".join([f"arg '{item[0]}' - ({item[1], item[2]})" for item in different_args])
            assert len(different_args) == 0, f"Different argument values were given between continue_training and training: " + assertion_print
        elif mode == 'testing':
            if config_args is not None:
                different_args = [(key, config[key], user_config[key]) for key in user_config if config[key] != user_config[key]]
                changed_args_print = "\n    ".join([f"'{item[0]}': {item[1]} --> {item[2]}" for item in different_args])
                if len(different_args) != 0:
                    print(f"Inference arguments changed:\n    {changed_args_print}")

        config = user_config

    config['progress_bar'] = progress_bar
    config['results_path'] = results_path

    last_model_path = os.path.join(results_path, 'checkpoints', 'last_model.pth')
    best_model_path = os.path.join(results_path, 'checkpoints', 'best_model.pth')
    config['load_model_path'] = last_model_path if mode == 'continue_training' else best_model_path if mode == 'testing' else None

    return config