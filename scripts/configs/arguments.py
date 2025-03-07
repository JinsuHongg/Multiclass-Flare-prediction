import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser()
    
    # General arguments
    parser.add_argument("--config", type=str, default="./Multiclass-Flare-prediction/scripts/configs/cp_config.yaml", help="Path to YAML config file")
    
    # Directory arguments
    parser.add_argument("--img_dir", type=str, default=None)

    # Training arguments
    parser.add_argument("--model", type=str, default="Alexnet")
    parser.add_argument("--train_set", type=float, nargs="+", default=[1, 2])
    parser.add_argument("--test_set", type=int, default=4)
    parser.add_argument("--file_tag", type=str, default="cp")

    # Optimization arguments
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--wt_decay", type=float, nargs="+", default=[0])

    args = parser.parse_args()

    # Load YAML if provided
    if args.config:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
        
        # Flatten and update args
        for key, sub_dict in yaml_config.items():
            if isinstance(sub_dict, dict):  # If it's a nested dictionary
                for sub_key, value in sub_dict.items():
                    if hasattr(args, sub_key):
                        setattr(args, sub_key, value)
            else:
                if hasattr(args, key):
                    setattr(args, key, sub_dict)

    return args