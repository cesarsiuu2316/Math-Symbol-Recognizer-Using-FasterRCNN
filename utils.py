import json
import os

def load_config(config_path="configs/config_cpu.json"):
    """
    Loads the configuration from a JSON file.
    
    Args:
        config_path (str): Path to the config.json file.
        
    Returns:
        dict: The configuration dictionary.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def update_config(config, updates: dict):
    """
    General-purpose config updater.
    
    Args:
        config (dict): The loaded config dictionary.
        updates (dict): A nested dictionary with the keys to update.
            Example:
            {
                "transform_params": {"scaling_factor": 0.5},
                "anchor_params": {"sizes": [10,20,30]}
            }
    """
    # recursively update config keys
    def _recursive_update(base, new_values):
        for key, val in new_values.items():
            if isinstance(val, dict) and key in base:
                _recursive_update(base[key], val)
            else:
                base[key] = val

    _recursive_update(config, updates)

    # save back to file
    config_path = config["paths"]["config_path"]
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)

    print(f"Updated {config_path} with updates: {updates}")
