import os
import json
import subprocess
from datetime import datetime

def get_git_commit_hash():
    """Retrieves the current Git commit hash of the repository."""
    try:
        # Runs the git command in the terminal to get the hash
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
        return commit_hash
    except Exception:
        return "git_not_found_or_uncommitted"

def save_experiment_metadata(results_folder, experiment_name, config_dict):
    """Saves the RNG seed, parameters, and Git hash to the results/ folder."""
    os.makedirs(results_folder, exist_ok=True)
    
    # Add reproducibility trackers to the dictionary
    config_dict["git_commit"] = get_git_commit_hash()
    config_dict["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save as a JSON file
    filename = f"{experiment_name}_metadata.json"
    filepath = os.path.join(results_folder, filename)
    
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=4)
        
    print(f" Run metadata (including Git Hash) saved to: {filepath}")