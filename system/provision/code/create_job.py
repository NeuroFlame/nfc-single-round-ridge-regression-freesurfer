import os
import shutil
import json
import sys
from typing import Dict, Any

def generate_job_meta(min_clients: int) -> Dict[str, Any]:
    return {
        "resource_spec": {},
        "min_clients": min_clients,
        "deploy_map": {
            "app": ["@ALL"]
        }
    }

def create_job(app_path: str, job_path: str, min_clients: int) -> None:
    if not os.path.isdir(app_path):
        raise FileNotFoundError(f"Source app path '{app_path}' does not exist.")
    
    # Prepare the destination path for the app
    job_app_path = os.path.join(job_path, 'app')
    os.makedirs(job_app_path, exist_ok=True)

    # Copy the app directory
    shutil.copytree(app_path, job_app_path, dirs_exist_ok=True)
    update_client_tasks_from_spec(job_app_path)

    # Generate and write job_meta to meta.json
    job_meta = generate_job_meta(min_clients)
    with open(os.path.join(job_path, 'meta.json'), 'w') as meta_file:
        json.dump(job_meta, meta_file, indent=2)

# Example usage:
# create_job('/path/to/app_folder', '/path/to/job_folder', min_clients=2)


def update_client_tasks_from_spec(app_path: str) -> None:
    code_path = os.path.join(app_path, "code")
    sys.path.insert(0, code_path)
    try:
        from computation.spec import SPEC
        from framework.workflow import get_task_names
    finally:
        if sys.path and sys.path[0] == code_path:
            sys.path.pop(0)

    config_path = os.path.join(app_path, "config", "config_fed_client.json")
    with open(config_path, "r+") as config_file:
        config = json.load(config_file)
        if "executors" in config and config["executors"]:
            config["executors"][0]["tasks"] = get_task_names(SPEC.workflow)
        config_file.seek(0)
        json.dump(config, config_file, indent=2)
        config_file.truncate()
