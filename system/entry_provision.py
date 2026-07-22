import json
import os
import logging
import sys
import argparse
from provision.code.provision_run import provision_run

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)

def load_provision_input(provision_input_path: str) -> dict:
    try:
        with open(provision_input_path, 'r') as file:
            provision_input = json.load(file)
        logger.info(f"Provision input loaded from {provision_input_path}")
        return provision_input
    except Exception as e:
        logger.error(f"Failed to load provision input: {e}")
        raise


def validate_users(provision_input: dict) -> tuple:
    users = provision_input.get('users')
    if not isinstance(users, list) or not users:
        raise ValueError("Provision input must include a non-empty 'users' list")

    user_ids = []
    site_id_name_map = {}
    for index, user in enumerate(users):
        if not isinstance(user, dict):
            raise ValueError(f"Provision input user at index {index} must be an object with 'id' and 'name'")

        user_id = user.get('id')
        site_name = user.get('name')
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError(f"Provision input user at index {index} must include a non-empty string 'id'")
        if not isinstance(site_name, str) or not site_name.strip():
            raise ValueError(f"Provision input user at index {index} must include a non-empty string 'name'")

        user_ids.append(user_id)
        site_id_name_map[user_id] = site_name

    if len(set(user_ids)) != len(user_ids):
        raise ValueError("Provision input 'users' contains duplicate user ids")
    if len(set(site_id_name_map.values())) != len(site_id_name_map):
        raise ValueError("Provision input 'users' contains duplicate site names")

    return user_ids, site_id_name_map


def load_computation_parameters(provision_input: dict) -> dict:
    raw_parameters = provision_input.get('computation_parameters') or '{}'
    if isinstance(raw_parameters, dict):
        return dict(raw_parameters)
    if not isinstance(raw_parameters, str):
        raise ValueError("Provision input 'computation_parameters' must be a JSON object string or object")

    try:
        parameters = json.loads(raw_parameters)
    except json.JSONDecodeError as exc:
        raise ValueError("Provision input 'computation_parameters' must be valid JSON") from exc

    if not isinstance(parameters, dict):
        raise ValueError("Provision input 'computation_parameters' must decode to a JSON object")
    return parameters


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run provisioning script")
    
    # Input provision file argument
    parser.add_argument(
        '--input',
        default='/provisioning/provision_input.json',
        help='Path to the provision input file (default: /provisioning/provision_input.json)'
    )


    args = parser.parse_args()
    provision_input_path = args.input
    
    # Load provision input
    provision_input = load_provision_input(provision_input_path)
    path_run = os.path.join('/provisioning')

    # Extract arguments from provision input
    user_ids, site_id_name_map = validate_users(provision_input)
    computation_parameters_dict = load_computation_parameters(provision_input)
    computation_parameters_dict['site_id_name_map'] = site_id_name_map
    computation_parameters = json.dumps(computation_parameters_dict)
    fed_learn_port = provision_input.get('fed_learn_port')
    admin_port = provision_input.get('admin_port')
    host_identifier = provision_input.get('host_identifier')
    
    print(f'user_ids: {user_ids}')
    print(f'path_run: {path_run}')
    print(f'computation_parameters: {computation_parameters}')
    print(f'fed_learn_port: {fed_learn_port}')
    print(f'admin_port: {admin_port}')
    print(f'host_identifier: {host_identifier}')

    
    # Call the provision_run function with the loaded arguments
    provision_run(
        user_ids=user_ids,
        path_run=path_run,
        path_app="/workspace/app",
        computation_parameters=computation_parameters,
        fed_learn_port=fed_learn_port,
        admin_port=admin_port,
        host_identifier=host_identifier,
    )

if __name__ == '__main__':
    main()
