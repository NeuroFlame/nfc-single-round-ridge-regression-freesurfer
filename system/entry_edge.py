import os
import subprocess

from framework.errors import raise_for_terminal_errors


STARTUP_SCRIPT_PATH = "/workspace/runKit/startup/sub_start.sh"


def main():
    completed_process = subprocess.run(["/bin/bash", STARTUP_SCRIPT_PATH], check=False)
    raise_for_terminal_errors(os.getenv("OUTPUT_DIR", "/workspace/output"))
    completed_process.check_returncode()


if __name__ == "__main__":
    main()
