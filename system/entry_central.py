"""Start the central NVFlare service and submit the computation job."""

import os
import subprocess

from framework.errors import raise_for_terminal_errors
from nvflare.apis.job_def import JobMetaKey, RunStatus
from nvflare.fuel.flare_api.flare_api import Session, new_secure_session

# Path Constants
STARTUP_SCRIPT_DIRECTORY = "/workspace/runKit/server/startup"
STARTUP_SCRIPT_PATH = "/workspace/runKit/server/startup/start.sh"
ADMIN_DIRECTORY_PATH = "/workspace/runKit/admin"
JOB_DIRECTORY_PATH = "/workspace/runKit/job/"
ADMIN_USER_EMAIL = "admin@admin.com"


def start_server():
    """Start the provisioned NVFlare server process."""
    subprocess.run(
        ["/bin/bash", STARTUP_SCRIPT_PATH],
        cwd=STARTUP_SCRIPT_DIRECTORY,
        check=True,
    )


def job_status_callback(
    session: Session, job_id: str, job_meta, *cb_args, **cb_kwargs
) -> bool:
    """Log job status updates while monitoring a submitted job."""
    job_status = job_meta[JobMetaKey.STATUS.value]
    print(f"Job status: {job_status}")
    return True


def main():
    """Run the central service and wait for the job to finish."""
    start_server()
    session = new_secure_session(
        ADMIN_USER_EMAIL,
        ADMIN_DIRECTORY_PATH,
    )

    try:
        job_id = session.submit_job(JOB_DIRECTORY_PATH)
        session.monitor_job(
            job_id,
            timeout=3600,
            poll_interval=10,
            cb=job_status_callback,
        )
        job_meta = session.get_job_meta(job_id)
        job_status = job_meta[JobMetaKey.STATUS.value]
        if job_status != RunStatus.FINISHED_COMPLETED.value:
            raise_for_terminal_errors(os.getenv("OUTPUT_DIR", "/workspace/output"))
            raise RuntimeError(f"Job {job_id} ended with status {job_status}")
    finally:
        session.shutdown("all")


if __name__ == "__main__":
    main()
