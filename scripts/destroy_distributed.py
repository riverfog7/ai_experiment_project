#!/usr/bin/env -S uv run

import argparse
import os

import dotenv
import runpod

dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())

def get_api_key():
    return os.getenv("RUNPOD_API_KEY")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Destroy Distributed Pods on RunPod.io")
    parser.add_argument(
        "--save-file",
        type=str,
        default=".pod_ids",
        help="File containing pod IDs to destroy",
    )
    args = parser.parse_args()

    with open(args.save_file, "r") as f:
        pod_ids = [line.strip() for line in f.readlines()]

    runpod.set_credentials(api_key=get_api_key(), overwrite=True)
    for pod_id in pod_ids:
        try:
            pod = runpod.terminate_pod(pod_id)
            print(f"Terminated pod {pod_id} successfully.")
        except Exception as e:
            print(f"Failed to terminate pod {pod_id}: {e}")

    os.remove(args.save_file)
