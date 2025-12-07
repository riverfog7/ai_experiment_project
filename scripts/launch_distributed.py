#!/usr/bin/env -S uv run

import argparse
import os
import time

import dotenv
import requests

dotenv.load_dotenv(dotenv_path=dotenv.find_dotenv())

GPU_ID="NVIDIA GeForce RTX 3090"
COOLDOWN_MIN=5
TEMPLATE_ID="4pwm2f7ygg"
NAMING_SCHEME="search-worker-{}"

def get_api_key():
    return os.getenv("RUNPOD_API_KEY")

def create_pod(id: int, save_file: str):
    url = "https://rest.runpod.io/v1/pods"
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Content-Type": "application/json"
    }
    payload = {
        "name": NAMING_SCHEME.format(id),
        "templateId": TEMPLATE_ID,
        "gpuTypeIds": [GPU_ID],
        "cloudType": "COMMUNITY",
        "interruptible": True,
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code < 300:
        pod_id = response.json().get("id")
        with open(save_file, "a") as f:
            f.write(f"{pod_id}\n")
        print(f"Created pod {pod_id} with name {NAMING_SCHEME.format(id)}")
    else:
        raise Exception(f"Failed to create pod: {response.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch Distributed Pods on RunPod.io")
    parser.add_argument(
        "--num-pods",
        type=int,
        default=8,
        help="Number of pods to launch",
    )
    parser.add_argument(
        "--save-file",
        type=str,
        default=".pod_ids",
        help="File to save launched pod IDs",
    )
    args = parser.parse_args()

    for i in range(args.num_pods):
        create_pod(i, args.save_file)
        if i < args.num_pods - 1:
            print(f"Waiting for {COOLDOWN_MIN} minutes before launching next pod...")
            time.sleep(COOLDOWN_MIN * 60)
