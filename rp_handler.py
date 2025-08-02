from pathlib import Path
from scripts.inference import main
from omegaconf import OmegaConf
import argparse
import requests
import os
import boto3
from dotenv import load_dotenv
import runpod
from runpod.serverless.modules.rp_logger import RunPodLogger

logger = RunPodLogger()

# Reuse your existing CONFIG_PATH and CHECKPOINT_PATH
CONFIG_PATH = Path("configs/unet/stage2_efficient.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")

load_dotenv()
# Environment Variables
AWS_REGION = os.getenv("AWS_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Check for missing critical values
if not all([AWS_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
    raise EnvironmentError(
        "Missing one or more AWS credentials or S3_BUCKET_NAME environment variables."
    )


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# Create a global S3 client on import to reuse connections
s3_client = boto3.client(
    "s3",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
)


def upload_to_s3(local_file_path: str, s3_key: str) -> str:
    """
    Uploads a local file to AWS S3 and returns the public URL.

    Args:
        local_file_path: Path to the local file.
        s3_key: Key/path in the S3 bucket where the file will be stored.
        bucket_name: Optional; defaults to S3_BUCKET_NAME env variable.

    Returns:
        Public URL of the uploaded file.
    """

    try:
        print(f"Uploading {local_file_path} to s3://{S3_BUCKET_NAME}/{s3_key}...")
        s3_client.upload_file(local_file_path, S3_BUCKET_NAME, s3_key)
        url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{s3_key}"
        print(f"Upload complete: {url}")
        return url

    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise e


def concurrent_handler(job):
    """
    RunPod concurrent_handler that processes multiple jobs in a single request.

    Expects job like:
    {
        "input": {
            "jobs": [
                {
                    "video_path": "...",
                    "audio_path": "...",
                    "campaign_id": "...",
                    "lead_id": "...",
                    "guidance_scale": 1.5,
                    "inference_steps": 20,
                    "seed": 1234
                },
                ...
            ]
        }
    }
    """
    try:
        input_data = job.get("input", job)
        jobs = input_data.get("jobs", [])

        if not jobs:
            return {"status": "failed", "error": "No jobs provided."}

        results = []

        for j in jobs:
            try:
                video_url = j["video_path"]
                audio_url = j["audio_path"]
                campaign_id = j["campaign_id"]
                lead_id = j["lead_id"]

                guidance_scale = j.get("guidance_scale", 1.5)
                inference_steps = j.get("inference_steps", 20)
                seed = j.get("seed", 1247)

                output_dir = Path(f"./outputs/temp_{campaign_id}_{lead_id}")
                output_dir.mkdir(parents=True, exist_ok=True)

                video_path = output_dir / "video.mp4"
                audio_path = output_dir / "audio.wav"
                output_path = output_dir / f"{campaign_id}_{lead_id}.mp4"

                print(f"[{campaign_id}-{lead_id}] Downloading video...")
                download_file(video_url, str(video_path))
                print(f"[{campaign_id}-{lead_id}] Downloading audio...")
                download_file(audio_url, str(audio_path))

                config = OmegaConf.load(CONFIG_PATH)
                config["run"].update(
                    {
                        "guidance_scale": guidance_scale,
                        "inference_steps": inference_steps,
                    }
                )

                args = create_args(
                    str(video_path),
                    str(audio_path),
                    str(output_path),
                    inference_steps,
                    guidance_scale,
                    seed,
                )

                print(f"[{campaign_id}-{lead_id}] Running inference...")
                main(config=config, args=args)

                s3_key = f"{campaign_id}/{lead_id}.mp4"
                s3_url = upload_to_s3(str(output_path), s3_key)

                results.append(
                    {
                        "status": "success",
                        "lead_id": lead_id,
                        "campaign_id": campaign_id,
                        "output_url": s3_url,
                    }
                )

            except Exception as e:
                traceback.print_exc()
                results.append(
                    {
                        "status": "failed",
                        "lead_id": j.get("lead_id"),
                        "campaign_id": j.get("campaign_id"),
                        "error": str(e),
                    }
                )

        return {"results": results}

    except Exception as e:
        traceback.print_exc()
        return {
            "status": "failed",
            "error": str(e),
            "hint": "Check input format or unexpected top-level error.",
        }


def create_args(
    video_path, audio_path, output_path, inference_steps, guidance_scale, seed
):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--temp_dir", type=str, default="temp")
    parser.add_argument("--seed", type=int, default=1247)
    parser.add_argument("--enable_deepcache", action="store_true")

    return parser.parse_args(
        [
            "--inference_ckpt_path",
            CHECKPOINT_PATH.absolute().as_posix(),
            "--video_path",
            video_path,
            "--audio_path",
            audio_path,
            "--video_out_path",
            output_path,
            "--inference_steps",
            str(inference_steps),
            "--guidance_scale",
            str(guidance_scale),
            "--seed",
            str(seed),
            "--temp_dir",
            "temp",
            "--enable_deepcache",
        ]
    )


if __name__ == "__main__":
    import traceback

    try:
        logger.info("Starting RunPod Serverless...")
        runpod.serverless.start({"handler": concurrent_handler, "concurrency": 2})
    except Exception as e:
        traceback.print_exc()
        logger.error(f"Ran into an error: {e}")
        raise e
