import os
import requests
import boto3
import argparse
import runpod
import asyncio
import traceback
import subprocess
import shutil

from pathlib import Path
from omegaconf import OmegaConf
from dotenv import load_dotenv
from runpod.serverless.modules.rp_logger import RunPodLogger
from concurrent.futures import ProcessPoolExecutor
from filelock import FileLock


logger = RunPodLogger()

# Your global model cache dictionary
global_models = {}

# Reuse your existing CONFIG_PATH and CHECKPOINT_PATH
CONFIG_PATH = Path("configs/unet/stage2_efficient.yaml")
CHECKPOINT_PATH = Path("checkpoints/latentsync_unet.pt")

load_dotenv()
# Environment Variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Check for missing critical values
if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, S3_BUCKET_NAME]):
    raise EnvironmentError(
        "Missing one or more AWS credentials or S3_BUCKET_NAME environment variables."
    )


def is_valid_video(path):
    try:
        # Run ffprobe (part of ffmpeg) to check if video is readable
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return bool(result.stdout.strip())  # True if codec info found
    except Exception:
        return False


def download_file(url: str, dest_path: str):
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


async def safe_download(url: str, dest_path: Path, validate_fn=None, max_retries=3):
    dest_path = Path(dest_path)
    for attempt in range(1, max_retries + 1):
        try:
            tmp_path = dest_path.with_suffix(".tmp")
            print(f"[safe_download] Downloading from {url} → {tmp_path}")

            # Download to a temporary file
            await asyncio.to_thread(download_file, url, tmp_path)

            # Move atomically after download finishes
            tmp_path.rename(dest_path)

            # Validate file if needed
            if dest_path.stat().st_size == 0:
                raise RuntimeError(f"Downloaded file is empty: {dest_path}")
            if validate_fn and not validate_fn(dest_path):
                raise RuntimeError(f"Invalid/corrupt file: {dest_path}")

            return  # Success

        except Exception as e:
            print(f"[safe_download] Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise


# Create a global S3 client on import to reuse connections
s3_client = boto3.client(
    "s3",
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
        url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_key}"
        print(f"Upload complete: {url}")
        return url

    except Exception as e:
        print(f"Error uploading to S3: {e}")
        raise e


# Function to load your model once per worker
def load_unet_model(config, checkpoint_path, device="cuda"):
    from latentsync.models.unet import UNet3DConditionModel  # adjust import

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        checkpoint_path,
        device=device,
    )
    return unet


def ensure_checkpoint(url: str, zip_path: Path, extract_dir: Path):
    lock_path = zip_path.with_suffix(".lock")
    with FileLock(lock_path):
        if extract_dir.exists() and any(extract_dir.iterdir()):
            return  # Already extracted

        # Download zip if missing
        if not zip_path.exists() or zip_path.stat().st_size == 0:
            tmp_path = zip_path.with_suffix(".tmp")
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            tmp_path.rename(zip_path)

        # Unzip
        shutil.unpack_archive(str(zip_path), str(extract_dir))


# Worker initializer, called once per process in the pool
def init_worker():
    global global_models
    config = OmegaConf.load(CONFIG_PATH)
    global_models["unet"] = load_unet_model(config, CHECKPOINT_PATH, device="cuda")
    print("[Worker] UNet model loaded.")

    # Ensure auxiliary checkpoint exists
    ensure_checkpoint(
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        zip_path=Path("checkpoints/auxiliary/models/buffalo_l.zip"),
        extract_dir=Path("checkpoints/auxiliary/models/buffalo_l"),
    )
    print("[Worker] Auxiliary checkpoint ensured and extracted.")


# Create process pool with initializer
process_pool = ProcessPoolExecutor(
    max_workers=2,  # tune this based on VRAM/CPU cores
    initializer=init_worker,
)


def run_inference(config, args):
    unet = global_models.get("unet")
    if unet is None:
        raise RuntimeError("UNet model not loaded in worker!")
    from scripts.inference import main

    main(config, args, unet=unet)


async def process_single_job(j):
    try:
        video_url = j["video_path"]
        audio_url = j["audio_path"]
        campaign_id = j["campaign_id"]
        lead_id = j["lead_id"]

        guidance_scale = j.get("guidance_scale", 1.5)
        inference_steps = j.get("inference_steps", 20)
        seed = j.get("seed", 1247)

        output_dir = Path(f"./outputs/{campaign_id}_{lead_id}")
        output_dir.mkdir(parents=True, exist_ok=True)

        video_path = output_dir / f"video_{lead_id}.mp4"
        audio_path = output_dir / f"audio_{lead_id}.wav"
        output_path = output_dir / f"{campaign_id}_{lead_id}.mp4"

        # Download video first (with retries + validation built in)
        await safe_download(video_url, str(video_path), validate_fn=is_valid_video)

        # Download audio next (same retries + validation)
        await safe_download(audio_url, str(audio_path))

        # Load config and prepare args in main process
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

        # Run inference in a separate process (CPU/GPU intensive)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(process_pool, run_inference, config, args)

        # Check if output file exists and is non-empty
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError(
                f"Inference did not produce output file for lead_id={lead_id} at {output_path}"
            )

        # Upload result concurrently (I/O bound)
        s3_key = f"{campaign_id}/{lead_id}.mp4"
        s3_url = await asyncio.to_thread(upload_to_s3, str(output_path), s3_key)

        return {
            "status": "success",
            "lead_id": lead_id,
            "campaign_id": campaign_id,
            "output_url": s3_url,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "status": "failed",
            "lead_id": j.get("lead_id"),
            "campaign_id": j.get("campaign_id"),
            "error": str(e),
        }


async def concurrent_handler(job):
    try:
        input_data = job.get("input", job)
        jobs = input_data.get("jobs", [])
        if not jobs:
            return {"status": "failed", "error": "No jobs provided."}

        # Run all jobs concurrently
        results = await asyncio.gather(*(process_single_job(j) for j in jobs))
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
