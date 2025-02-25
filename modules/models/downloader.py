import argparse
import os
import requests
from huggingface_hub import hf_hub_download

def download_model(repo_id: str = None, filename: str = None, cache_dir: str = None, custom_url: str = None) -> str:
    """
    Download a model checkpoint from Hugging Face Hub or a custom URL.

    Args:
        repo_id (str, optional): The repository ID on Hugging Face Hub (e.g., "ultralytics/yolov8").
        filename (str, optional): The filename of the model checkpoint to download.
        cache_dir (str, optional): A directory to cache the downloaded model.
        custom_url (str, optional): A custom URL to download the model from.

    Returns:
        str: The local file path to the downloaded model checkpoint.
    """
    if custom_url:
        # Downloading from a custom URL using requests
        local_file_name = os.path.basename(custom_url)
        print(f"Downloading from custom URL: {custom_url}")
        response = requests.get(custom_url, stream=True)
        response.raise_for_status()  # Ensure we catch HTTP errors
        with open(local_file_name, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Model downloaded to: {local_file_name}")
        return os.path.abspath(local_file_name)
    else:
        # Downloading using Hugging Face Hub
        if not repo_id or not filename:
            raise ValueError("repo_id and filename must be provided when custom_url is not used.")
        local_file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
        print(f"Model downloaded to: {local_file_path}")
        return local_file_path

def main():
    parser = argparse.ArgumentParser(
        description="Download a model checkpoint from Hugging Face Hub or a custom URL."
    )
    
    # Mutually exclusive group: either custom_url or repo_id/filename
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--custom_url",
        type=str,
        help="The custom URL to download the model from."
    )
    group.add_argument(
        "--hf",
        action="store_true",
        help="Flag to indicate downloading from Hugging Face Hub (requires --repo_id and --filename)."
    )
    
    parser.add_argument(
        "--repo_id",
        type=str,
        help="The repository ID on Hugging Face Hub (e.g., 'ultralytics/yolov8')."
    )
    parser.add_argument(
        "--filename",
        type=str,
        help="The filename to download from the repository (e.g., 'yolov8n-seg.pt')."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Optional directory to cache the downloaded model."
    )

    args = parser.parse_args()

    if args.custom_url:
        download_model(custom_url=args.custom_url)
    else:
        if not args.repo_id or not args.filename:
            parser.error("When using the Hugging Face Hub, --repo_id and --filename are required.")
        download_model(repo_id=args.repo_id, filename=args.filename, cache_dir=args.cache_dir)

if __name__ == "__main__":
    main()
