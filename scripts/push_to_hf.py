import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from data_pipeline.hf_uploader import prepare_and_push_to_hf

def main():
    parser = argparse.ArgumentParser(description="Push prepared IndusCraft dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repository ID (e.g., username/induscraft-dataset)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token")
    parser.add_argument("--private", action="store_true", help="Make repository private on HF Hub")
    args = parser.parse_args()

    prepare_and_push_to_hf(repo_id=args.repo_id, token=args.token, private=args.private)

if __name__ == "__main__":
    main()
