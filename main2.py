import os
import requests
import argparse
from pathlib import Path
from tqdm import tqdm

def download_model(url, save_path):
    """
    Download the model weights from a given URL and save them to the specified path.
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an error if the request failed

    # Save the file
    with open(save_path, 'wb') as f:
        for chunk in tqdm(response.iter_content(chunk_size=1024), desc=f"Downloading {os.path.basename(save_path)}"):
            if chunk:
                f.write(chunk)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Download Hugging Face models and save to specified paths.")
    parser.add_argument(
        '--models', type=str, required=True,
        help='List of models in the format: URL1:PATH1,URL2:PATH2 (comma-separated pairs).'
    )
    args = parser.parse_args()

    # Parse the models argument
    model_pairs = args.models.split(",")
    models = [
        {"url": pair.split("::")[0], "path": pair.split("::")[1]}
        for pair in model_pairs
    ]

    print("HEHEH")
    print(models)



    # Ensure the directories exist and download the models
    for model in models:
        save_dir = Path(model["path"])
        save_dir.mkdir(parents=True, exist_ok=True)

        save_file = save_dir / "model.bin"
        print(f"Downloading from {model['url']} to {save_file}")
        download_model(model["url"], save_file)

if __name__ == "__main__":
    main()
