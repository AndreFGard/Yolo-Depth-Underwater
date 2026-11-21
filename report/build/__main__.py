from pathlib import Path
import subprocess
import sys

from tqdm import tqdm


def main():
    build_dir = Path(__file__).parent.resolve()
    scripts = [file for file in build_dir.iterdir() if file.suffix == ".py"]

    print(f"Found {len(scripts)} files")

    for file in tqdm(scripts):
        print(f"Running '{file}:")
        subprocess.run([sys.executable, file], check=True)
