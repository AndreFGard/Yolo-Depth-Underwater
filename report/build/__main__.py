from pathlib import Path
import subprocess
import sys


def main() -> None:
    build_dir = Path(__file__).parent.resolve()
    entrypoint = Path(__file__).resolve()
    scripts = sorted(
        file for file in build_dir.glob("*.py") if file.resolve() != entrypoint
    )

    print(f"Found {len(scripts)} files")

    for file in scripts:
        print(f"Running {file.name}:", flush=True)
        subprocess.run([sys.executable, file], check=True)


if __name__ == "__main__":
    main()
