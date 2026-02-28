"""
Environment setup helper.

Usage (from the `fraud_detection_project` folder):

    python setup_env.py

This script will:
- Create a virtual environment `venv` if it does not exist
- Install all dependencies from `requirements.txt` into `venv`
- Download NLTK data: punkt, stopwords
- Download spaCy model: en_core_web_sm
"""

import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(os.path.dirname(PROJECT_ROOT), "venv")


def run(cmd, env=None):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    # Create venv if needed
    if not os.path.exists(VENV_DIR):
        print(f"Creating virtual environment at: {VENV_DIR}")
        run(f"{sys.executable} -m venv \"{VENV_DIR}\"")
    else:
        print(f"Virtual environment already exists at: {VENV_DIR}")

    # Path to python inside venv (Windows vs others)
    if os.name == "nt":
        py_path = os.path.join(VENV_DIR, "Scripts", "python.exe")
    else:
        py_path = os.path.join(VENV_DIR, "bin", "python")

    if not os.path.exists(py_path):
        raise SystemExit(f"Could not find python inside venv at: {py_path}")

    # Install requirements
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    if not os.path.exists(req_path):
        raise SystemExit(f"requirements.txt not found at: {req_path}")

    run(f"\"{py_path}\" -m pip install --upgrade pip")
    run(f"\"{py_path}\" -m pip install -r \"{req_path}\"")

    # Download NLTK data
    run(f"\"{py_path}\" -m nltk.downloader punkt stopwords")

    # Download spaCy model
    run(f"\"{py_path}\" -m spacy download en_core_web_sm")

    print("Environment setup complete.")


if __name__ == "__main__":
    main()

