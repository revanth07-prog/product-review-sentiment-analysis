"""
Utility script to download and extract pre-trained GloVe embeddings.
Downloads glove.6B.zip from Stanford / HuggingFace mirror and extracts glove.6B.100d.txt.
"""

import os
import sys
import zipfile
import urllib.request
import time

GLOVE_URLS = [
    "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip",
    "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip",
    "https://nlp.stanford.edu/data/glove.6B.zip",
]

def download_glove(dest_dir="embeddings", dim=100):
    target_file = os.path.join(dest_dir, f"glove.6B.{dim}d.txt")
    if os.path.exists(target_file) and os.path.getsize(target_file) > 1000000:
        print(f"GloVe file already exists at: {target_file}")
        return target_file

    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "glove.6B.zip")

    downloaded = False
    for url in GLOVE_URLS:
        try:
            print(f"Attempting download from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, "wb") as out_file:
                total_size = int(response.headers.get("Content-Length", 0))
                chunk_size = 1024 * 1024  # 1MB
                downloaded_size = 0
                start_time = time.time()
                last_print = start_time

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded_size += len(chunk)
                    now = time.time()
                    if now - last_print >= 5:
                        pct = (downloaded_size / total_size * 100) if total_size else 0
                        speed = (downloaded_size / (1024 * 1024)) / (now - start_time)
                        print(f"Downloaded {downloaded_size / (1024*1024):.1f} MB ({pct:.1f}%) at {speed:.2f} MB/s...")
                        last_print = now

            print(f"Download complete! Saved to {zip_path}")
            downloaded = True
            break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)

    if not downloaded or not os.path.exists(zip_path):
        raise RuntimeError("Failed to download glove.6B.zip from all available mirrors.")

    print(f"Extracting glove.6B.{dim}d.txt from {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extract(f"glove.6B.{dim}d.txt", dest_dir)
        # Also extract 50d if present for lightweight options
        if "glove.6B.50d.txt" in z.namelist() and not os.path.exists(os.path.join(dest_dir, "glove.6B.50d.txt")):
            z.extract("glove.6B.50d.txt", dest_dir)

    print(f"Extracted successfully: {target_file}")
    
    # Clean up the large zip file to save disk space
    try:
        os.remove(zip_path)
        print("Cleaned up temporary zip archive.")
    except Exception as e:
        print(f"Notice: Could not remove zip file: {e}")

    return target_file

if __name__ == "__main__":
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    download_glove(dim=dim)
