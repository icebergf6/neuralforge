"""
LeagueOfLLMs — Smart Model Download Manager v2

Features:
- Download model dari HuggingFace Hub API
- Progress bar dengan resume support (Range header)
- SHA256 checksum verification
- Auto-detect & list available models
- Multi-model support: TinyStories 15M / 42M / 110M
"""

import os
import sys
import json
import time
import hashlib
import urllib.request
import urllib.error
import urllib.parse

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# ── Model Registry ─────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "stories15M": {
        "description": "TinyStories 15M — smallest, fastest (60 MB)",
        "url":         "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin",
        "filename":    "stories15M.bin",
        "sha256":      None,   # Not available from source
        "size_mb":     60,
    },
    "stories42M": {
        "description": "TinyStories 42M — medium quality (167 MB)",
        "url":         "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories42M.bin",
        "filename":    "stories42M.bin",
        "sha256":      None,
        "size_mb":     167,
    },
    "stories110M": {
        "description": "TinyStories 110M — best quality (435 MB)",
        "url":         "https://huggingface.co/karpathy/tinyllamas/resolve/main/stories110M.bin",
        "filename":    "stories110M.bin",
        "sha256":      None,
        "size_mb":     435,
    },
    "tokenizer": {
        "description": "Llama 2 BPE Tokenizer (vocab 32k) — required for all models",
        "url":         "https://github.com/karpathy/llama2.c/raw/master/tokenizer.bin",
        "filename":    "tokenizer.bin",
        "sha256":      None,
        "size_mb":     0.4,
    }
}


# ── Progress Bar ───────────────────────────────────────────────────────────────
class ProgressBar:
    def __init__(self, total: int, width: int = 40):
        self.total    = total
        self.width    = width
        self.start_t  = time.time()

    def update(self, downloaded: int):
        if self.total <= 0:
            sys.stdout.write(f"\r  Downloaded: {downloaded // 1024 // 1024:.1f} MB")
            sys.stdout.flush()
            return

        pct    = min(downloaded / self.total, 1.0)
        filled = int(self.width * pct)
        bar    = "█" * filled + "░" * (self.width - filled)

        elapsed = time.time() - self.start_t
        speed   = (downloaded / elapsed / 1024 / 1024) if elapsed > 0 else 0
        eta     = ((self.total - downloaded) / (downloaded / elapsed)) if downloaded > 0 and elapsed > 0 else 0

        dl_mb  = downloaded / 1024 / 1024
        tot_mb = self.total  / 1024 / 1024

        sys.stdout.write(
            f"\r  [{bar}] {pct*100:5.1f}%  {dl_mb:.1f}/{tot_mb:.1f} MB  "
            f"{speed:.1f} MB/s  ETA: {int(eta)}s   "
        )
        sys.stdout.flush()

    def done(self):
        elapsed = time.time() - self.start_t
        print(f"\r  [{'█' * self.width}] 100.0%  done in {elapsed:.1f}s            ")


# ── Download with Resume ───────────────────────────────────────────────────────
def download_file(url: str, target_path: str, expected_sha256: str = None) -> bool:
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    # Get file size from server
    try:
        req  = urllib.request.Request(url, method="HEAD",
                    headers={"User-Agent": "LeagueOfLLMs/2.0"})
        with urllib.request.urlopen(req, timeout=10) as r:
            total_size = int(r.headers.get("Content-Length", 0))
    except Exception:
        total_size = 0

    # Check if already fully downloaded
    existing = os.path.getsize(target_path) if os.path.exists(target_path) else 0
    if existing == total_size and total_size > 0:
        print(f"  [OK] Already downloaded: {target_path}")
        return True

    # Resume download if partial
    resume_from = existing if existing > 0 and existing < total_size else 0
    if resume_from:
        print(f"  [Resume] Resuming from {resume_from // 1024 // 1024:.1f} MB...")

    print(f"  [Download] {os.path.basename(target_path)}")
    print(f"  URL: {url}")

    pbar = ProgressBar(total=total_size)
    sha  = hashlib.sha256()

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "LeagueOfLLMs/2.0",
                "Range":      f"bytes={resume_from}-" if resume_from else ""
            }
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            mode       = "ab" if resume_from else "wb"
            downloaded = resume_from

            with open(target_path, mode) as f:
                # Hash existing data if resuming
                if resume_from:
                    with open(target_path, "rb") as existing_f:
                        sha.update(existing_f.read())

                while True:
                    chunk = response.read(64 * 1024)  # 64KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    sha.update(chunk)
                    downloaded += len(chunk)
                    pbar.update(downloaded)

        pbar.done()

    except urllib.error.URLError as e:
        print(f"\n  [Error] Download failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n  [Interrupted] Partial file saved. Re-run to resume.")
        return False

    # Verify SHA256 if provided
    if expected_sha256:
        actual = sha.hexdigest()
        if actual.lower() == expected_sha256.lower():
            print(f"  [OK] SHA256 verified.")
        else:
            print(f"  [ERROR] SHA256 mismatch!")
            print(f"    Expected: {expected_sha256}")
            print(f"    Got:      {actual}")
            return False

    print(f"  [Saved] {target_path}")
    return True


# ── HuggingFace Hub Model Search ──────────────────────────────────────────────
def search_hf_models(query: str = "gguf llama", limit: int = 5) -> list[dict]:
    """Search HuggingFace Hub for models (requires internet)."""
    try:
        params  = urllib.parse.urlencode({"search": query, "limit": limit, "filter": "gguf"})
        url     = f"https://huggingface.co/api/models?{params}"
        req     = urllib.request.Request(url, headers={"User-Agent": "LeagueOfLLMs/2.0"})
        with urllib.request.urlopen(req, timeout=8) as r:
            models = json.loads(r.read().decode("utf-8"))
            return [
                {
                    "id":          m.get("id", ""),
                    "downloads":   m.get("downloads", 0),
                    "likes":       m.get("likes", 0),
                    "tags":        m.get("tags", [])[:3],
                }
                for m in models[:limit]
            ]
    except Exception as e:
        return [{"error": str(e)}]


# ── List Downloaded Models ─────────────────────────────────────────────────────
def list_downloaded_models() -> list[dict]:
    if not os.path.exists(MODELS_DIR):
        return []

    models = []
    for fname in os.listdir(MODELS_DIR):
        fpath = os.path.join(MODELS_DIR, fname)
        if os.path.isfile(fpath):
            size_mb = os.path.getsize(fpath) / 1024 / 1024
            models.append({
                "filename": fname,
                "size_mb":  round(size_mb, 1),
                "path":     fpath,
                "known":    fname in [m["filename"] for m in MODEL_REGISTRY.values()]
            })
    return models


# ── Main CLI ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LeagueOfLLMs — Model Download Manager v2")
    print("=" * 60)

    # Show currently downloaded models
    downloaded = list_downloaded_models()
    if downloaded:
        print("\nCurrently in models/:")
        for m in downloaded:
            print(f"  {m['filename']:30s} {m['size_mb']:8.1f} MB")
    else:
        print("\n  No models downloaded yet.")

    print("\nAvailable models to download:")
    model_names = list(MODEL_REGISTRY.keys())
    for i, name in enumerate(model_names):
        info = MODEL_REGISTRY[name]
        # Check if already downloaded
        target = os.path.join(MODELS_DIR, info["filename"])
        status = "[downloaded]" if os.path.exists(target) else f"~{info['size_mb']} MB"
        print(f"  {i+1}. {name:15s} — {info['description']} ({status})")

    print("\nOptions:")
    print("  all   — Download tokenizer + stories15M (recommended)")
    print("  1-4   — Download specific model by number")
    print("  hf    — Search HuggingFace Hub")
    print()

    choice = input("Your choice [all]: ").strip().lower() or "all"

    if choice == "all":
        targets = ["tokenizer", "stories15M"]
    elif choice == "hf":
        query = input("Search query [llama gguf]: ").strip() or "llama gguf"
        results = search_hf_models(query)
        print("\nHuggingFace Models:")
        for m in results:
            if "error" not in m:
                print(f"  {m['id']} — {m['downloads']:,} downloads, {m['likes']} likes")
        return
    elif choice.isdigit() and 1 <= int(choice) <= len(model_names):
        targets = [model_names[int(choice) - 1]]
    else:
        targets = choice.split()

    print()
    os.makedirs(MODELS_DIR, exist_ok=True)

    for target_name in targets:
        if target_name not in MODEL_REGISTRY:
            print(f"  [Skip] Unknown model: {target_name}")
            continue
        info  = MODEL_REGISTRY[target_name]
        path  = os.path.join(MODELS_DIR, info["filename"])
        ok    = download_file(info["url"], path, info.get("sha256"))
        if ok:
            print(f"  [Done] {target_name} ready.")
        else:
            print(f"  [Fail] {target_name} download failed.")
        print()

    print("All done! Run: python tools/serve.py")


if __name__ == "__main__":
    main()
