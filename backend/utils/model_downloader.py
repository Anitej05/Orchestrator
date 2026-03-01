import os
import logging
from pathlib import Path

# Try importing required packages
try:
    from huggingface_hub import hf_hub_download
    import sentence_transformers
except ImportError as e:
    logging.warning(f"Model downloading dependencies missing: {e}. Please ensure huggingface-hub and sentence-transformers are installed.")
    hf_hub_download = None

logger = logging.getLogger(__name__)

def download_models_if_missing():
    """
    Checks if necessary models (Kokoro, All-MPNet) are downloaded locally.
    If not, it downloads them securely.
    Ensures that when the unified agent server starts, all models are ready for Inference without delay.
    """
    logger.info("Initializing Model Downloader Check...")
    
    if not hf_hub_download:
        logger.warning("Skipping model autodownload. `huggingface_hub` not available.")
        return

    # 1. Download Kokoro Model Weights
    kokoro_repo = "hexgrad/Kokoro-82M"
    kokoro_dir = Path(__file__).parent.parent / "models" / "kokoro"
    kokoro_dir.mkdir(parents=True, exist_ok=True)
    
    # We specifically need kokoro-v1_0.onnx from HF, and voices.bin from GitHub releases
    
    # Download ONNX model from HuggingFace
    onnx_file = "kokoro-v1_0.onnx"
    target_onnx = kokoro_dir / onnx_file
    if not target_onnx.exists():
        logger.info(f"Downloading Kokoro ONNX model '{onnx_file}' to {kokoro_dir}...")
        try:
            hf_hub_download(
                repo_id=kokoro_repo,
                filename=onnx_file,
                local_dir=kokoro_dir,
                local_dir_use_symlinks=False
            )
            logger.info(f"✅ Successfully downloaded {onnx_file}")
        except Exception as e:
            logger.warning(f"⚠️ Kokoro TTS model not available: {e}")
            logger.info(f"TTS features will be disabled. The app will continue to function normally.")
    else:
        logger.debug(f"Kokoro ONNX model '{onnx_file}' already exists.")

    # Download voices.bin from GitHub
    voices_file = "voices.bin"
    target_voices = kokoro_dir / voices_file
    if not target_voices.exists():
        logger.info(f"Downloading Kokoro voices metadata '{voices_file}' to {kokoro_dir}...")
        try:
            import requests
            url = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/{voices_file}"
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(target_voices, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"✅ Successfully downloaded {voices_file}")
        except Exception as e:
            # Fallback to voices.json if voices.bin fails
            logger.warning(f"Failed to download voices.bin: {e}. Trying voices.json...")
            fallback_file = "voices.json"
            target_fallback = kokoro_dir / fallback_file
            if not target_fallback.exists():
                try:
                    url = f"https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/{fallback_file}"
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(target_fallback, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    logger.info(f"✅ Successfully downloaded {fallback_file}")
                except Exception as ex:
                    logger.error(f"❌ Failed to download fallback {fallback_file}: {ex}")
    else:
        logger.debug(f"Kokoro voices metadata '{voices_file}' already exists.")

    # 2. Download All-MPNet-Base-V2 (Sentence Transformers)
    # This automatically downloads to the HF Cache if it doesn't exist
    logger.info("Verifying all-mpnet-base-v2 is cached locally...")
    try:
        # Disable JAX to avoid jaxlib dependency issues during loading
        os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
        os.environ['JAX_PLATFORMS'] = ''
        
        from sentence_transformers import SentenceTransformer
        # Calling this will trigger the download if it's missing from ~/.cache/huggingface
        model = SentenceTransformer('all-mpnet-base-v2')
        logger.info("✅ all-mpnet-base-v2 is ready.")
    except Exception as e:
        logger.error(f"❌ Failed to verify sentence-transformers model: {e}")

    # 3. Check and Install Playwright Browsers (Chromium)
    logger.info("Verifying Playwright browser binaries...")
    try:
        import subprocess
        # Run playwright install chromium. It's safe to run multiple times.
        # We run it synchronously so the browser agent doesn't crash on first boot.
        result = subprocess.run(
            ["python", "-m", "playwright", "install", "chromium"], 
            capture_output=True, 
            text=True,
            check=False
        )
        if result.returncode == 0:
            logger.info("✅ Playwright browsers are ready.")
        else:
            logger.error(f"❌ Failed to install Playwright browsers. Exit code {result.returncode}:\n{result.stderr}")
    except Exception as e:
        logger.error(f"❌ Failed to trigger Playwright install: {e}")
