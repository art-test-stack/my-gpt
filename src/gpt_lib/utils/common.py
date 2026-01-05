from pathlib import Path
import os


def slugify(text: str) -> str:
    """Convert text to a slug suitable for filenames and URLs."""
    return "".join(c if c.isalnum() else "_" for c in text.replace(" ", "-")).lower()

def get_repo_dir():
    if os.getenv("GPT_LIB_BASE_DIR"):
        return Path(os.getenv("GPT_LIB_BASE_DIR"))
    
    else:
        home_dir = Path.home()
        cache_dir = home_dir / ".gpt_lib"
        repo_dir = cache_dir / "gpt_lib"
        return repo_dir