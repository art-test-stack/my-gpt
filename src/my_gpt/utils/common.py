from pathlib import Path
import os


def slugify(text: str) -> str:
    """Convert text to a slug suitable for filenames and URLs."""
    return "".join(c if c.isalnum() else "_" for c in text.replace(" ", "-")).lower()

def get_repo_dir():
    if os.getenv("MY_GPT_BASE_DIR"):
        return Path(os.getenv("MY_GPT_BASE_DIR"))
    
    else:
        home_dir = Path.home()
        cache_dir = home_dir / ".my_gpt"
        repo_dir = cache_dir / "my-gpt"
        return repo_dir