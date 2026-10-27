import os
import sys

# Crucial Windows fix for pyarrow access violation 0xC0000005
os.environ["PYARROW_IGNORE_TIMEZONE"] = "1"

# Ensure repository root is on sys.path
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
