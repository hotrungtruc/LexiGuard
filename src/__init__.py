"""LexiGuard AI Package Root"""

import os

# Prevent pandas/pyarrow timezone initialization from loading an incompatible
# native path on Windows before Neo4j or Docling starts.
os.environ.setdefault("PYARROW_IGNORE_TIMEZONE", "1")

__version__ = "0.1.0"
