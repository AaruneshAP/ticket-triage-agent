"""CLI entrypoint for incremental issue synchronization."""

import os
import sys

sys.path.insert(0, os.path.abspath("src"))

from triage.ingest.incremental import main

if __name__ == "__main__":
    main()

