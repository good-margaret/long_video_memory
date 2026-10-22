"""Allow `python -m memvid ...` (equivalent to the `memvid` console script)."""

import sys

from memvid.cli import main

sys.exit(main())
