import os
from pathlib import Path

# Derive project root from this file's location rather than CWD-based git
# discovery, which breaks with nested git repos (e.g. third_party/IsaacLab).
# __file__ is at source/lehome/lehome/utils/constant.py → 5 parents up = project root
git_root = str(Path(os.path.abspath(__file__)).parent.parent.parent.parent.parent)

ASSETS_ROOT = os.path.join(git_root, 'Assets')
