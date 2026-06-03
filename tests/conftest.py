from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PARENT = PROJECT_ROOT.parent

if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))
