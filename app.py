"""Streamlit Dashboard entry point for FaultMesh AI 2.0.

Provides root-level entry point supporting both 'streamlit run app.py'
and 'streamlit run dashboard/app.py' seamlessly on Streamlit Community Cloud.
"""

import sys
from pathlib import Path
import runpy

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Delegate execution to the complete FaultMesh AI 2.0 dashboard
dashboard_entry = REPO_ROOT / "dashboard" / "app.py"
runpy.run_path(str(dashboard_entry), run_name="__main__")
