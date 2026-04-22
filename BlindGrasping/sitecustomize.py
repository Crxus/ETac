"""
Automatically add the parent directory to `sys.path` when executing Python with BlindGrasping as the working directory,
enabling imports like `from ETac.etac import ETac` to work.

sitecustomize will be automatically imported by Python during startup if it is found in the current sys.path. 
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
etac_pkg_dir = os.path.join(project_root, "ETac")

if os.path.isdir(etac_pkg_dir) and project_root not in sys.path:
    sys.path.insert(0, project_root)
