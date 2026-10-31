"""The plain GPU baseline must remain an optional dependency-free import."""
import os
from pathlib import Path
import subprocess
import sys


def test_linked_list_lru_import_does_not_load_device_frameworks():
    root=Path(__file__).resolve().parents[3]
    env=dict(os.environ,PYTHONPATH=str(root/'src'))
    code="import sys; import ulacache.eviction.kernels.linked_list_lru; assert all(x not in sys.modules for x in ('torch','triton','sglang'))"
    subprocess.run([sys.executable,'-c',code],check=True,env=env)
