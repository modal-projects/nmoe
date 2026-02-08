"""Interactive debug environment for Modal.

Usage:
  modal shell modal/debug.py
"""

import modal

from image import nmoe_build_image, SOURCE_MOUNT_IGNORE, REPO_ROOT, data_vol

# Extend the build image with debug deps, then add source mount last.
# (Can't pip_install after a copy=False mount, so we start from the build image.)
debug_image = nmoe_build_image.uv_pip_install("psutil", "pytest", "ipdb").add_local_dir(
    str(REPO_ROOT), "/workspace/nmoe", ignore=SOURCE_MOUNT_IGNORE
)

app = modal.App("nmoe-debug")


@app.function(image=debug_image, gpu="B200", volumes={"/data": data_vol}, timeout=86400)
def debug():
    pass
