# The MIT License (MIT)
#
# Copyright (c) 2018-2026 CubitPy Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Provide functions to transfer files between the local and remote
machines."""

import subprocess  # nosec B404
from pathlib import Path, PureWindowsPath

from cubitpy import cupy


def transfer_file_from_remote(
    remote_path: Path | PureWindowsPath, local_path: Path
) -> None:
    """Copy a file from the remote machine to the local host."""

    if not cupy.is_remote():
        raise RuntimeError("File transfer is only supported for remote connections.")

    ssh_user = cupy.get_remote_user()
    ssh_host = cupy.get_remote_host()
    cmd = ["scp", f"{ssh_user}@{ssh_host}:{remote_path.as_posix()}", str(local_path)]
    subprocess.check_output(cmd, stderr=subprocess.STDOUT)  # nosec B603
