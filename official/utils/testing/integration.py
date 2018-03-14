# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Helper code to run complete models from within python.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import shutil
import subprocess
import sys
import time


PYTHON_CMD = "python2" if sys.version_info[0] == 2 else "python3"


def run_synthetic(file_path):
  """Performs a minimal run of a model.

    This function is intended to test for syntax errors throughout a model. A
  very limited run is performed using synthetic data.

  Args:
    file_path: The absolute path of a python script to be tested.

  Raises:
    OSError: Any error detected in the run causes an OSError. Because the code
      is run as a subprocess, it is not able to manifest the specific python
      error that was raised.
  """

  model_dir = "/tmp/model_test_{}".format(hash(time.time()))
  if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

  args = [PYTHON_CMD, file_path, "--model_dir", model_dir,
          "--train_epochs", "1", "--epochs_per_eval", "1",
          "--use_synthetic_data",
          "--synthetic_batches_per_epoch", "10"]

  p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE)
  output, err = p.communicate()
  returncode = p.returncode

  if os.path.exists(model_dir):
    shutil.rmtree(model_dir)

  if returncode != 0:
    raise OSError("Run failed:\nStdout:\n{}\n\nStderr:\n{}".format(
        output.decode("utf-8"), err.decode("utf-8")
    ))
