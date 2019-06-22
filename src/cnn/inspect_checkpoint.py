# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('checkpoint_filename', None,
                           'The checkpoint to inspect')

if FLAGS.checkpoint_filename is None:
    sys.exit('The --checkpoint_filename option is required')
print_tensors_in_checkpoint_file(file_name=FLAGS_checkpoint_filename,tensor_name="",all_tensors=True)

