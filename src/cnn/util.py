from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import numpy as np

from constants import *

def parse_int_list(in_str, num_vals, low_bound=None, high_bound=None):
    """
    Given a string of the form "123,456,789", separate the string
    into a list of integers.  The integers are tested against the given
    bounds, if any.
    """
    vL = [int(elt.strip()) for elt in in_str.split(',')]
    assert len(vL) == num_vals, 'Wrong number of values in %s' % in_str
    if lower_bounds is not None:
        assert [v >= b for v, b in zip(vL, low_bound)], 'invalid low value found in %s' % in_str
    if upper_bounds is not None:
        assert [v <= b for v, b in zip(vL, high_bound)], 'invalid high value found in %s' % in_str
    return vL

        