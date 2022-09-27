# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# CubitPy: Cubit utility funcitons and a cubit wrapper for python3
#
# MIT License
#
# Copyright (c) 2021 Ivo Steinbrecher
#                    Institute for Mathematics and Computer-Based Simulation
#                    Universitaet der Bundeswehr Muenchen
#                    https://www.unibw.de/imcs-en
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# -----------------------------------------------------------------------------
"""
Utility functions for the python2 to 3 wrapper.
"""


def object_to_id(obj):
    """
    Return list representing the cubit object. The first entry is the python id
    of the object, the second entry is the string representation.
    """
    return ["cp2t3id_" + str(id(obj)), str(obj)]


def cubit_item_to_id(cubit_data_list):
    """Return the id from a cubit data list."""
    if not isinstance(cubit_data_list, list):
        return None
    if len(cubit_data_list) == 0:
        return None
    if not isinstance(cubit_data_list[0], str):
        return None
    if cubit_data_list[0].startswith("cp2t3id_"):
        return int(cubit_data_list[0][8:])
    else:
        return None


def is_base_type(obj):
    """
    Check if the object is of a base type that does not need conversion for the
    connection between python2 and python3.
    """
    if (
        isinstance(obj, str)
        or isinstance(obj, int)
        or isinstance(obj, float)
        or isinstance(obj, type(None))
    ):
        return True
    else:
        return False
