# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# CubitPy: Cubit utility functions and a cubit wrapper for python3
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
This module defines a global object that manages all kind of stuff regarding
cubitpy.
"""

# Python imports.
import os
import getpass

# Cubitpy imports.
from .cubitpy_types import (
    FiniteElementObject,
    GeometryType,
    ElementType,
    CubitItems,
    BoundaryConditionType,
)


class CubitOptions(object):
    """Object for types in cubitpy."""

    def __init__(self):
        # Temporary directory for cubitpy.
        self.temp_dir = os.path.join(
            "/tmp/cubitpy_{}".format(getpass.getuser()), "pid_{}".format(os.getpid())
        )
        self.temp_log = os.path.join(self.temp_dir, "cubitpy.log")

        # Check if temp path exits, if not create it.
        os.makedirs(self.temp_dir, exist_ok=True)

        # Geometry types.
        self.geometry = GeometryType

        # Finite element types.
        self.finite_element_object = FiniteElementObject

        # Element shape types.
        self.element_type = ElementType

        # Cubit internal items.
        self.cubit_items = CubitItems

        # Boundary condition type.
        self.bc_type = BoundaryConditionType

        # Tolerance for geometry.
        self.eps_pos = 1e-10

    @staticmethod
    def get_default_paths(name, throw_error=True):
        """Look for and return a path to cubit or pre_exodus."""

        if name == "cubit":
            environment_variable = "CUBIT2023"
            test_function = os.path.isdir
        elif name == "pre_exodus":
            environment_variable = "BACI_PRE_EXODUS"
            test_function = os.path.isfile
        else:
            raise ValueError("Type {} not implemented!".format(name))

        # Check if he environment variable is set and the path exits.
        if environment_variable in os.environ.keys():
            if test_function(os.environ[environment_variable]):
                return os.environ[environment_variable]

        # No valid path found or given.
        if throw_error:
            raise ValueError("Path for {} not found!".format(name))
        else:
            return None


# Global object with options for cubitpy.
cupy = CubitOptions()
