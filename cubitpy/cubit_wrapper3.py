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
This module creates object that are used to connect between cubit in python2
and python3.
"""


# Import python modules.
import execnet
import os
import numpy as np

# Import global options.
from .conf import cupy

# Import utility functions for cubitpy.
from .cubit_wrapper_utility import cubit_item_to_id, is_base_type


class CubitConnect(object):
    """
    This class holds a connection to a python2 interpreter and initializes
    cubit there. It is possible to send function calls to that interpreter and
    receive the output.
    """

    def __init__(
        self,
        cubit_arguments,
        #interpreter="popen//python=/home/ivo/opt/cubit-2023/bin/python3/python3",
        interpreter="popen//python=python3",
        cubit_bin_path=None,
    ):
        """
        Initialize the connection between python2 and python3. And load the
        cubit module in python2.

        Args
        ----
        cubit_arguments: [str]
            Arguments to initialize cubit with.
        interpreter: str
            Interpreter for python2 that will be used.
        cubit_path: str
            Path to the cubit executable.
        """

        if cubit_bin_path is None:
            raise ValueError("Path to cubit was not given!")

        # Set up the python2 interpreter.
        self.gw = execnet.makegateway(interpreter)
        self.gw.reconfigure(py3str_as_py2str=True)

        # Load the python2 code.
        python2_file = os.path.join(os.path.dirname(__file__), "cubit_wrapper2.py")
        with open(python2_file, "r") as myfile:
            data = myfile.read()

        # Set up the connection channel.
        self.channel = self.gw.remote_exec(data)

        # Send parameters to the python2 interpreter
        parameters = {}
        parameters["__file__"] = __file__
        parameters["cubit_bin_path"] = cubit_bin_path

        # Check if a log file was given in the cubit arguments.
        for arg in cubit_arguments:
            if arg.startswith("-log="):
                log_given = True
                break
        else:
            log_given = False

        self.log_check = False

        if not log_given:
            # Write the log to a temporary file and
            # check the contents after each call to cubit.
            cubit_arguments.extend(["-log", cupy.temp_log])
            parameters["tty"] = cupy.temp_log
            self.log_check = True

        # Send the parameters to python2
        self.send_and_return(parameters)

        # Initialize cubit.
        cubit_id = self.send_and_return(["init", cubit_arguments])
        self.cubit = CubitObjectMain(self, cubit_id)

    def send_and_return(self, argument_list, check_number_of_channels=False):
        """
        Send arguments to python2 and collect the return values.

        Args
        ----
        argument_list: list
            First item is either a string with the action, or a cubit item id.
            In the second case a method will be called on the item, with the
            arguments stored in the second entry in argument_list.
        check_number_of_channels: bool
            If true it is checked if the channel still exists. This is
            necessary in cases where we delete items after the connection has
            been closed.
        """

        if check_number_of_channels:
            if len(self.gw._channelfactory.channels()) == 0:
                return None

        self.channel.send(argument_list)
        return self.channel.receive()

    def get_attribute(self, cubit_object, name):
        """
        Return the attribute 'name' of cubit_object. If the attribute is
        callable a function is returned, otherwise the attribute value is
        returned.

        Args
        ----
        cubit_object: CubitObject
            The object on which the method is called.
        name: str
            Name of the method.
        """

        def function(*args):
            """This function gets returned from the parent method."""

            def serialize_item(item):
                """
                Serialize an item, also nested lists.
                """

                if (
                    isinstance(item, tuple)
                    or isinstance(item, list)
                    or isinstance(item, np.ndarray)
                ):
                    arguments = []
                    for sub_item in item:
                        arguments.append(serialize_item(sub_item))
                    return arguments
                elif isinstance(item, CubitObject):
                    return item.cubit_id
                elif isinstance(item, float):
                    return float(item)
                elif isinstance(item, int):
                    return int(item)
                elif isinstance(item, cupy.geometry):
                    return item.get_cubit_string()
                else:
                    return item

            if self.log_check:
                # Check if the log file is empty. If it is not, empty it.
                if os.stat(cupy.temp_log).st_size != 0:
                    with open(cupy.temp_log, "w"):
                        pass

            # Check if there are cubit objects in the arguments.
            arguments = serialize_item(args)

            # Call the method on the cubit object.
            cubit_return = self.send_and_return(
                [cubit_object.cubit_id, name, arguments]
            )

            if self.log_check:
                # Print the content of the log file.
                with open(cupy.temp_log, "r") as log_file:
                    print(log_file.read(), end="")

            # Check if the return value is a cubit object.
            if cubit_item_to_id(cubit_return) is not None:
                return CubitObject(self, cubit_return)
            elif isinstance(cubit_return, list):
                # If the return value is a list, check if any entry of the list
                # is a cubit object.
                return_list = []
                for item in cubit_return:
                    if cubit_item_to_id(item) is not None:
                        return_list.append(CubitObject(self, item))
                    elif is_base_type(item):
                        return_list.append(item)
                    else:
                        raise TypeError(
                            "Expected cubit object, or base_type, "
                            + "got {}!".format(item)
                        )
                return return_list
            elif is_base_type(cubit_return):
                return cubit_return
            else:
                raise TypeError(
                    "Expected cubit object, or base_type, "
                    + "got {}!".format(cubit_return)
                )

            return cubit_return

        # Depending on the type of attribute, return the attribute value or a
        # callable function.
        if self.send_and_return(["iscallable", cubit_object.cubit_id, name]):
            return function
        else:
            return function()


class CubitObject(object):
    """
    This class holds a link to a cubit object in python2. Methods that are
    called on this class will 'really' be called in python2.
    """

    def __init__(self, cubit_connect, cubit_data_list):
        """
        Initialize the object.

        Args
        ----
        cubit_connect: CubitConnect
            A link to the cubit_connec object that will be used to call
            methods.
        cubit_data_list: []
            A list of strings that contains info about the cubit object.
            The first item is the id of this object in python2.
        """

        # Check formating of cubit_id.
        if cubit_item_to_id(cubit_data_list) is None:
            raise TypeError("Wrong type {}".format(cubit_data_list))

        self.cubit_connect = cubit_connect
        self.cubit_id = cubit_data_list

    def __getattribute__(self, name, *args, **kwargs):
        """
        This function gets called for each attribute in this object.
        First it is checked if the attribute exists in python3 (basic stuff),
        if not the attribute is called on python2.

        For now if an attribute is sent to python2, it is assumed that it is a
        method.
        """

        # Check if the attribute exists in python3.
        try:
            return object.__getattribute__(self, name, *args, **kwargs)
        except AttributeError:
            return self.cubit_connect.get_attribute(self, name)

    def __del__(self):
        """
        When this object is deleted, the object in the wraper can also be
        deleted.
        """
        self.cubit_connect.send_and_return(
            ["delete", self.cubit_id], check_number_of_channels=True
        )

    def __str__(self):
        """Return the string from python2."""
        return '<CubitObject>"' + self.cubit_id[1] + '"'

    def isinstance(self, geom_type):
        """
        Check if this object is of geom_type.

        Args
        ----
        geom_type: str
            Name of the geometry to compare (vertex, curve, surface, volume).
        """

        # Compare in python2.
        return self.cubit_connect.send_and_return(
            ["isinstance", self.cubit_id, geom_type]
        )

    def get_self_dir(self):
        """
        Return a list of all cubit child items of this object. Also return a
        flag if the child item is callable or not.
        """
        return self.cubit_connect.send_and_return(["get_self_dir", self.cubit_id])

    def get_methods(self):
        """Return a list of all callable cubit methods for this object."""
        return [method for method, callable in self.get_self_dir() if callable]

    def get_attributes(self):
        """Return a list of all non callable cubit methods for this object."""
        return [method for method, callable in self.get_self_dir() if not callable]

    def get_geometry_type(self):
        """Return the type of this item."""

        if self.isinstance("cubitpy_vertex"):
            return cupy.geometry.vertex
        elif self.isinstance("cubitpy_curve"):
            return cupy.geometry.curve
        elif self.isinstance("cubitpy_surface"):
            return cupy.geometry.surface
        elif self.isinstance("cubitpy_volume"):
            return cupy.geometry.volume

        # Default value -> not a valid geometry.
        raise TypeError("The item is not a valid geometry!")

    def get_node_ids(self):
        """
        Return a list with the node IDs (index 1) of this object.

        This is done my creating a temporary node set that this geometry is
        added to. It is not possible to get the node list directly from cubit.
        """

        # Get a node set ID that is not yet taken.
        node_set_ids = [0]
        node_set_ids.extend(self.cubit_connect.cubit.get_nodeset_id_list())
        temp_node_set_id = max(node_set_ids) + 1

        # Add a temporary node set with this geometry.
        self.cubit_connect.cubit.cmd(
            "nodeset {} {} {}".format(
                temp_node_set_id, self.get_geometry_type().get_cubit_string(), self.id()
            )
        )

        # Get the nodes in the created node set.
        node_ids = self.cubit_connect.cubit.get_nodeset_nodes_inclusive(
            temp_node_set_id
        )

        # Delete the temp node set and return the node list.
        self.cubit_connect.cubit.cmd("delete nodeset {}".format(temp_node_set_id))
        return node_ids


class CubitObjectMain(CubitObject):
    """
    The main cubit object will be of this type, it can not delete itself.
    """

    def __del__(self):
        pass
