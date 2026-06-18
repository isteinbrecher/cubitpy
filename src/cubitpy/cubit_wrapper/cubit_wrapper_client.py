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
# type: ignore
"""This script gets called with the cubit python interpreter and loads the
cubit module.

With the package execnet in the host python interpreter a connection is
established between the two different python interpreters and data and
commands can be exchanged. The exchange happens in a serial matter,
items are sent to this script, and results are sent back, until None is
sent. If cubit creates a cubit object it is saved in a dictionary in
this script, with the key being the id of the object. The host
interpreter only knows the id of this object and can pass it to this
script to call a function on it or use it as an argument.

Note: The contents of `cubit_wrapper_utility.py` will be added to the start
    of this file during the setup of the remote process.
"""

import os
import sys

# Cubit constants
cubit_vertex = "cubitpy_vertex"
cubit_curve = "cubitpy_curve"
cubit_surface = "cubitpy_surface"
cubit_volume = "cubitpy_volume"
cubit_body = "cubitpy_body"


# Default parameters
parameters = {}


def out(string):
    """The print version does over different interpreters, so this function
    prints strings to an active console.

    Insert the path of your console to get the
    right output.
    To get the current path of your console type: tty
    """

    if "tty" in parameters.keys():
        out_console = parameters["tty"]
    else:
        out_console = "/dev/pts/18"
    escaped_string = "{}".format(string).replace('"', '\\"')
    os.system('echo "{}" > {}'.format(escaped_string, out_console))  # nosec


def is_cubit_type(obj):
    """Check if the object is of a cubit base."""
    if (
        isinstance(obj, cubit.Body)
        or isinstance(obj, cubit.Vertex)
        or isinstance(obj, cubit.Curve)
        or isinstance(obj, cubit.Surface)
        or isinstance(obj, cubit.Volume)
        or isinstance(obj, cubit.MeshImport)
    ):
        return True
    else:
        return False


class DefaultMessageHandler:
    """This class is a dummy class that can be overwritten later on to
    intercept messages and errors from Cubit."""

    def pop(self):
        """Return two empty lists for messages and errors."""
        return [], []


message_handler = DefaultMessageHandler()


def channel_send(argument):
    """Wrapper for all send calls to the host interpreter.

    This wrapper appends information about messages and errors from
    Cubit.
    """
    messages, errors = message_handler.pop()
    channel.send(
        {
            "return_value": argument,
            "messages": messages,
            "errors": errors,
        }
    )


# All cubit items that are created are stored in this dictionary. The keys are
# the unique object ids. The items are deleted once they run out of scope in
# the host interpreter.
cubit_objects = {}


# The first call are parameters needed in this script
parameters = channel.receive()
channel_send(None)
if not isinstance(parameters, dict):
    raise TypeError(
        "The first item should be a dictionary. Got {}!\nparameters={}".format(
            type(parameters), parameters
        )
    )

# Add paths to cubit libs to sys, so cubit can be imported.
for path in parameters["additional_sys_paths"]:
    if path not in sys.path:
        sys.path.append(path)

import cubit

# The second call is the initialization call for cubit
# init = ['init', cubit_path, [args]]
init = channel.receive()
if not init[0] == "init":
    raise ValueError("The second call must be init!")
if not len(init) == 2:
    raise ValueError("Two arguments must be given to init!")
cubit.init(init[1])
cubit_objects[id(cubit)] = cubit
channel_send(object_to_id(cubit))


if parameters["is_remote"]:
    import platform
    import subprocess  # nosec B404
    import tempfile
    import time

    # On remote systems, create a temporary directory.
    temp_dir = tempfile.TemporaryDirectory(prefix="cubitpy_temp_dir")


# Try to add a custom message handler to Cubit
try:

    class MessageHandler(cubit.CubitMessageHandler):
        """This class intercepts messages and errors from Cubit."""

        def setup(self):
            """Initialize the variables that track the messages and errors."""
            self.messages = []
            self.errors = []

        def pop(self):
            """Return the stored messages and errors and reset the
            variables."""
            return_value = [self.messages, self.errors]
            self.setup()
            return return_value

        def print_message(self, message):
            """Append the message to the list of messages."""
            self.messages.append(message)

        def print_error(self, message):
            """Append the error to the list of errors."""
            self.errors.append(message)

    message_handler_cubit = MessageHandler()
    message_handler_cubit.setup()
    cubit.set_cubit_message_handler(message_handler_cubit)

    # Everything worked, so overwrite the message handler with the one linked to Cubit.
    message_handler = message_handler_cubit
except Exception:
    pass  # nosec B110


# Now start an endless loop (until None is sent) and perform the cubit functions
while 1:
    # Get input from the python host.
    receive = channel.receive()

    # If None is sent, break the connection and exit
    if receive is None:
        break

    # The first argument decides that functionality will be performed:
    # 'cubit_object': return an attribute of a cubit object. If the attribute is
    #       callable, it is executed with the given arguments.
    #       [[cubit_object], 'name', ['arguments']]
    # 'iscallable': Check if a name is callable or not
    # 'isinstance': Check if the cubit object is of a certain instance
    # 'get_self_dir': Return the attributes in a cubit_object
    # 'delete': Delete the cubit object from the dictionary
    # 'get_temp_dir': Get the temporary directory that is accessible by Cubit.
    # 'display_in_cubit': Launch cubit in the GUI.

    if cubit_item_to_id(receive[0]) is not None:
        # The first item is an id for a cubit object. Return an attribute of
        # this object.

        # Get object and attribute name
        call_object = cubit_objects[cubit_item_to_id(receive[0])]
        name = receive[1]

        def deserialize_item(item):
            """Deserialize the item, also if it contains nested nested
            lists."""
            item_id = cubit_item_to_id(item)
            if item_id is not None:
                return cubit_objects[item_id]
            elif isinstance(item, tuple) or isinstance(item, list):
                arguments = []
                for sub_item in item:
                    arguments.append(deserialize_item(sub_item))
                return arguments
            else:
                return item

        if callable(getattr(call_object, name)):
            # Call the function
            arguments = deserialize_item(receive[2])
            cubit_return = call_object.__getattribute__(name)(*arguments)
        else:
            # Get the attribute value
            cubit_return = call_object.__getattribute__(name)

        # Check what to return
        if is_base_type(cubit_return):
            # The return item is a string, integer or float
            channel_send(cubit_return)

        elif isinstance(cubit_return, tuple):
            # A tuple was returned, loop over each entry and check its type
            return_list = []
            for item in cubit_return:
                if is_base_type(item):
                    return_list.append(item)
                elif is_cubit_type(item):
                    cubit_objects[id(item)] = item
                    return_list.append(object_to_id(item))
                else:
                    raise TypeError(
                        "Expected string, int, float or cubit object! Got {}!".format(
                            item
                        )
                    )
            channel_send(return_list)

        elif is_cubit_type(cubit_return):
            # Store the object locally and return the id
            cubit_objects[id(cubit_return)] = cubit_return
            channel_send(object_to_id(cubit_return))

        else:
            raise TypeError(
                "Expected string, int, float, cubit object or tuple! Got {}!".format(
                    cubit_return
                )
            )

    elif receive[0] == "iscallable":
        cubit_object = cubit_objects[cubit_item_to_id(receive[1])]
        channel_send(callable(getattr(cubit_object, receive[2])))

    elif receive[0] == "get_object_type":
        # Get the type of the cubit object
        compare_object = cubit_objects[cubit_item_to_id(receive[1])]
        if isinstance(compare_object, cubit.Vertex):
            channel_send(cubit_vertex)
        elif isinstance(compare_object, cubit.Curve):
            channel_send(cubit_curve)
        elif isinstance(compare_object, cubit.Surface):
            channel_send(cubit_surface)
        elif isinstance(compare_object, cubit.Volume):
            channel_send(cubit_volume)
        elif isinstance(compare_object, cubit.Body):
            channel_send(cubit_body)
        else:
            channel_send(None)

    elif receive[0] == "get_self_dir":
        # Return a list with all callable methods of this object
        cubit_object = cubit_objects[cubit_item_to_id(receive[1])]
        channel_send(
            [
                [method_name, callable(getattr(cubit_object, method_name))]
                for method_name in dir(cubit_object)
            ]
        )

    elif receive[0] == "delete":
        # Get the id of the object to delete
        cubit_id = cubit_item_to_id(receive[1])
        if cubit_id is None:
            raise TypeError("Expected cubit object! Got {}!".format(item))

        # Delete the object from the dictionary.
        if cubit_id in cubit_objects.keys():
            del cubit_objects[cubit_id]
        else:
            raise ValueError(
                "The id {} is not in the cubit_objects dictionary".format(cubit_id)
            )

        # Return to python host
        channel_send(None)

    elif receive[0] == "get_temp_dir":
        channel_send(temp_dir.name)

    elif receive[0] == "display_in_cubit":
        # receive = ["display_in_cubit", parameters]
        parameters = receive[1]

        # Launch cubit in the GUI (for Windows remote systems). This is done by
        # creating a journal file which specifies the view options and then cubit is
        # run using this journal file.
        # The launch is done using the Windows task scheduler, which allows to run the
        # GUI app from a non-GUI process. The script waits until the task is finished,
        # i.e., cubit is closed.
        if not platform.system() == "Windows":
            raise NotImplementedError(
                "Launching the GUI is only implemented for Windows remote systems! "
                '"Got platform "{}".'.format(platform.system())
            )

        # Write file that opens the state in cubit.
        with open(parameters["journal_path"], "w") as journal:
            journal.write(parameters["journal_text"])

        # Get the command and arguments to open cubit with.
        cubit_command = parameters["cubit_command"]

        # Delete the task if it already exists
        task = "RunCubit"
        subprocess.run(  # nosec
            ["schtasks", "/Delete", "/TN", task, "/F"], check=False
        )

        # Create the task to run the GUI app
        subprocess.run(  # nosec
            [
                "schtasks",
                "/Create",
                "/TN",
                task,
                "/TR",
                cubit_command,
                "/SC",
                "ONCE",
                "/ST",
                "23:59",
                "/RL",
                "LIMITED",
                "/IT",
            ],
            check=True,
        )

        # Launch the task
        subprocess.run(  # nosec
            ["schtasks", "/Run", "/TN", task], check=True
        )

        # Wait for the task to complete by checking its status
        while True:
            r = subprocess.run(  # nosec
                [
                    "powershell",
                    "-Command",
                    "(Get-ScheduledTask -TaskName '{}').State".format(task),
                ],
                text=True,
                capture_output=True,
                check=True,
            )
            if r.stdout.strip() == "Ready":
                break

            time.sleep(2)
        channel_send(None)

    else:
        raise ValueError('The case of "{}" is not implemented!'.format(receive[0]))

if parameters["is_remote"]:
    temp_dir.cleanup()

# Send EOF
channel.send("EOF")
