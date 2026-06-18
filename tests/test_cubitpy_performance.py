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
"""This script is used to test the performance of CubitPy."""

import warnings
from typing import Callable

import numpy as np
import pytest

from cubitpy.conf import cupy
from cubitpy.cubitpy import CubitPy


@pytest.fixture()
def benchmark_cubitpy(benchmark, request) -> Callable:
    """Return a function that can be used to benchmark CubitPy functions."""

    def _benchmark_cubitpy(
        function: Callable,
        *,
        reference_times: list[float],
        rounds: int,
        iterations: int,
        **kwargs,
    ):
        """Benchmark a CubitPy function.

        Args:
            function: The function to benchmark.
            reference_times: A list of two reference times, one for local and one for remote execution.
            rounds: The number of rounds to run the benchmark.
            iterations: The number of iterations to run the benchmark in each round.
            **kwargs: Additional keyword arguments to pass to the benchmark function.

        Returns:
            The result of the benchmark function.
        """
        reference_time = (
            reference_times[0] if not cupy.is_remote() else reference_times[1]
        )
        return_value = benchmark.pedantic(
            function, rounds=rounds, iterations=iterations, **kwargs
        )
        mean_run_time = benchmark.stats.stats.mean
        if mean_run_time > reference_time:
            warnings.warn(
                f"{request.node.name}: mean runtime {mean_run_time * 1e6} us exceeds {reference_time * 1e6} us"
            )
        return return_value

    return _benchmark_cubitpy


def test_cubitpy_performance_object_creation(benchmark_cubitpy):
    """Check the performance of object creation."""

    cubit = CubitPy()
    cubit.cmd("brick x 1 y 1 z 1")

    created_objects = []

    benchmark_cubitpy(
        lambda: created_objects.append(cubit.body(1)),
        reference_times=[0.0004, 0.0006],
        rounds=10,
        iterations=500,
    )


def test_cubitpy_performance_operations(benchmark_cubitpy):
    """Create a block and move it around numerous times to check the
    performance of the execnet connection."""

    cubit = CubitPy()
    cubit.cmd("brick x 1 y 1 z 1")
    body = cubit.body(1)
    vector = np.array([0.01, 0.02, 0.03])

    benchmark_cubitpy(
        lambda: cubit.move(body, vector),
        reference_times=[0.00065, 0.00082],
        rounds=10,
        iterations=500,
    )


def test_cubitpy_performance_receive_large_data(benchmark_cubitpy):
    """Check the performance of receiving large data from CubitPy."""

    cubit = CubitPy()

    benchmark_cubitpy(
        cubit.cubit.get_self_dir,
        reference_times=[0.0065, 0.0082],
        rounds=10,
        iterations=100,
    )


def test_cubitpy_performance_send_large_data(benchmark_cubitpy):
    """Check the performance of sending large data to CubitPy."""

    cubit = CubitPy()
    large_string = "a" * 1_000

    benchmark_cubitpy(
        lambda: cubit.cmd(f'echo "{large_string}"'),
        reference_times=[0.0085, 0.0085],
        rounds=10,
        iterations=100,
    )
