# -*- coding: utf-8 -*-
"""
This script is used to test the functionality of the cubitpy module.
"""

# Python imports.
import unittest
import os
import sys
import subprocess
import numpy as np

# Define the testing paths.
testing_path = os.path.abspath(os.path.dirname(__file__))
testing_input = os.path.join(testing_path, 'input-files-ref')
testing_temp = os.path.join(testing_path, 'testing-tmp')

# Set path to find cubitpy.
sys.path.insert(0, os.path.abspath(os.path.join(testing_path, '..')))

# Cubitpy imports.
from cubitpy import CubitPy, cupy, get_surface_center
from cubitpy.mesh_creation_functions import create_brick


# Global variable if this test is run by GitLab.
if ('TESTING_GITLAB' in os.environ.keys()
        and os.environ['TESTING_GITLAB'] == '1'):
    TESTING_GITLAB = True
else:
    TESTING_GITLAB = False


def check_tmp_dir():
    """Check if the temp directory exists, if not create it."""
    os.makedirs(testing_temp, exist_ok=True)


def compare_strings(string_ref, string_compare):
    """
    Compare two stings. If they are not identical open kompare and show
    differences.
    """

    # Check if the strings are equal, if not fail the test and show the
    # differences in the strings.
    name = 'cubitpy_testing'
    compare = string_ref == string_compare
    if not compare:
        check_tmp_dir()
        strings = [string_ref, string_compare]
        files = []
        files.append(os.path.join(testing_temp, '{}_ref.dat'.format(name)))
        files.append(
            os.path.join(testing_temp, '{}_compare.dat'.format(name)))
        for i, file in enumerate(files):
            with open(file, 'w') as input_file:
                input_file.write(strings[i])
        if TESTING_GITLAB:
            subprocess.run(['diff', files[0], files[1]])
        else:
            child = subprocess.Popen(
                ['kompare', files[0], files[1]], stderr=subprocess.PIPE)
            child.communicate()
    return compare


class TestCubitPy(unittest.TestCase):
    """This class tests the implementation of the CubitPy class."""

    def compare(self, cubit, name, dat_lines_compare=False,
            single_precision=False):
        """
        Write create the dat file from the cubit mesh and compare to a
        reference file.

        Args
        ----
        cubit: Cubit object.
        name: str
            Name of the test case. A reference file 'name' + '_ref.dat' must
            exits in the reference file folder.
        dat_lines_compare: bool
            If the created file should be compared or the list of lines
            returned by get_dat_lines.
        single_precision: bool
            If the output of cubit is single or double precision.
        """

        # Create the dat file for the solid.
        check_tmp_dir()

        if single_precision:
            cubit.cmd('set exodus single precision on')

        # Get the string of the input file, depending on the chosen method.
        if not dat_lines_compare:
            dat_file = os.path.join(testing_temp, name + '.dat')
            cubit.create_dat(dat_file)
            with open(dat_file, 'r') as text_file:
                string2 = text_file.read()
        else:
            string2 = ''.join(cubit.get_dat_lines())

        # Compare with the ref file.
        ref_file = os.path.join(testing_input, name + '_ref.dat')
        with open(ref_file, 'r') as text_file:
            string1 = text_file.read()
        self.assertTrue(
            compare_strings(string1, string2), name)

    def create_block(self, cubit, dat_lines_compare=False, np_arrays=False):
        """
        Create a block with cubit.

        Args
        ----
        dat_lines_compare: bool
            If the created dat file should be compared or the list of lines
            returned by get_dat_lines.
        np_arrays: bool
            If the move operations sent to cubit are numpy or python arrays.
        """

        # Set head
        cubit.head = '''
            // Header processed by cubit.
            '''

        # Dimensions of the block.
        lx, ly, lz = [0.1, 1, 10]

        # Number of elements in the directions.
        [nx, ny, nz] = [2, 4, 8]

        # Create the block.
        block = cubit.brick(lx, ly, lz)

        # Move the block.
        move_array = [0, 0, block.bounding_box()[2]]
        if np_arrays:
            move_array = np.array(move_array)
        cubit.move(block, move_array)

        # Set the meshing parameters for the curves.
        for line in block.curves():
            point_on_line = line.position_from_fraction(0.5)
            tangent = np.array(line.tangent(point_on_line))
            if np.abs(np.dot(tangent, [1, 0, 0])) > 1e-5:
                cubit.set_line_interval(line, nx)
            elif np.abs(np.dot(tangent, [0, 1, 0])) > 1e-5:
                cubit.set_line_interval(line, ny)
            elif np.abs(np.dot(tangent, [0, 0, 1])) > 1e-5:
                cubit.set_line_interval(line, nz)
            else:
                raise ArithmeticError('Error')

        # Mesh the block.
        block.mesh()
        cubit.add_element_type(block.volumes()[0], cupy.element_type.hex8,
            name='block', material='MAT 1',
            bc_description='KINEM nonlinear EAS none')

        # Create node sets.
        for i, surf in enumerate(block.surfaces()):
            normal = np.array(surf.normal_at(get_surface_center(surf)))
            if np.dot(normal, [0, 0, -1]) == 1:
                cubit.add_node_set(surf, name='fix',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 6 ONOFF 1 1 1 0 0 0 '
                        + 'VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0 0 0 0')
            elif np.dot(normal, [0, 0, 1]) == 1:
                cubit.add_node_set(surf, name='load',
                    bc_section='DESIGN SURF DIRICH CONDITIONS',
                    bc_description='NUMDOF 6 ONOFF 1 1 1 0 0 0 '
                        + 'VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0 0 0 0')
            else:
                cubit.add_node_set(surf, name='load{}'.format(i),
                    bc_section='DESIGN SURF NEUMANN CONDITIONS',
                    bc_description='NUMDOF 6 ONOFF 1 1 1 0 0 0 '
                        + 'VAL 0.0 0.0 0.0 0.0 0.0 0.0 '
                        + 'FUNCT 0 0 0 0 0 0')

        # Compare the input file created for baci.
        self.compare(cubit, 'test_create_block',
            dat_lines_compare=dat_lines_compare)

    def test_create_block(self):
        """
        Test the creation of a cubit block.
        """

        # Initialize cubit.
        cubit = CubitPy()
        self.create_block(cubit)

    def test_create_block_numpy_arrays(self):
        """
        Test the creation of a cubit block.
        """

        # Initialize cubit.
        cubit = CubitPy()
        self.create_block(cubit, np_arrays=True)

    def test_create_block_multiple(self):
        """
        Test the creation of a cubit block multiple time to check that cubit
        can be reset.
        """

        # Initialize cubit.
        cubit = CubitPy()
        self.create_block(cubit)

        # Delete the old cubit object and run the function twice on the new.
        cubit = CubitPy()
        for _i in range(2):
            self.create_block(cubit)
            cubit.reset()

        # Create two object and keep them in parallel.
        cubit = CubitPy()
        cubit_2 = CubitPy()
        self.create_block(cubit)
        self.create_block(cubit_2)

    def test_create_block_dat_lines(self):
        """
        Test the creation of a cubit block, with the get_dat_lines method.
        """

        # Initialize cubit.
        cubit = CubitPy()
        self.create_block(cubit, dat_lines_compare=True)

    def test_element_types(self):
        """Create a curved solid with different element types."""

        # Initialize cubit.
        cubit = CubitPy()

        def add_arc(radius, angle):
            """Add a arc segment."""
            cubit.cmd(('create curve arc radius {} center location 0 0 0 '
                + 'normal 0 0 1 start angle 0 stop angle {}').format(
                    radius, angle))

        element_type_list = [
            cupy.element_type.hex8,
            cupy.element_type.hex20,
            cupy.element_type.hex27,
            cupy.element_type.tet4,
            cupy.element_type.tet10,
            cupy.element_type.hex8sh
            ]

        for i, element_type in enumerate(element_type_list):

            # Offset for the next volume.
            offset_point = i * 12
            offset_curve = i * 12
            offset_surface = i * 6
            offset_volume = i

            # Add two arcs.
            add_arc(1.1, 60)
            add_arc(0.9, 60)

            # Add the closing lines.
            cubit.cmd('create curve vertex {} {}'.format(2 + offset_point,
                4 + offset_point))
            cubit.cmd('create curve vertex {} {}'.format(1 + offset_point,
                3 + offset_point))

            # Create the surface.
            cubit.cmd('create surface curve {} {} {} {}'.format(
                1 + offset_curve,
                2 + offset_curve,
                3 + offset_curve,
                4 + offset_curve))

            # Create the volume.
            cubit.cmd('sweep surface {} perpendicular distance 0.2'.format(
                1 + offset_surface
                ))

            # Move the volume.
            cubit.cmd('move Volume {} x 0 y 0 z {}'.format(1 + offset_volume,
                i * 0.4))

            # Set the element type.
            cubit.add_element_type(
                cubit.volume(1 + offset_volume),
                element_type,
                name='block_' + str(i),
                material='MAT 1',
                bc_description=None)

            # Set mesh properties.
            cubit.cmd('volume {} size auto factor 7'.format(1 + offset_volume))
            cubit.cmd('mesh volume {}'.format(1 + offset_volume))

            # Add the node sets.
            cubit.add_node_set(
                cubit.surface(5 + offset_surface),
                name='fix_' + str(i),
                bc_section='DESIGN SURF DIRICH CONDITIONS',
                bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0')

        # Set the head string.
        cubit.head = '''
            -------------------------------------------------------------FUNCT1
            COMPONENT 0 FUNCTION t
            ----------------------------------------------------------MATERIALS
            MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 1.0e+09 NUE 0.3 DENS 0.0
            ------------------------------------IO/RUNTIME VTK OUTPUT/STRUCTURE
            OUTPUT_STRUCTURE                Yes
            DISPLACEMENT                    Yes
            '''

        # Compare the input file created for baci.
        self.compare(cubit, 'test_element_types', single_precision=True)

    def test_block_function(self):
        """Create a solid block with different element types."""

        # Initialize cubit.
        cubit = CubitPy()

        element_type_list = [
            cupy.element_type.hex8,
            cupy.element_type.hex20,
            cupy.element_type.hex27,
            cupy.element_type.tet4,
            cupy.element_type.tet10,
            cupy.element_type.hex8sh
            ]

        for i, element_type in enumerate(element_type_list):
            # Create the cube with factor as mesh size.
            cube = create_brick(cubit, 0.5, 0.8, 1.1,
                element_type=element_type, mesh_factor=9,
                name=str(element_type) + str(i), mesh=False,
                material='test material string')
            cubit.move(cube, [i, 0, 0])
            cube.volumes()[0].mesh()

        for i, element_type in enumerate(element_type_list):
            # Create the cube with intervals as mesh size.
            cube = create_brick(cubit, 0.5, 0.8, 1.1,
                element_type=element_type, mesh_interval=[3, 2, 1],
                name=str(element_type) + str(i + 5), mesh=False,
                material='test material string')
            cubit.move(cube, [i + 5, 0, 0])
            cube.volumes()[0].mesh()

        # Compare the input file created for baci.
        self.compare(cubit, 'test_block_function', single_precision=True)

    def test_node_set_geometry_type(self):
        """Create the boundary conditions via the bc_type enum."""

        # First create the solid mesh.
        cubit = CubitPy()
        solid = create_brick(cubit, 1, 1, 1, mesh_interval=[1, 1, 1])

        # Add all possible boundary conditions.

        # Dirichlet and Neumann.
        cubit.add_node_set(
            solid.vertices()[0],
            name='vertex',
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 1')
        cubit.add_node_set(
            solid.curves()[0],
            name='curve',
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 2')
        cubit.add_node_set(
            solid.surfaces()[0],
            name='surface',
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 3')
        cubit.add_node_set(
            solid.volumes()[0],
            name='volume',
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 4')

        # Define boundary conditions on explicit nodes.
        cubit.add_node_set(
            cubit.group(add_value='add node 2'),
            name='point2',
            geometry_type=cupy.geometry.vertex,
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 4')
        cubit.add_node_set(
            cubit.group(add_value='add node {}'.format(' '.join(
                [str(i + 1) for i in range(cubit.get_node_count())]))),
            name='point3',
            geometry_type=cupy.geometry.vertex,
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 4')

        # Coupling.
        cubit.add_node_set(
            solid.volumes()[0],
            name='coupling_btsv',
            bc_type=cupy.bc_type.beam_to_solid_volume_meshtying,
            bc_description='COUPLING_ID 1'
            )
        cubit.add_node_set(
            solid.surfaces()[0],
            name='coupling_btss',
            bc_type=cupy.bc_type.beam_to_solid_surface_meshtying,
            bc_description='COUPLING_ID 1'
            )

        # Set the head string.
        cubit.head = '''
            ----------------------------------------------------------MATERIALS
            MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 10 NUE 0.0 DENS 0.0'''

        # Compare the input file created for baci.
        self.compare(cubit, 'test_node_set_geometry_type')

    def test_coupling(self):
        """Create node-node and vertex-vertex coupling."""

        # First create two blocks.
        cubit = CubitPy()
        solid_1 = create_brick(cubit, 1, 1, 1, mesh_interval=[2, 2, 2],
            mesh=False)
        cubit.move(solid_1, [0.0, -0.5, 0.0])
        solid_2 = create_brick(cubit, 1, 2, 1, mesh_interval=[2, 4, 2],
            mesh=False)
        cubit.move(solid_2, [0.0, 1.0, 0.0])

        # Mesh the blocks.
        solid_1.mesh()
        solid_2.mesh()

        # Couple all nodes on the two surfaces. Therefore we first have to get
        # the surfaces of the two blocks that are at the interface.
        surfaces = cubit.group(name='interface_surfaces')
        surfaces.add('add surface with -0.1 < y_coord and y_coord < 0.1')

        # Check each node with each other node. If they are at the same
        # position, add a coupling.
        surf = surfaces.get_geometry_objects(cupy.geometry.surface)
        for node_id_1 in surf[0].get_node_ids():
            coordinates_1 = np.array(cubit.get_nodal_coordinates(node_id_1))
            for node_id_2 in surf[1].get_node_ids():
                coordinates_2 = cubit.get_nodal_coordinates(node_id_2)
                if (np.linalg.norm(coordinates_2 - coordinates_1)
                        < cupy.eps_pos):
                    cubit.add_node_set(
                        cubit.group(add_value='add node {} {}'.format(
                            node_id_1, node_id_2)),
                        geometry_type=cupy.geometry.vertex,
                        bc_type=cupy.bc_type.point_coupling,
                        bc_description='NUMDOF 3 ONOFF 1 1 1'
                        )

        # Also add coupling explicitly to the on corners.
        for point_1 in solid_1.vertices():
            coordinates_1 = np.array(point_1.coordinates())
            for point_2 in solid_2.vertices():
                coordinates_2 = np.array(point_2.coordinates())
                if (np.linalg.norm(coordinates_2 - coordinates_1)
                        < cupy.eps_pos):

                    # Here a group has to be created.
                    group = cubit.group()
                    group.add([point_1, point_2])
                    cubit.add_node_set(
                        group,
                        bc_type=cupy.bc_type.point_coupling,
                        bc_description='NUMDOF 3 ONOFF 1 2 3'
                        )

        # Compare the input file created for baci.
        self.compare(cubit, 'test_point_coupling')

    def test_groups_block_with_volume(self):
        """
        Test the group functions where the block is created by adding the
        volume.
        """
        self.xtest_groups(True)

    def test_groups_block_with_hex(self):
        """
        Test the group functions where the block is created by adding the hex
        elements directly.
        """
        self.xtest_groups(False)

    def xtest_groups(self, block_with_volume):
        """
        Test that groups are handled correctly when creating node sets and
        element blocks.

        Args
        ----
        block_with_volume: bool
            If the element block should be added via a group containing the
            geometry volume or via a group containing the hex elements.
        """

        # Create a solid brick.
        cubit = CubitPy()
        cubit.brick(4, 2, 1)

        # Add to group by string.
        volume = cubit.group(name='all_vol')
        volume.add('add volume all')

        # Add to group via string.
        surface_fix = cubit.group(name='fix_surf',
            add_value='add surface in volume in all_vol with x_coord < 0')
        surface_load = cubit.group(name='load_surf',
            add_value='add surface in volume in all_vol with x_coord > -1.99')

        # Add to group by CubitPy object.
        surface_load_alt = cubit.group(name='load_surf_alt')
        surface_load_alt.add(cubit.surface(1))
        surface_load_alt.add([cubit.surface(i) for i in [2, 3, 5, 6]])

        # Create a group without a name.
        group_no_name = cubit.group()
        group_no_name.add('add surface in volume in all_vol with x_coord < 0')

        # Create a group without a name.
        group_explicit_type = cubit.group()
        group_explicit_type.add('add surface 2')
        group_explicit_type.add('add curve 1')
        group_explicit_type.add('add vertex 3')

        if block_with_volume:
            # Set element type.
            cubit.add_element_type(volume, cupy.element_type.hex8,
                material='MAT 1', bc_description='KINEM nonlinear EAS none')

        # Add BCs.
        cubit.add_node_set(surface_fix,
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0')
        cubit.add_node_set(surface_load,
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 0 0 1 VAL 0 0 1 FUNCT 0 0 0')
        cubit.add_node_set(surface_load_alt,
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 0 0 1 VAL 0 0 1 FUNCT 0 0 0')
        cubit.add_node_set(group_no_name,
            name='fix_surf_no_name_group',
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0')
        cubit.add_node_set(group_explicit_type,
            name='fix_group_explicit_type',
            geometry_type=cupy.geometry.vertex,
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0')

        # Mesh the model.
        cubit.cmd('volume {} size auto factor 8'.format(volume.id()))
        cubit.cmd('mesh {}'.format(volume))

        if not block_with_volume:
            all_hex = cubit.group(add_value='add hex all')
            cubit.add_element_type(all_hex, cupy.element_type.hex8,
                material='MAT 1', bc_description='KINEM nonlinear EAS none')

        # Add a group containing elements and nodes.
        mesh_group = cubit.group(name='mesh_group')
        mesh_group.add('add node 1 4 18 58 63')
        mesh_group.add('add face 69')
        mesh_group.add('add hex 17')
        cubit.add_node_set(mesh_group,
            geometry_type=cupy.geometry.vertex,
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0')

        # Set the head string.
        cubit.head = '''
            ----------------------------------------------------------MATERIALS
            MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 10 NUE 0.0 DENS 0.0'''

        # Compare the input file created for baci.
        self.compare(cubit, 'test_groups')

    def test_groups_multiple_sets(self):
        """
        Test that multiple sets can be created from a single group object.
        """

        # Create a solid brick.
        cubit = CubitPy()
        cubit.brick(4, 2, 1)

        # Add to group by string.
        volume = cubit.group(name='all_vol')
        volume.add('add volume all')

        # Add BCs.
        cubit.add_node_set(volume,
            bc_type=cupy.bc_type.dirichlet,
            bc_description='NUMDOF 3 ONOFF 1 1 1 VAL 0 0 0 FUNCT 0 0 0')
        cubit.add_node_set(volume,
            bc_type=cupy.bc_type.neumann,
            bc_description='NUMDOF 3 ONOFF 0 0 1 VAL 0 0 1 FUNCT 0 0 0')

        # Add blocks.
        cubit.add_element_type(volume, cupy.element_type.hex8)

        # Mesh the model.
        cubit.cmd('volume {} size auto factor 8'.format(volume.id()))
        cubit.cmd('mesh {}'.format(volume))

        # Set the head string.
        cubit.head = '''
            ----------------------------------------------------------MATERIALS
            MAT 1 MAT_Struct_StVenantKirchhoff YOUNG 10 NUE 0.0 DENS 0.0'''

        # Compare the input file created for baci.
        self.compare(cubit, 'test_groups_multiple_sets')

    def test_reset_block(self):
        """
        Test that the block counter can be reset in cubit.
        """

        # Create a solid brick.
        cubit = CubitPy()
        block_1 = cubit.brick(1, 1, 1)
        block_2 = cubit.brick(2, 0.5, 0.5)
        cubit.cmd('volume 1 size auto factor 10')
        cubit.cmd('volume 2 size auto factor 10')
        cubit.cmd('mesh volume 1')
        cubit.cmd('mesh volume 2')

        cubit.add_element_type(block_1.volumes()[0], cupy.element_type.hex8)
        self.compare(cubit, 'test_reset_block_1')

        cubit.reset_blocks()
        cubit.add_element_type(block_2.volumes()[0], cupy.element_type.hex8)
        self.compare(cubit, 'test_reset_block_2')

    def test_get_id_functions(self):
        """
        Test if the get_ids and get_items methods work as expected.
        """

        cubit = CubitPy()

        cubit.cmd('create vertex 0 0 0')
        cubit.cmd('create curve location 0 0 0 location 1 1 1')
        cubit.cmd('create surface circle radius 1 zplane')
        cubit.cmd('brick x 1')

        self.assertEqual(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            cubit.get_ids(cupy.geometry.vertex))
        self.assertEqual(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            cubit.get_ids(cupy.geometry.curve))
        self.assertEqual(
            [1, 2, 3, 4, 5, 6, 7],
            cubit.get_ids(cupy.geometry.surface))
        self.assertEqual(
            [2], cubit.get_ids(cupy.geometry.volume))

    def test_get_node_id_function(self):
        """
        Test if the get_node_ids methods in the cubit objects work as expected.
        """

        # Create brick.
        cubit = CubitPy()
        brick = create_brick(cubit, 1, 1, 1, mesh_interval=[2, 2, 2])

        # Compare volume, surface, curve and vertex nodes.
        node_ids = brick.volumes()[0].get_node_ids()
        node_ids.sort()
        self.assertEqual(node_ids, list(range(1, 28)))

        node_ids = brick.surfaces()[3].get_node_ids()
        node_ids.sort()
        self.assertEqual(node_ids, [4, 6, 7, 13, 15, 16, 19, 22, 23])

        node_ids = brick.curves()[4].get_node_ids()
        node_ids.sort()
        self.assertEqual(node_ids, [10, 11, 12])

        node_ids = brick.vertices()[7].get_node_ids()
        node_ids.sort()
        self.assertEqual(node_ids, [15])

    def test_nested_lists(self):
        """
        Test that nested lists can be send to cubit correctly.
        """

        cubit = CubitPy()
        block_1 = cubit.brick(1, 1, 0.25)
        block_2 = cubit.brick(0.5, 0.5, 0.5)
        subtracted_block = cubit.subtract([block_2], [block_1])
        cubit.cmd('volume {} size auto factor 10'.format(
            subtracted_block[0].volumes()[0].id()))
        subtracted_block[0].volumes()[0].mesh()
        cubit.add_element_type(subtracted_block[0].volumes()[0],
            cupy.element_type.hex8)
        self.compare(cubit, 'test_nested_lists',
            dat_lines_compare=False)

    def test_display_in_cubit(self):
        """
        Call the display_in_cubit function without actually opening the graphic
        version of cubit. Compare that the created journal file is correct.
        """

        # Create brick.
        cubit = CubitPy()
        create_brick(cubit, 1, 1, 1, mesh_interval=[2, 2, 2])

        # Check the journal file which is created in the display_in_cubit
        # function.
        journal_path = cubit.display_in_cubit(labels=[
            cupy.geometry.vertex,
            cupy.geometry.curve,
            cupy.geometry.surface,
            cupy.geometry.volume,
            cupy.finite_element_object.node,
            cupy.finite_element_object.edge,
            cupy.finite_element_object.face,
            cupy.finite_element_object.triangle,
            cupy.finite_element_object.hex,
            cupy.finite_element_object.tet
            ], testing=True)
        with open(journal_path, 'r') as journal:
            journal_text = journal.read()
        ref_text = ('open "/tmp/cubitpy/state.cub"\n'
            'label volume On\n'
            'label surface On\n'
            'label curve On\n'
            'label vertex On\n'
            'label hex On\n'
            'label tet On\n'
            'label face On\n'
            'label tri On\n'
            'label edge On\n'
            'label node On\n'
            'display')
        self.assertTrue(journal_text.strip() == ref_text.strip())


if __name__ == '__main__':
    unittest.main()
