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
This module contains ENums for types used in cubitpy as well as functions to
convert them to strings for cubit or baci commands or the python2 wrapper.
"""

# Python imports.
from enum import Enum, auto


class GeometryType(Enum):
    """Enum for geometry types."""

    vertex = auto()
    curve = auto()
    surface = auto()
    volume = auto()

    def get_cubit_string(self):
        """Return the string that represents this item in cubit."""

        if self == self.vertex:
            return "vertex"
        elif self == self.curve:
            return "curve"
        elif self == self.surface:
            return "surface"
        elif self == self.volume:
            return "volume"
        else:
            raise ValueError("Got unexpected type {}!".format(self))

    def get_dat_bc_section_string(self):
        """
        Return the string that represents this item in a dat file section.
        """

        if self == self.vertex:
            return "POINT"
        elif self == self.curve:
            return "LINE"
        elif self == self.surface:
            return "SURF"
        elif self == self.volume:
            return "VOL"
        else:
            raise ValueError("Got unexpected type {}!".format(self))


class FiniteElementObject(Enum):
    """Enum for finite element objects."""

    hex = auto()
    tet = auto()
    face = auto()
    triangle = auto()
    edge = auto()
    node = auto()

    def get_cubit_string(self):
        """Return the string that represents this item in cubit."""

        if self == self.hex:
            return "hex"
        elif self == self.tet:
            return "tet"
        elif self == self.face:
            return "face"
        elif self == self.triangle:
            return "tri"
        elif self == self.edge:
            return "edge"
        elif self == self.node:
            return "node"

    def get_dat_bc_section_string(self):
        """
        Return the string that represents this item in a dat file section.
        Currently this only makes sense for the node type, when explicitly
        defining boundary conditions on nodes.
        """
        if self == self.node:
            return "POINT"
        else:
            raise ValueError("Got unexpected type {}!".format(self))


class CubitItems(Enum):
    """Enum for cubit internal items such as groups."""

    group = auto()


class ElementType(Enum):
    """Enum for finite element shape types."""

    hex8 = auto()
    hex20 = auto()
    hex27 = auto()
    tet4 = auto()
    tet10 = auto()
    hex8sh = auto()
    hex8_fluid = auto()

    def get_cubit_names(self):
        """
        Get the strings that are needed to mesh and describe this element in
        cubit.
        """

        # Get the element type parameters.
        if self == self.hex8 or self == self.hex8sh or self == self.hex8_fluid:
            cubit_scheme = "Auto"
            cubit_element_type = "HEX8"
        elif self == self.hex20:
            cubit_scheme = "Auto"
            cubit_element_type = "HEX20"
        elif self == self.hex27:
            cubit_scheme = "Auto"
            cubit_element_type = "HEX27"
        elif self == self.tet4:
            cubit_scheme = "Tetmesh"
            cubit_element_type = "TETRA4"
        elif self == self.tet10:
            cubit_scheme = "Tetmesh"
            cubit_element_type = "TETRA10"
        else:
            raise ValueError("Got wrong element type {}!".format(self))

        return cubit_scheme, cubit_element_type

    def get_baci_name(self):
        """Get the name of this element in baci."""

        # Get the element type parameters.
        if self == self.hex8:
            return "SOLIDH8"
        elif self == self.hex20:
            return "SOLIDH20"
        elif self == self.hex27:
            return "SOLIDH27"
        elif self == self.tet4:
            return "SOLIDT4"
        elif self == self.tet10:
            return "SOLIDT10"
        elif self == self.hex8sh:
            return "SOLIDSH8"
        elif self == self.hex8_fluid:
            return "FLUID"
        else:
            raise ValueError("Got wrong element type {}!".format(self))

    def get_baci_section(self):
        """Get the correct section name of this element in baci."""

        if self == self.hex8_fluid:
            return "FLUID"
        elif (
            self == self.hex20
            or self == self.hex8
            or self == self.hex20
            or self == self.hex27
            or self == self.tet4
            or self == self.hex8sh
            or self == self.tet10
        ):
            return "STRUCTURE"
        else:
            raise ValueError("Got wrong element type {}!".format(self))

    def get_default_baci_description(self):
        """
        Get the default text for the description in baci after the material
        string.
        """

        # Get the element type parameters.
        if self == self.hex8:
            return "KINEM nonlinear EAS none"
        elif (
            self == self.hex20
            or self == self.hex27
            or self == self.tet4
            or self == self.tet10
        ):
            return "KINEM nonlinear"
        elif self == self.hex8sh:
            return "KINEM nonlinear EAS none ANS none THICKDIR auto"
        elif self == self.hex8_fluid:
            return "NA ALE"
        else:
            raise ValueError("Got wrong element type {}!".format(self))


class BoundaryConditionType(Enum):
    """Enum for boundary conditions types."""

    dirichlet = auto()
    neumann = auto()
    point_coupling = auto()
    beam_to_solid_volume_meshtying = auto()
    beam_to_solid_surface_meshtying = auto()
    beam_to_solid_surface_contact = auto()
    solid_to_solid_surface_contact = auto()
    fsi_coupling = auto()
    ale_dirichlet = auto()

    def get_dat_bc_section_header(self, geometry_type):
        """
        Get the header string for the boundary condition input section in the
        dat file.
        """

        if self == self.dirichlet or self == self.neumann:
            if self == self.dirichlet:
                self_string = "DIRICH"
            else:
                self_string = "NEUMANN"

            return "DESIGN {} {} CONDITIONS".format(
                geometry_type.get_dat_bc_section_string(), self_string
            )
        elif (
            self == self.beam_to_solid_volume_meshtying
            and geometry_type == GeometryType.volume
        ):
            return "BEAM INTERACTION/BEAM TO SOLID VOLUME MESHTYING VOLUME"
        elif (
            self == self.beam_to_solid_surface_meshtying
            and geometry_type == GeometryType.surface
        ):
            return "BEAM INTERACTION/BEAM TO SOLID SURFACE MESHTYING SURFACE"
        elif (
            self == self.beam_to_solid_surface_contact
            and geometry_type == GeometryType.surface
        ):
            return "BEAM INTERACTION/BEAM TO SOLID SURFACE CONTACT SURFACE"
        elif self == self.point_coupling and (
            geometry_type == GeometryType.vertex
            or geometry_type == FiniteElementObject.node
        ):
            return "DESIGN POINT COUPLING CONDITIONS"
        elif self == self.solid_to_solid_surface_contact and (
            geometry_type == GeometryType.surface
            or geometry_type == FiniteElementObject.node
        ):
            return "DESIGN SURF MORTAR CONTACT CONDITIONS 3D"
        elif self == self.fsi_coupling and (
            geometry_type == GeometryType.surface
            or geometry_type == FiniteElementObject.node
        ):
            return "DESIGN FSI COUPLING SURF CONDITIONS"
        elif self == self.ale_dirichlet and (
            geometry_type == GeometryType.surface
            or geometry_type == FiniteElementObject.node
        ):
            return "DESIGN SURF ALE DIRICH CONDITIONS"
        else:
            raise ValueError(
                "No implemented case for {} and {}!".format(self, geometry_type)
            )
