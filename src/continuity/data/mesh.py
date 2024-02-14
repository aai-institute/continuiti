"""Gmsh."""

import torch
import gmsh
import numpy as np


class Gmsh:
    """
    A `Gmsh` objects contains the content of a Gmsh `.msh` file.

    Args:
        filename: Path to `.msh` file.
    """

    def __init__(
        self,
        filename: str,
    ):
        self.filename = filename

        gmsh.initialize()
        gmsh.open(self.filename)
        self.dim = gmsh.model.getDimension()
        self.nodes = gmsh.model.mesh.getNodes()
        self.elements = gmsh.model.mesh.getElements()
        gmsh.finalize()

        super().__init__()

    def get_vertices(self) -> torch.Tensor:
        """Get vertices (nodes) as tensor."""
        v = torch.tensor(self.nodes[1], dtype=torch.get_default_dtype())
        v = v.reshape(-1, 3)  # Gmsh vertices are always 3D
        return v

    def get_cells(self) -> torch.Tensor:
        """Get cells (elements) as tensor."""
        cells = np.array(self.elements[2], dtype=int)
        e = torch.tensor(cells)
        e = e.reshape(-1, 3)
        e = e - 1  # Gmsh uses 1-based indexing
        return e
