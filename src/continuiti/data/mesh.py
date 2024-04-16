"""
`continuiti.data.mesh`

Mesh file readers.
"""

import torch
import numpy as np


class Gmsh:
    """
    [Gmsh](https://gmsh.info/) is an open source 3D finite element mesh generator.

    You can find example `.msh` files in the `data/meshes` directory.

    A `Gmsh` object reads and provides the content of a Gmsh file
    in a tensor format that can also be used with `matplotlib`.

    Example:
        ```python
        mesh = Gmsh("path/to/file.msh")
        vertices = mesh.get_vertices()
        cells = mesh.get_cells()

        from matplotlib.tri import Triangulation
        tri = Triangulation(vertices[:, 0], vertices[:, 1], cells)
        ```

    Args:
        filename: Path to `.msh` file.
    """

    def __init__(
        self,
        filename: str,
    ):
        import gmsh

        self.filename = filename

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
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
        v = v.transpose(0, 1)
        return v

    def get_cells(self) -> torch.Tensor:
        """Get cells (elements) as tensor."""
        cells = np.array(self.elements[2], dtype=int)
        e = torch.tensor(cells)
        e = e.reshape(-1, 3)
        e = e - 1  # Gmsh uses 1-based indexing
        return e
