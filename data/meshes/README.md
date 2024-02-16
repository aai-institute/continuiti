# Meshes

## Gmsh

For demonstration purposes, we provide a few example geometries and meshes in the Gmsh format.

Please install `gmsh` to generate the meshes. You can download it from the [official website](https://gmsh.info/) or via `pip install gmsh`.

### Mediterranean

The `mediterranean.geo` is a 2D surface geometry of the Mediterranean Sea. ([Source](https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/benchmarks/2d_large/mediterranean.geo))

To generate the mesh, run the following command:

```
gmsh -2 mediterranean.geo
```

### Naca0012

The `naca0012.geo` is a 3D geometry of the NACA0012 wing profile.
([Source](https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/benchmarks/3d/naca0012.geo))

To generate the mesh, run the following command:

```
gmsh -3 naca0012.geo
```
