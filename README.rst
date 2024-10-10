========
fem_glue
========

Welcome! fem_glue is (or will be, it is currently in an early stage of development) a package for setting up finite element method (FEM) problems through a simple API. 

The aim of fem_glue is to empower anyone to set up FEM problems with complex geometries and boundary conditions with relative ease.
There already exists extraordinary open-source FEM packages like fenicsx, and in fact, fem_glue will delegate most of the numerical heavy lifting to these packages.
Packages like fenicsx are great for setting up research and academic problems, but they are not so straightforward for setting up complex geometries, like those of buildings for example.
Furthermore, a relatively deep understanding of FEM is generally necessary for using something like fenicsx.
Therefore, the purpose of fem_glue is to simplify the process of setting up FEM problems so that it can be used in day-to-day FEM workflows by regular engineers and FEM enthusiasts.

Note
====
This package is still at an early stage of development, so its end-goal functionality still has to be achieved.

Currently, the _geometry_ package of fem_glue is being developed. 
This package defines shapes in 3D space and operations between them. 
For example, finding the intersection between lines and polygons.
This functionality is similar to that of the _shapely_ package, but for 3D space.
Eventually, the _geometry_ sub-package will become its own standalone package and repository, but for this early stage its happy where it is.

After the _geometry_ package is done, a package for defining meshes from the geometry will be written.
This package will collect all the metadata for meshing, and delegate the actual meshing to a well-established meshing software such as _gmsh_.

Once the _geometry_ and meshing packages are complete, the actual _fem_glue_ will be developed.
This package will use the other two packages for defining geometries and meshes.
It will define finite element shapes and boundary conditions that can then be delegated to well-established solvers like _fenicsx_.

Installation
====
To install the package as a user run the following command at the top-level of the repository:
``
pip install .
``

To install it as a developer run the following command instead:
``
pip install -e .[testing,dev]
``

.. include:: CONTRIBUTING.rst
