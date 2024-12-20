.. _gui:

Graphical User Interface (GUI)
------------------------------
The GUI is mainly meant for educational purpose, providing a user-friendly way to illustrate the relationships between
the components of the stiffness tensor of a single crystal, depending on its symmetries. It also allows to directly plot
the engineering properties.

The figure bellow illustrates the Young's modulus of
`monoclinic TiNi <https://next-gen.materialsproject.org/materials/mp-1048>`_:

.. image:: images/GUI.png
    :width: 600

To launch this GUI, just run::

    from Elasticipy.gui import crystal_elastic_plotter
    crystal_elastic_plotter()