# pymfem-parla-integration
This repo holds the code developed for the project involving the integration of Parla with MFEM.

# Remarks on building PyMFEM
There are several complications involving the build for PyMFEM. This project is under continuous development, so there is limited documentation. Additionally, building PyMFEM requires building the various dependencies for MFEM, which, at a minimum, requires Metis and Hypre. Other libraries such as libCEED, GLVis, and GSLib require additional commands. In order to build PyMFEM, users will need to have SWIG installed. We used swig/4.0.0 on Frontera for our experiments.

The parallel CPU build requires Metis, which (at the time of this writing) has undergone signficant changes and does not appear to function within MFEM. Instead, we recommend grabbing the tarball `Metis-5.1.0.tar.gz` in `mfem/tpls` and building with `cc=icc` and `shared=1`. Then, one needs to copy the `.so` file in the build directory to a new `lib` directory so that MFEM can see this.

Once we have built Metis, we can navigate to the top level of the PyMFEM directory and run the following:

`python setup.py install --with-parallel --metis-prefix=~/Projects/metis-5.1.0/ --CC=icc --CXX=icc --MPICC=mpiicc --MPICXX=mpiicc`

This installation command passes the location of the shared lib file for Metis to our script. The remaining builds for Hypre and MFEM are called internally in the script; however, it is possible to pass pre-built lib files to this script to reduce the build time. There is a help command in the file `setup.py` that lists some build options.

# Remarks on the documentation for PyMFEM
Documentation for PyMFEM is located in `docs/install.txt` and `docs/manual.txt`. The installation file provides several use cases for building the wrappers, most of which will likely be depracated. Details concerning some of the basic functionality of the code and some methods, including accessing Numpy arrays can be found in the manual file. For more complicated use cases, we found that it was necessary to parse through some of the SWIG generated python files to get information about the classes and interfaces. While much of the notation and code functionality mimics MFEM, some of the interfaces are different.

# Remarks on the documentation for Parla
The steps for the installation of Parla are fairly straightforward and are easily accessible from the repo. As for the code documentation, things are fairly well documented, but the code is under continued development. For more complicated use cases, we recommend viewing the tutorials before the more complicated examples. There are some conceptual gaps between these two sets of examples, though some of the documentation for the more advanced use cases is described in the SC'22 paper.
