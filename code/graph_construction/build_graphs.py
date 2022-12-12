"""
   Code to build a series of directed graphs associated with a collection of discrete ordinates 
   for 2-D and 3-D computational meshes. This code was converted from ex1.py in PyMFEM.

   How to run:
      python <arguments>

   Example of some mesh arguments:
      build_graphs.py star.mesh
      build_graphs.py square-disc.mesh
      build_graphs.py escher.mesh
      build_graphs.py fichera.mesh
      build_graphs.py square-disc-p3.mesh
      build_graphs.py square-disc-nurbs.mesh
      build_graphs.py disc-nurbs.mesh
      build_graphs.py pipe-nurbs.mesh
      build_graphs.py star-surf.mesh
      build_graphs.py square-disc-surf.mesh
      build_graphs.py inline-segment.mesh
      build_graphs.py amr-quad.mesh
      build_graphs.py amr-hex.mesh
      build_graphs.py fichera-amr.mesh
      build_graphs.py mobius-strip.mesh

   Description:  This code takes as input a mesh file and angular quadrature order
   and constructs a collection of directed graphs, which are written to files in a
   sparse matrix format.
"""

import argparse
import os
from os.path import expanduser, join 
import sys

# Add the location of the PyMFEM directory to the path
sys.path.append("../PyMFEM/mfem")

# Parse the command line data
parser = argparse.ArgumentParser(description="Build directed graphs for a discrete ordinates calculation.")
parser.add_argument("-meshfile", default="star.mesh", type=str, help="Mesh file to use.")
parser.add_argument("-M", type=int, default=4, help="Order of the Chebyshev-Legendre quadrature.")
parser.add_argument("-max_size", type=int, default=100000, help="Upper limit on the total number of mesh elements.")
parser.add_argument("-debug", type=bool, default=False, help="Whether or not to run the debug script. If true, we don't run main!")

args = parser.parse_args()

print("\nOptions used:\n")

for arg in vars(args):
    print(arg, "=", getattr(args, arg))

print("\n")

# Set some variables from the command line input
meshfile = args.meshfile
M = args.M
max_size = args.max_size
run_debug = args.debug

# Specify the location of the FEM meshfile
meshfile_path = expanduser(join(os.path.dirname(__file__), "./sample_meshes/", meshfile))


import numpy as np
import scipy.sparse as sparse
import scipy.io as io

import mfem.ser as mfem

import time

# Define the quadrature sets for 2-D and 3-D meshes
# We will be using the Chebyshev-Legendre nodes to do this

def get_chebyshev_legendre_quadrature2D(N):
    """
    In the case of a 2-D domain, we assume that we are looking
    at z = 0 in the unit sphere. This returns a set of 2N points.
    """

    # Storage for the quadrature set
    w_CL = np.zeros([2*N])
    Omega_CL = np.zeros([2*N,2])

    for k in range(2*N):

        # Compute the lateral angle of the sphere
        phi_k = (2*k + 1)*np.pi/(2*N)

        # Compute the quadrature weights
        w_CL[k] = 2*np.pi/N

        # Quadrature locations
        Omega_CL[k,0] = np.cos(phi_k)
        Omega_CL[k,1] = np.sin(phi_k)

    return w_CL, Omega_CL


def get_chebyshev_legendre_quadrature3D(N):
    """
    Computes the order N Chebyshev-Legendre quadrature set on the unit sphere.
    The CL quadratures of order N require 2N^{2} integration weights/nodes. It
    is constructed from a tensor product of an order N Gauss-legendre quadrature
    with a midpoint quadrature using 2N points
    """

    # First get the GL quadrature locations and weights on [-1,1]
    mu_GL, w_GL = np.polynomial.legendre.leggauss(N)

    # Storage for the quadrature set
    w_CL = np.zeros([2*N,N])
    Omega_CL = np.zeros([2*N,N,3])

    for k in range(2*N):

        # Compute the lateral angle of the sphere
        phi_k = (2*k + 1)*np.pi/(2*N)

        for ell in range(N):

            # Quadrature weights
            w_CL[k,ell] = (np.pi/N)*w_GL[ell]

            # Quadrature locations
            Omega_CL[k,ell,0] = np.cos(phi_k)*np.sqrt(1 - mu_GL[ell]**2)
            Omega_CL[k,ell,1] = np.sin(phi_k)*np.sqrt(1 - mu_GL[ell]**2)
            Omega_CL[k,ell,2] = mu_GL[ell]

    return w_CL, Omega_CL


def build_directed_graph(mesh, ordinate, fname):
    """
    Helper function to build a directed graph, for a given
    ordinate, using an unstructured mesh.

    The basic procedure is as follows. For each interior face:
    1) Compute the normal vector for that is shared by a pair of elements.
    2) Determine the orientation of the normal relative to the ordinate.
    3) Write the corresponding entries into a sparse data structure.
    4) Write the sparse data to a file.

    Note: The filename that is passed in does not include an
    extension. We supply it inside this function, since we are
    using the utilities provided by SciPy.
    """

    # Number of mesh elements and problem dimension
    NE = mesh.GetNE()
    dim = mesh.Dimension()

    # The mesh dimension and ordinate dimension should match
    assert dim == ordinate.size

    # First, create the sparse data structure that will hold this particular
    # directed graph. We use a dictionary of keys structure that holds integers.
    # If the (i,j) entry is 1, then this means that there is an ordered edge connecting
    # elements i and j. We can use scipy utilities to convert the formats to other types
    dir_graph = sparse.dok_matrix((NE,NE), dtype=np.int32)

    # Create mfem::Vector to hold the norm of the face
    normal = mfem.Vector(dim)

    for i in range(mesh.GetNumFaces()):

        # There are no elements upwind of exterior faces 
        # so we only look at those on the interior
        if mesh.FaceIsInterior(i):

            # Get the elements and transformation associated with this shared face
            # Note: The convention is that the normal vector for element 2 points
            # in the opposite direction as the normal vector for element 1
            elem1, elem2 = mesh.GetFaceElements(i)

            # Conventions for MFEM regarding conforming and non-conforming faces
            # Conforming: Element 1 (smaller ID) handles the integration
            # Non-conforming: The slave element handles the integration
            # Source: FaceInfo documentation
            #
            # Question: How to access the bool: NCFace?
            # This is normally part of the FaceInfo struct, but the method below doesn't
            # seem to allow access to this....
            # elem1Inf, elem2Inf = mesh.GetFaceInfos(i)

            # Once this situation is resolved, we need to add a condition
            # that checks if these conventions are met, and if so, perform those
            # steps. Also, we should be careful with the ordering of the elements

            # PyMFEM doesn't allow us to access the mesh method "GetFaceElementTransformations"
            # Instead, we call the "GetFaceTransformation" method and associate the norm with elem1
            FTr = mesh.GetFaceTransformation(i)

            # Get the integration rule which will be used to set the points evaluating the normal
            # We will use the order extracted from the element face transformation to do this
            ir = mfem.IntRules.Get(mesh.GetFaceGeometry(i), 2*FTr.Order())

            # Next, we will loop over the integration points in the rule and compute the
            # normal at each of these points. We then determine the orientation of this
            # point relative to the ordinate direction. 
            for j in range(ir.GetNPoints()):

                # Extract the integration point and set the location in the transformation object
                # This gives an orientation to the normal vector
                ip = ir.IntPoint(j)
                FTr.SetIntPoint(ip)

                # Compute the normal vector (not necessarily unit length)
                # We take this to be associated with element 1
                # See: https://mfem.org/howto/outer_normals/
                mfem.CalcOrtho(FTr.Jacobian(), normal)

                # Now check the orientation of the normal relative to the ordinate
                alignment = np.dot(ordinate, normal.GetDataArray())

                if alignment > 0.0:

                    # The normal on element 1 aligns with the ordinate so
                    # there is an edge connecting elem1 to elem2
                    dir_graph[elem1, elem2] = 1

                elif alignment < 0.0:

                    # If the normal on element 1 points in the opposite direction
                    # of the ordinate, there is an edge connecting elem2 to elem1
                    # In other words, the normal on element 2 is aligned
                    dir_graph[elem2, elem1] = 1

                else:

                    # We use the convention that element faces orthogonal to the ordinate
                    # are not considered to be upwind/downwind, so we do nothing here.
                    pass

    # Change the dok_matrix to a csr format
    dir_graph_csr = dir_graph.tocsr(copy=False)

    # Write the sparse matrix to a Matrix Market format
    io.mmwrite(fname + ".mtx", dir_graph_csr, comment="", field="integer", precision=None, symmetry=None)

    return None


def debug_directed_graph(mesh, ordinate):
    """
    Helper function to debug the construction of a directed graph, for a given
    ordinate, using an unstructured mesh.
    """

    # Number of mesh elements and problem dimension
    NE = mesh.GetNE()
    dim = mesh.Dimension()

    # The mesh dimension and ordinate dimension should match
    assert dim == ordinate.size

    # Create mfem::Vector to hold the norm of the face
    normal = mfem.Vector(dim)

    for i in range(mesh.GetNumFaces()):

        # There are no elements upwind of exterior faces 
        # so we only look at those on the interior
        if mesh.FaceIsInterior(i):

            # Get the elements and transformation associated with this shared face
            # Note: The convention is that the normal vector for element 2 points
            # in the opposite direction as the normal vector for element 1
            elem1, elem2 = mesh.GetFaceElements(i)

            # PyMFEM doesn't allow us to access the mesh method "GetFaceElementTransformations"
            # Instead, we call the "GetFaceTransformation" method and associate the norm with elem1
            FTr = mesh.GetFaceTransformation(i)

            # Set the point at which the Jacobian is to be evaluated. This will give the orientation
            # of the normal vector. We use the geometric center of the face
            FTr.SetIntPoint(mfem.Geometries.GetCenter(FTr.GetGeometryType()))

            # Compute the normal vector (not necessarily unit length)
            # We take this to be associated with element 1
            # See: https://mfem.org/howto/outer_normals/
            mfem.CalcOrtho(FTr.Jacobian(), normal)

            # Test code:
            print("Working on face i = %d"%i)

            print("elem1 = %d"%elem1)
            print("elem2 = %d"%elem2)

            # Compute the element centers
            elem1_center = mesh.GetElementCenterArray(elem1)
            elem2_center = mesh.GetElementCenterArray(elem2)

            print("elem1 center = ", elem1_center)
            print("elem2 center = ", elem2_center)

            print("ordinate = ", ordinate)
            print("Normal vector associated with elem1 = ", normal.GetDataArray())
            print("Dot product elem1 n_f * ordinate = ", np.dot(ordinate, normal.GetDataArray()))
            print("Dot product elem2 n_f * ordinate = ", np.dot(ordinate, -normal.GetDataArray()))

            print("\n")

    return None


def debug_setup():
    """
    Simple script to aid the debugging of the graph traversal

    We fix some ordinate directions and check that the conditions
    are met for storing an edge, based on the physical location of the elements.

    This won't work for all elements, but it is a quick way to check the basic components.
    """

    global meshfile, meshfile_path, M, max_size

    # Read in the meshfile
    mesh = mfem.Mesh(meshfile_path, 1, 1)
    dim = mesh.Dimension()

    print("Number of spatial dimensions: %d"%dim)

    assert dim == 2, "Error: We only support 2-D meshes for debugging."

    # Keep the mesh coarse for debugging purposes

    print("Number of finite elements: " + str(mesh.GetNE()))

    # Hard code the ordinate_directions for a 2-D case
    # For this, we'll look at the unit vectors to make sure things point in the correct direction
    ord_case0 = np.array([1.0,0.0])
    ord_case1 = np.array([-1.0,0.0])
    ord_case2 = np.array([0.0, 1.0])
    ord_case3 = np.array([0.0,-1.0])

    # Case 0:
    debug_directed_graph(mesh, ord_case0)
    print("Finished processing test ordinate 0.\n")

    # Case 1:
    debug_directed_graph(mesh, ord_case1)
    print("Finished processing test ordinate 1.\n")

    # Case 2:
    debug_directed_graph(mesh, ord_case2)
    print("Finished processing test ordinate 2.\n")

    # Case 3:
    debug_directed_graph(mesh, ord_case3)
    print("Finished processing test ordinate 3.\n")

    return None


def main():

    global meshfile, meshfile_path, M, max_size

    # Read in the meshfile
    mesh = mfem.Mesh(meshfile_path, 1, 1)
    dim = mesh.Dimension()

    print("Number of spatial dimensions: %d"%dim)

    assert dim > 1, "Error: Only 2-D and 3-D meshes are supported."

    # Refine the mesh to increase the resolution. In this example we do
    # 'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    # largest number that gives a final mesh with no more than 'max_size'elements.
    ref_levels = int(np.floor(np.log(max_size / mesh.GetNE()) / np.log(2.) /dim))

    for x in range(ref_levels):
        mesh.UniformRefinement()

    print("Number of finite elements: " + str(mesh.GetNE()))

    # Construct the ordinate arrays for the mesh based on the dimension
    # We use Chebyshev-Legendre nodes for this
    if dim == 2:
        weights_cl, ordinates_cl = get_chebyshev_legendre_quadrature2D(M)
    else:
        weights_cl, ordinates_cl = get_chebyshev_legendre_quadrature3D(M)

    # Since the 2-D and 3-D ordinates are given (respectively) 
    # as 2-D and 3-D arrays, we flatten them to provide a unique accessor
    shape = ordinates_cl.shape
    ordinates_cl = ordinates_cl.flatten()

    # Calculate the number of entries in the ordinate array 
    if dim == 2:
        num_ordinates = shape[0]
    else:
        num_ordinates = shape[0]*shape[1]

    # Specify the directory to store the graph data files that have been created
    # We will give it the name of the mesh and the order of the angular quadrature
    output_dir = meshfile + "-M-" + str(M)

    # Remove this directory and its contents
    os.system(" ".join(["rm", "-rf", output_dir]))
    os.makedirs(output_dir)

    build_start = time.perf_counter()

    # Loop over the ordinates and construct the directed graph for the mesh
    for n in range(num_ordinates):

        s_idx = dim*n
        e_idx = s_idx + dim

        # Base file name where the graph data will be stored
        # The file type is specified in the function that builds the graph
        file_name = output_dir + "/" + meshfile + "-M-" + str(M) + "-idx-" + str(n)

        build_directed_graph(mesh, ordinates_cl[s_idx:e_idx], file_name)

        print("Finished processing ordinate %d" %n)

    build_end = time.perf_counter()
    total_build = build_end - build_start

    print("Total build time took %e (s)"%total_build)

if __name__ == "__main__":

    if run_debug:

        debug_setup()

    else:

        main()






