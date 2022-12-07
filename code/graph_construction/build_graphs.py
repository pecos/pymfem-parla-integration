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


# Parse the command line data
parser = argparse.ArgumentParser(description="Build directed graphs for a discrete ordinates calculation.")
parser.add_argument("-meshfile", default="star.mesh", type=str, help="Mesh file to use.")
parser.add_argument("-M", type=int, default=4, help="Order of the Chebyshev-Legendre quadrature.")
parser.add_argument("-max_size", type=int, default=100000, help="Upper limit on the total number of mesh elements.")

args = parser.parse_args()

print("\nOptions used:\n")

for arg in vars(args):
    print(arg, "=", getattr(args, arg))

print("\n")

# Set some variables from the command line input
meshfile = args.meshfile
M = args.M
max_size = args.max_size

meshfile_path = expanduser(join(os.path.dirname(__file__), '../PyMFEM/', 'data', meshfile))


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
    Helper function to build a directed graph for a given
    ordinate using an unstructured mesh.

    The basic procedure is as follows. For each interior face:
    1) Compute the normal vector for that is shared by a pair of elements.
    2) Determine the orientation of the normal relative to the ordinate.
    3) Write the corresponding entry into a sparse data structure.
    4) Write the sparse data to a file

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
        if mesh.FaceIsInterior(i):

            # Get the elements and transformation associated with this shared face
            # Note: The convention is that the normal vector for element 2 is taken to be
            # the normal vector for element 1 multiplied by -1.
            elem1, elem2 = mesh.GetFaceElements(i)

            # PyMFEM doesn't allow us to access the mesh method "GetFaceElementTransformations"
            # Instead, we call the "GetFaceTransformation" method and associate the norm with elem1
            FTr = mesh.GetFaceTransformation(i)

            # Set the point at which the Jacobian is to be evaluated. This will give the orientation
            # of the normal vector. We use the geometric center of the face
            FTr.SetIntPoint(mfem.Geometries.GetCenter(FTr.GetGeometryType()))

            # Compute the normal vector (not necessarily unit length)
            # We take this to be associated with element 1
            mfem.CalcOrtho(FTr.Jacobian(), normal)

            # Now check the orientation of the normal relative to the ordinate
            if np.dot(ordinate, normal.GetDataArray()) > 0.0:

                # The normal points in the same direction as the ordinate 
                # This means that there is an edge connecting elem1 to elem2
                dir_graph[elem1, elem2] = 1

            # We use the convention that element faces orthogonal to the ordinate
            # are not considered to be downwind

    # Change the dok_matrix to a csr format
    dir_graph_csr = dir_graph.tocsr(copy=False)

    # Write the sparse matrix to a Matrix Market format
    io.mmwrite(fname + ".mtx", dir_graph_csr, comment="", field="integer", precision=None, symmetry=None)

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
    output_dir = meshfile.replace(".data","") + "-M-" + str(M)

    # Remove this directory and its contents
    os.system(" ".join(["rm", "-rf", output_dir]))
    os.makedirs(output_dir)

    build_start = time.perf_counter()

    # Loop over the ordinates and construct the directed graph for the mesh
    for n in range(num_ordinates):

        s_idx = dim*n
        e_idx = s_idx + dim

        # Put the file name here, but do not include the extension ".data" or file type
        # The file type is specified in the function that builds the graph
        file_name = output_dir + "/" + meshfile.replace(".data","") + "-M-" + str(M) + "-idx-" + str(n)

        build_directed_graph(mesh, ordinates_cl[s_idx:e_idx], file_name)

        print("Finished processing ordinate %d" %n)

    build_end = time.perf_counter()
    total_build = build_end - build_start

    print("Total build time took %e (s)"%total_build)

if __name__ == "__main__":

    main()






