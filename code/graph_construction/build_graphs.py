"""
   Code to build a series of directed graphs associated with a collection of discrete ordinates 
   for a computational mesh. This code was converted from ex1.py in PyMFEM.

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

import os
from os.path import expanduser, join
import numpy as np
import scipy as sp

import mfem.ser as mfem

# Define the quadrature sets for 1-, 2-, and 3-D meshs
# We will be using the Chebyshev-Legendre nodes to do this

def get_1D_chebyshev_legendre(N):
    """
    In 1-D, this is the same as using N Gauss-Legendre nodes
    """

    mu_GL, w_GL = np.polynomial.legendre.leggauss(N)

    return mu_GL, w_GL 


def get_2D_chebyshev_legendre(N):
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
        w_CL[k] = np.pi/N

        # Quadrature locations
        Omega_CL[k,0] = np.cos(phi_k)
        Omega_CL[k,1] = np.sin(phi_k)

    return w_CL, Omega_CL


def get_3D_chebyshev_legendre(N):
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
    dir_graph = sp.sparse.dok_matrix((NE,NE), dtype=np.int32)

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
            if np.dot(ordinate, normal.GetData()) > 0.0:

                # The normal points in the same direction as the ordinate 
                # This means that there is an edge connecting elem1 to elem2
                dir_graph[elem1, elem2] = 1

            # We use the convention that element faces orthogonal to the ordinate
            # are not considered to be downwind

    # Change the dok_matrix to a csr format
    dir_graph_csr = dir_graph.tocsr(copy=False)

    # Write the sparse data to a file 
    sp.sparse.save_npz(filename + ".npz", dir_graph_csr)

    return None







def run(order=1, static_cond=False,
        meshfile='', visualization=False,
        device='cpu', pa=False):
    '''
    run ex1
    '''

    device = mfem.Device(device)
    device.Print()

    mesh = mfem.Mesh(meshfile, 1, 1)
    dim = mesh.Dimension()

    #   3. Refine the mesh to increase the resolution. In this example we do
    #      'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    #      largest number that gives a final mesh with no more than 50,000
    #      elements.
    ref_levels = int(
        np.floor(
            np.log(
                50000. /
                mesh.GetNE()) /
            np.log(2.) /
            dim))
    for x in range(ref_levels):
        mesh.UniformRefinement()

    # 5. Define a finite element space on the mesh. Here we use vector finite
    #   elements, i.e. dim copies of a scalar finite element space. The vector
    #   dimension is specified by the last argument of the FiniteElementSpace
    #   constructor. For NURBS meshes, we use the (degree elevated) NURBS space
    #   associated with the mesh nodes.
    if order > 0:
        fec = mfem.H1_FECollection(order, dim)
    elif mesh.GetNodes():
        fec = mesh.GetNodes().OwnFEC()
        print("Using isoparametric FEs: " + str(fec.Name()))
    else:
        order = 1
        fec = mfem.H1_FECollection(order, dim)
    fespace = mfem.FiniteElementSpace(mesh, fec)
    print('Number of finite element unknowns: ' +
          str(fespace.GetTrueVSize()))

    # 5. Determine the list of true (i.e. conforming) essential boundary dofs.
    #    In this example, the boundary conditions are defined by marking all
    #    the boundary attributes from the mesh as essential (Dirichlet) and
    #    converting them to a list of true dofs.
    ess_tdof_list = mfem.intArray()
    if mesh.bdr_attributes.Size() > 0:
        ess_bdr = mfem.intArray([1] * mesh.bdr_attributes.Max())
        ess_bdr = mfem.intArray(mesh.bdr_attributes.Max())
        ess_bdr.Assign(1)
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    # 6. Set up the linear form b(.) which corresponds to the right-hand side of
    #   the FEM linear system, which in this case is (1,phi_i) where phi_i are
    #   the basis functions in the finite element fespace.
    b = mfem.LinearForm(fespace)
    one = mfem.ConstantCoefficient(1.0)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()



    # Try a test involving my variant of the LFIntegrator
    my_b = mfem.LinearForm(fespace)
    my_b.AddDomainIntegrator(MyDomainLFIntegrator(one))
    my_b.Assemble()


    # 7. Define the solution vector x as a finite element grid function
    #   corresponding to fespace. Initialize x with initial guess of zero,
    #   which satisfies the boundary conditions.
    x = mfem.GridFunction(fespace)
    x.Assign(0.0)

    # 8. Set up the bilinear form a(.,.) on the finite element space
    #   corresponding to the Laplacian operator -Delta, by adding the Diffusion
    #   domain integrator.
    a = mfem.BilinearForm(fespace)
    if pa:
        a.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))

    # 9. Assemble the bilinear form and the corresponding linear system,
    #   applying any necessary transformations such as: eliminating boundary
    #   conditions, applying conforming constraints for non-conforming AMR,
    #   static condensation, etc.
    if static_cond:
        a.EnableStaticCondensation()
    a.Assemble()

    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()

    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)
    print("Size of linear system: " + str(A.Height()))

    # Should also form the linear system for my modified B
    my_B = mfem.Vector()
    a.FormLinearSystem(ess_tdof_list, x, my_b, A, X, my_B)

    # Compare the output of the two RHS functions
    print("B =", B.GetDataArray(),"\n")
    print("my_B =", my_B.GetDataArray(),"\n")

    # Covert the MFEM Vectors to Numpy arrays for norms
    B_array = B.GetDataArray()
    my_B_array = my_B.GetDataArray()

    # Relative error against the MFEM output in the 2-norm
    rel_err_1 = np.linalg.norm(my_B_array - B_array, 1)/np.linalg.norm(B_array, 1)
    rel_err_2 = np.linalg.norm(my_B_array - B_array, 2)/np.linalg.norm(B_array, 2)
    rel_err_inf = np.linalg.norm(my_B_array - B_array, np.inf)/np.linalg.norm(B_array, np.inf)

    print("Relative error in the rhs (1-norm):", rel_err_1, "\n")
    print("Relative error in the rhs (2-norm):", rel_err_2, "\n")
    print("Relative error in the rhs (inf-norm):", rel_err_inf, "\n")









    # 10. Solve
    #if pa:
    #    if mfem.UsesTensorBasis(fespace):
    #        M = mfem.OperatorJacobiSmoother(a, ess_tdof_list)
    #        mfem.PCG(A, M, B, X, 1, 4000, 1e-12, 0.0)
    #    else:
    #        mfem.CG(A, B, X, 1, 400, 1e-12, 0.0)
    #else:
        #AA = mfem.OperatorHandle2SparseMatrix(A)
    #    AA = A.AsSparseMatrix()
    #    M = mfem.GSSmoother(AA)
    #    mfem.PCG(A, M, B, X, 1, 200, 1e-12, 0.0)

    # 11. Recover the solution as a finite element grid function.
    #a.RecoverFEMSolution(X, b, x)

    # 12. Save the refined mesh and the solution. This output can be viewed later
    #     using GLVis: "glvis -m refined.mesh -g sol.gf".
    #mesh.Print('refined.mesh', 8)
    #x.Save('sol.gf', 8)

    # 13. Send the solution by socket to a GLVis server.
    #if (visualization):
    #    sol_sock = mfem.socketstream("localhost", 19916)
    #    sol_sock.precision(8)
    #    sol_sock.send_solution(mesh, x)


if __name__ == "__main__":
    from mfem.common.arg_parser import ArgParser

    parser = ArgParser(description='Ex1 (Laplace Problem)')
    parser.add_argument('-m', '--mesh',
                        default='star.mesh',
                        action='store', type=str,
                        help='Mesh file to use.')
    parser.add_argument('-vis', '--visualization',
                        action='store_true',
                        help='Enable GLVis visualization')
    parser.add_argument('-o', '--order',
                        action='store', default=1, type=int,
                        help="Finite element order (polynomial degree) or -1 for isoparametric space.")
    parser.add_argument('-sc', '--static-condensation',
                        action='store_true',
                        help="Enable static condensation.")
    parser.add_argument("-pa", "--partial-assembly",
                        action='store_true',
                        help="Enable Partial Assembly.")
    parser.add_argument("-d", "--device",
                        default="cpu", type=str,
                        help="Device configuration string, see Device::Configure().")

    args = parser.parse_args()
    parser.print_options(args)

    order = args.order
    static_cond = args.static_condensation
    # ../PyMFEM/data
    meshfile = expanduser(
        join(os.path.dirname(__file__), '../PyMFEM/', 'data', args.mesh))
    visualization = args.visualization
    device = args.device
    pa = args.partial_assembly

    run(order=order,
        static_cond=static_cond,
        meshfile=meshfile,
        visualization=visualization,
        device=device,
        pa=pa)







