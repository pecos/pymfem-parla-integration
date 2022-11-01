'''
   MFEM example 1 (converted from ex1.cpp)

   See c++ version in the MFEM library for more detail

   How to run:
      python <arguments>

   Example of arguments:
      ex1.py -m star.mesh
      ex1.py -m square-disc.mesh
      ex1.py -m escher.mesh
      ex1.py -m fichera.mesh
      ex1.py -m square-disc-p3.mesh -o 3
      ex1.py -m square-disc-nurbs.mesh -o -1
      ex1.py -m disc-nurbs.mesh -o -1
      ex1.py -m pipe-nurbs.mesh -o -1
      ex1.py -m star-surf.mesh
      ex1.py -m square-disc-surf.mesh
      ex1.py -m inline-segment.mesh
      ex1.py -m amr-quad.mesh
      ex1.py -m amr-hex.mesh
      ex1.py -m fichera-amr.mesh
      ex1.py -m mobius-strip.mesh
      ex1.py -m mobius-strip.mesh -o -1 -sc

   Description:  This example code demonstrates the use of MFEM to define a
                 simple finite element discretization of the Laplace problem
                 -Delta u = 1 with homogeneous Dirichlet boundary conditions.

'''
import os
from os.path import expanduser, join
import argparse
import mkl
#import gil_load

# Parse the command line information
parser = argparse.ArgumentParser(description='Ex1 (Laplace Problem)')
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
parser.add_argument('-blocks', type=int, 
                    default=1, help="Number of parallel blocks/partitions. Assume 1 thread per block.")
parser.add_argument('-trials', type=int, 
                    default=1, help="Number of repititions for timing regions of code.")

args = parser.parse_args()

print(args,"\n")

order = args.order
static_cond = args.static_condensation

meshfile = expanduser(
    join(os.path.dirname(__file__), '../PyMFEM/', 'data', args.mesh))
visualization = args.visualization
device = args.device
pa = args.partial_assembly

# Initialize the environment for the GIL tracker
#gil_load.init()

load = 1.0/args.blocks # We assume one thread per block

# Use only one thread inside the task
mkl.set_num_threads(1)
os.environ["NUMEXPR_NUM_THREADS"] = str(1)
os.environ["OMP_NUM_THREADS"] = str(1)
os.environ["OPENBLAS_NUM_THREADS"] = str(1)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)

num_trials = args.trials

import numpy as np
import numba as nb
import time

# Import any of the MFEM Python modules
import mfem.ser as mfem
from mfem.common.arg_parser import ArgParser

# Parla modules and decorators
from parla import Parla

# Here we focus on cpu devices
from parla.cpu import cpu

# Imports for the spawn decorator and TaskSpace declaration
from parla.tasks import spawn, TaskSpace

# Check the hardware
import psutil



@nb.njit([nb.void(nb.float64[:], nb.int64[:,:], nb.float64[:,:], nb.float64[:,:,:],
                  nb.float64[:,:,:], nb.int64, nb.int64)], 
                  nogil=True, cache=False, boundscheck=False)
def inner_quad_kernel(block_element_array, l2g_map, 
                      quad_wts, quad_pts, shape_pts, 
                      s_idx, e_idx):
    """
    Kernel that performs the quadrature evaluations over a particular block of elements.
    """
    # Get the dimension for the grid
    dim = quad_pts.shape[2]

    # Get the number of quad pts
    num_qp = quad_wts.shape[1]

    # Get the number of active dof within each element (assumed to be the same)
    ldof = shape_pts.shape[2]

    # Initialize a local array to hold the integral over each element
    local_integral = np.zeros((ldof))

    # Create a local array to store a quadrature point
    quad_pt = np.zeros((dim))

    # Loop over the mesh elements on this block 
    for j in range(s_idx, e_idx):

        for l in range(ldof):
            local_integral[l] = 0.0

        for k in range(num_qp):

            # Integration point from the rule in (x,y,z) format
            for d in range(dim):
                quad_pt[d] = quad_pts[j,k,d]

            # Quadrature evaluation (with f = 1)
            for l in range(ldof):
                local_integral[l] += quad_wts[j,k]*shape_pts[j,k,l] 

        # Accumulate the local integrals into the relevant entries 
        # of the block-wise global array
        for l in range(ldof):
            block_element_array[l2g_map[j,l]] += local_integral[l]

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

    print("Number of cores in system", psutil.cpu_count(),"\n")
    print("Number of physical cores in system\n")

    #   3. Refine the mesh to increase the resolution. In this example we do
    #      'ref_levels' of uniform refinement. We choose 'ref_levels' to be the
    #      largest number that gives a final mesh with no more than 50,000
    #      elements.
    #ref_levels = int(np.floor(np.log(50000./mesh.GetNE())/np.log(2.)/dim))

    ref_levels = 7

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
          str(fespace.GetTrueVSize()),"\n")

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
    pymfem_times = np.zeros([num_trials]) # Store the times for statistics

    for trial_idx in range(num_trials):

        pymfem_start = time.perf_counter()

        b = mfem.LinearForm(fespace)
        one = mfem.ConstantCoefficient(1.0)
        b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
        b.Assemble()

        pymfem_end = time.perf_counter()

        pymfem_times[trial_idx] = pymfem_end - pymfem_start

    #-----------------------------------------------------------------------------
    # Version based on Parla (naive approach)
    #-----------------------------------------------------------------------------

    # Pre-processing step
    # Get the quad information and store basis functions at the element quad pts
    # Also need to store the local-to-global index mappings used in the scatter
    intorder = 2*order # Order of the integration rule
    num_elements = mesh.GetNE() # Number of mesh elements
    gdof = fespace.GetNDofs() # Number of global dof
    ldof = fespace.GetFE(0).GetDof() # Number of local dof
    num_qp = mfem.IntRules.Get(fespace.GetFE(0).GetGeomType(), intorder).GetNPoints()

    quad_wts = np.zeros([num_elements, num_qp]) # Quad weights
    quad_pts = np.zeros([num_elements, num_qp, dim]) # Quad locations
    shape_pts = np.zeros([num_elements, num_qp, ldof]) # Basis at quad locations
    l2g_map = np.zeros([num_elements, ldof], dtype=np.int64) # Local-to-global index mappings

    for i in range(num_elements):
        
        # Extract the element and the local-to-global indices (stored for later)
        element = fespace.GetFE(i)
        vdofs = np.asarray(fespace.GetElementVDofs(i))
        l2g_map[i,:] = vdofs[:]

        Tr = mesh.GetElementTransformation(i)
        ir = mfem.IntRules.Get(element.GetGeomType(), intorder)
        shape = mfem.Vector(np.zeros([ldof]))
        
        for j in range(ir.GetNPoints()):

            # Get the integration point from the rule
            ip = ir.IntPoint(j)

            # Set an integration point in the element transformation
            Tr.SetIntPoint(ip)

            # Transform the reference integration point to a physical location
            transip = mfem.Vector(np.zeros([dim]))
            Tr.Transform(ip, transip)

            # Next, evaluate all the basis functions at this integration point
            element.CalcPhysShape(Tr, shape)

            # Compute the adjusted quadrature weight (volume factors)
            wt = ip.weight*Tr.Weight()

            # Store the quadrature data for later
            quad_wts[i,j] = wt
            quad_pts[i,j,:] = transip.GetDataArray() 
            shape_pts[i,j,:] = shape.GetDataArray()

    print("Finished with pre-processing... Running Parla tasks\n")

    #-----------------------------------------------------------------------------
    # Partitioning elements into tasks
    #-----------------------------------------------------------------------------

    # Specify a partition of the (global) list of elements
    num_blocks = args.blocks
    elements_per_block = num_elements // num_blocks # Block size
    leftover_blocks = num_elements % num_blocks
    
    # Adjust the number of elements if the block size doesn't divide the elements evenly
    element_block_sizes = elements_per_block*np.ones([num_blocks], dtype=np.int64)
    element_block_sizes[0:leftover_blocks] += 1
    
    print("Number of blocks: " + str(num_blocks))
    print("Number of left-over blocks: " + str(leftover_blocks))
    print("Elements per block:", element_block_sizes,"\n")
  
    # Create an array to hold the global data for the RHS
    element_global_array = np.zeros([gdof])

    # To use the blocking scheme, we'll
    # also use another set of arrays that
    # hold partial sums of this global array
    element_block_global_array = np.zeros([num_blocks, gdof])

    # Define the main task for parla
    @spawn(placement=cpu)
    async def LFIntegration_task():

        parla_time = 0.0 # Total time spent in the parla tasks (includes sleep)

        # Create the task space first
        ts = TaskSpace("LFTaskSpace")

        parla_times = np.zeros([num_trials]) # Store the times for statistics

        # Start tracking the GIL
        #gil_load.start()

        for trial_idx in range(num_trials):

            element_block_global_array[:,:] = 0.0 # Zero out the data from the previous run

            parla_start = time.perf_counter() # Only measure the time for tasks

            # Next we loop over each block which partitions the element indices
            for i in range(num_blocks):

                # It's really important that the taskids be unique
                # For this, we include another index for the trial
                @spawn(taskid=ts[trial_idx,i], vcus=load)
                def block_local_work():

                    # Need the offset for the element indices owned by this block
                    # This is the sum of all block sizes that came before it
                    s_idx = np.sum(element_block_sizes[:i])
                    e_idx = s_idx + element_block_sizes[i]

                    # Apply the Numba kernel to the block data
                    inner_quad_kernel(element_block_global_array[i,:], l2g_map, 
                                      quad_wts, quad_pts, shape_pts, 
                                      s_idx, e_idx)

            # Barrier for the task space associated with the loop over blocks
            await ts 

            # Perform the reduction across the blocks and store in the global_array
            element_global_array = np.sum(element_block_global_array, axis=0)

            parla_end = time.perf_counter()
            parla_times[trial_idx] = parla_end - parla_start

        # Stop tracking the GIL
        #gil_load.stop()

        # Print the GIL data
        #stats = gil_load.get()
        #print(gil_load.format(stats),"\n")

        print("PyMFEM times (min, max, avg) [s]: ", pymfem_times.min(),",",
                                                    pymfem_times.max(),",",
                                                    pymfem_times.mean(),
                                                    "\n",flush=True)

        print("Parla times (min, max, avg) [s] ", parla_times.min(),",",
                                                  parla_times.max(),",",
                                                  parla_times.mean(),
                                                  "\n",flush=True)

        #-----------------------------------------------------------------------------
        # End of the Parla test (the remainder checks for correctness)
        #-----------------------------------------------------------------------------

        # Define my own linear form for the RHS based on the above function
        # The 'FormLinearSystem' method, which performs additional manipulations
        # that simplify the RHS for the resulting linear system
        my_b = mfem.LinearForm(fespace)
        my_b.Assign(element_global_array)

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
        print("Size of linear system: " + str(A.Height()),"\n")

        # Build the linear system with my linear form
        my_B = mfem.Vector()
        a.FormLinearSystem(ess_tdof_list, x, my_b, A, X, my_B)

        # Covert the MFEM Vectors to Numpy arrays for norms
        B_array = B.GetDataArray()
        my_B_array = my_B.GetDataArray()

        # Compare the output of the two RHS vectors
        print("B =", B_array,"\n")
        print("my_B =", my_B_array,"\n")

        # Relative error against the MFEM output in the 2-norm
        rel_err_1 = np.linalg.norm(my_B_array - B_array, 1)/np.linalg.norm(B_array, 1)
        rel_err_2 = np.linalg.norm(my_B_array - B_array, 2)/np.linalg.norm(B_array, 2)
        rel_err_inf = np.linalg.norm(my_B_array - B_array, np.inf)/np.linalg.norm(B_array, np.inf)

        print("Relative error in the rhs (1-norm):", rel_err_1)
        print("Relative error in the rhs (2-norm):", rel_err_2)
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

    # First create the Parla context before spawning tasks
    with Parla():

        run(order=order,
            static_cond=static_cond,
            meshfile=meshfile,
            visualization=visualization,
            device=device,
            pa=pa)







