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
import argparse
import os
from os.path import expanduser, join

# Parse the command line options
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
parser.add_argument('-ngpus', default=1, type=int, 
                    help="How many gpus to use")

args = parser.parse_args()
print(args)

order = args.order
static_cond = args.static_condensation

meshfile = expanduser(
    join(os.path.dirname(__file__), '../PyMFEM/', 'data', args.mesh))

visualization = args.visualization
device = args.device
ngpus = args.ngpus
pa = args.partial_assembly

# Specify information about the devices
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')

if cuda_visible_devices is None:
    print("CUDA_VISIBLE_DEVICES is not set. Assuming 0-3")
    cuda_visible_devices = list(range(4))
else:
    cuda_visible_devices = cuda_visible_devices.strip().split(',')
    cuda_visible_devices = list(map(int, cuda_visible_devices))

gpus = cuda_visible_devices[:args.ngpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))

import numpy as np
import cupy as cp
import numba as nb
from numba import cuda
import time

# Import any of the MFEM Python modules
import mfem.ser as mfem 
from mfem.common.arg_parser import ArgParser

# Parla modules and decorators
# These should be imported after "CUDA_VISIBLE_DEVICES" is set
from parla import Parla, get_all_devices
from parla.cpu import cpu
from parla.cuda import gpu
from parla.array import copy
from parla.parray import asarray

# Imports for the spawn decorator and TaskSpace declaration
from parla.tasks import spawn, TaskSpace

# Check the hardware
import psutil

# Set an upper limit on the number of active basis functions on a given element
max_ldof = 10

# Why is there a weird type error here if I use a signature list?
#@cuda.jit([nb.void(nb.float64[:], nb.int64[:,:], nb.float64[:,:], nb.float64[:,:,:], nb.float64[:,:,:], nb.int64, nb.int64)])
@cuda.jit
def inner_quad_kernel(bg_array, l2g_map, quad_wts, quad_pts, shape_pts, s_idx, e_idx):
    """
    CUDA kernel that performs the quadrature evaluations over a collection of elements 
    inside a particular block.
    """
    # Get the number of dimensions
    dim = quad_pts.shape[2]

    # Get the number of quad pts
    num_qp = quad_wts.shape[1]

    # Get the number of active dof within each element (assumed to be the same)
    ldof = shape_pts.shape[2]

    # Initialize a local integral array to hold the integral over each element
    # Each thread will use this array as scratchpad memory
    local_integral = cuda.local.array(max_ldof, nb.float64)

    # Integration point array
    # Each thread will have its own copy
    ip = cuda.local.array(max_dim, nb.float64)

    # Assign one thread to each element of the mesh
    t_idx = nb.cuda.grid(1)

    # Since t_idx is in [0, block_size[i]), we need to shift its value
    # to a location in the global array
    j = t_idx + s_idx 

    if j < e_idx:

        local_integral[:ldof] = 0.0

        # Quadrature evaluation (with f = 1)
        for k in range(num_qp):
            quad_pt = quad_pts[j,k]

            for l in range(ldof):
                local_integral[l] += quad_wts[j,k]*shape_pts[j,k,l] 

        # Accumulate the local integral into the relevant entries of the block-wise global array
        # This needs to be done using atomics
        for l in range(ldof):
            cuda.atomic.add(bg_array, l2g_map[j,l], local_integral[l])

    return None


def run(order=1, static_cond=False,
        meshfile='', visualization=False,
        device='cpu', ngpus=ngpus, pa=False):
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
    ref_levels = int(np.floor(np.log(50000./mesh.GetNE())/np.log(2.)/dim))

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
    pymfem_start = time.perf_counter()

    b = mfem.LinearForm(fespace)
    one = mfem.ConstantCoefficient(1.0)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))
    b.Assemble()

    pymfem_end = time.perf_counter()

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

    # Data transfer: Move the quadrature data to the current device
    quad_wts_gpu = cp.asarray(quad_wts)
    quad_pts_gpu = cp.asarray(quad_pts)
    shape_pts_gpu = cp.asarray(shape_pts) 
    l2g_map_gpu = cp.asarray(l2g_map) 

    #-----------------------------------------------------------------------------
    # Partitioning elements into tasks
    #-----------------------------------------------------------------------------

    # Specify a partition of the (global) list of elements
    num_blocks = 2 # How many blocks do you want to use?! 
    elements_per_block = num_elements // num_blocks # Block size
    leftover_blocks = num_elements % num_blocks
    
    # Adjust the number of elements if the block size doesn't divide the elements evenly
    block_sizes = elements_per_block*np.ones([num_blocks], dtype=np.int64)
    block_sizes[0:leftover_blocks] += 1

    # Precompute the offsets for the element blocks
    # This tells us the starting index for each block of elements
    block_offsets = np.zeros([num_blocks], dtype=np.int64)
    for i in range(num_blocks):
        block_offsets[i] = np.sum(block_sizes[:i])

    # Copy the block sizes and offsets to the device
    block_sizes_gpu = cp.asarray(block_sizes)
    block_offsets_gpu = cp.asarray(block_offsets)

    print("Number of blocks: " + str(num_blocks))
    print("Number of left-over blocks: " + str(leftover_blocks))
    print("Elements per block:", block_sizes,"\n")

    # To use the blocking scheme, we'll
    # also use another set of arrays that
    # hold partial sums of this global array
    block_global_array_gpu = cp.zeros([num_blocks, gdof])

    parla_start = time.perf_counter()

    # Define the main task for parla
    @spawn(placement=cpu)
    async def LFIntegration_task():

        # Create the task space first
        ts = TaskSpace("LFTaskSpace")

        # Next we loop over each block which partitions the element indices
        for i in range(num_blocks):

            @spawn(taskid=ts[i], placement=gpu[0])
            def block_local_work():

                # Start and end indices for the elements owned by this block
                #s_idx = block_offsets_gpu[i] # Why do these commands produce weird 0-D arrays of int64?
                #e_idx = s_idx + block_sizes_gpu[i]
                s_idx = np.sum(block_sizes[:i])
                e_idx = s_idx + block_sizes[i]

                # blocks_per_grid = ceil(block_sizes[i]/threads_per_block)
                threads_per_block = 256 # Need to eventually increase the elements. Otherewise we have low occupancy
                blocks_per_grid = block_sizes[i] // threads_per_block + 1

                # Apply the Numba kernel to the block data
                inner_quad_kernel[blocks_per_grid, threads_per_block](block_global_array_gpu[i,:], l2g_map_gpu, quad_wts_gpu, quad_pts_gpu, shape_pts_gpu, s_idx, e_idx)

                cuda.synchronize()

        # Barrier for the task space associated with the loop over blocks
        await ts

        # Perform the reduction across the blocks and store in the global_array
        # This is done on the device
        global_array_gpu = cp.sum(block_global_array_gpu, axis=0)

        parla_end = time.perf_counter()

        # Transfer the global array back to the host
        global_array = cp.asnumpy(global_array_gpu)

        print("PyMFEM time (s): ", pymfem_end - pymfem_start, "\n",flush=True)
        print("Parla time (s): ", parla_end - parla_start, "\n",flush=True)

        #-----------------------------------------------------------------------------
        # End of the Parla test
        #-----------------------------------------------------------------------------

        # Define my own linear form for the RHS based on the above function
        # The 'FormLinearSystem' method, which performs additional manipulations
        # that simplify the RHS for the resulting linear system
        my_b = mfem.LinearForm(fespace)
        my_b.Assign(global_array)

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
            ngpus=ngpus,
            pa=pa)







