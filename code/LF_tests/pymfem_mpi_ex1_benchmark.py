'''
   MFEM example 1p

   See c++ version in the MFEM library for more detail 
'''
import sys
import os
from os.path import expanduser, join
import numpy as np

from mpi4py import MPI

from mfem.common.arg_parser import ArgParser
from mfem import path
import mfem.par as mfem

def_meshfile = expanduser(join(path, 'data', 'star.mesh'))

num_proc = MPI.COMM_WORLD.size
myid = MPI.COMM_WORLD.rank


def run(order=1, static_cond=False,
        meshfile=def_meshfile,
        visualization=False,
        device='cpu',
        pa=False,
        num_trials=1):

    algebraic_ceed = False

    device = mfem.Device(device)
    if myid == 0:
        device.Print()

    mesh = mfem.Mesh(meshfile, 1, 1)
    dim = mesh.Dimension()

    #ref_levels = int(np.floor(np.log(10000./mesh.GetNE())/np.log(2.)/dim))
    
    # Use a "really" fine mesh
    ref_levels = 9

    for x in range(ref_levels):
        mesh.UniformRefinement()
    mesh.ReorientTetMesh()
    pmesh = mfem.ParMesh(MPI.COMM_WORLD, mesh)
    del mesh

    #par_ref_levels = 2
    par_ref_levels = 0 # Don't refine the mesh anymore...

    for l in range(par_ref_levels):
        pmesh.UniformRefinement()

    if order > 0:
        fec = mfem.H1_FECollection(order, dim)
    elif mesh.GetNodes():
        fec = mesh.GetNodes().OwnFEC()
        print("Using isoparametric FEs: " + str(fec.Name()))
    else:
        order = 1
        fec = mfem.H1_FECollection(order, dim)

    fespace = mfem.ParFiniteElementSpace(pmesh, fec)
    fe_size = fespace.GlobalTrueVSize()

    if (myid == 0):
        print('Number of finite element unknowns: ' + str(fe_size))

    ess_tdof_list = mfem.intArray()
    if pmesh.bdr_attributes.Size() > 0:
        ess_bdr = mfem.intArray(pmesh.bdr_attributes.Max())
        ess_bdr.Assign(1)
        fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list)

    # Set up the linear form b(.) which corresponds to the right-hand side of
    # the FEM linear system, which in this case is (1,phi_i) where phi_i are
    # the basis functions in the finite element fespace.
    b = mfem.ParLinearForm(fespace)
    one = mfem.ConstantCoefficient(1.0)
    b.AddDomainIntegrator(mfem.DomainLFIntegrator(one))

    if myid == 0:
        pymfem_times = np.zeros([num_trials]) # Store the times for statistics 

    for trial_idx in range(num_trials):

        # Reset the value of the linear form
        # This is needed for correctness
        b.Assign(0.0)

        MPI.COMM_WORLD.Barrier()
        pymfem_start = MPI.Wtime()

        b.Assemble()

        MPI.COMM_WORLD.Barrier()
        pymfem_end = MPI.Wtime()

        # Only rank 0 stores the times
        if myid == 0:
            pymfem_times[trial_idx] = pymfem_end - pymfem_start

    # Print the statistics to the console
    # Only rank 0 does this
    if myid == 0:
        print("PyMFEM times (min, max, avg) [s]: ", pymfem_times.min(),",",
                                                    pymfem_times.max(),",", 
                                                    pymfem_times.mean(),
                                                    "\n",flush=True)

    x = mfem.ParGridFunction(fespace)
    x.Assign(0.0)

    a = mfem.ParBilinearForm(fespace)
    if pa:
        a.SetAssemblyLevel(mfem.AssemblyLevel_PARTIAL)
        
    a.AddDomainIntegrator(mfem.DiffusionIntegrator(one))

    if static_cond:
        a.EnableStaticCondensation()

    a.Assemble()

    A = mfem.OperatorPtr()
    B = mfem.Vector()
    X = mfem.Vector()
    a.FormLinearSystem(ess_tdof_list, x, b, A, X, B)

    # Don't run the solver here...

    #if pa:
    #    if mfem.UsesTensorBasis(fespace):
    #        if algebraic_ceed:
    #            assert False, "not supported"
                #prec = new ceed::AlgebraicSolver(a, ess_tdof_list);
    #        else:
    #            prec = mfem.OperatorJacobiSmoother(a, ess_tdof_list)
    #else:
    #    prec = mfem.HypreBoomerAMG()
        
    #cg = mfem.CGSolver(MPI.COMM_WORLD)
    #cg.SetRelTol(1e-12)
    #cg.SetMaxIter(200)
    #cg.SetPrintLevel(1)
    #cg.SetPreconditioner(prec)
    #cg.SetOperator(A.Ptr())
    #cg.Mult(B, X)

    #a.RecoverFEMSolution(X, b, x)

    #smyid = '{:0>6d}'.format(myid)
    #mesh_name = "mesh."+smyid
    #sol_name = "sol."+smyid

    #pmesh.Print(mesh_name, 8)
    #x.Save(sol_name, 8)

if __name__ == "__main__":
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
    parser.add_argument("-trials", "--trials",
                        default=1, type=int,
                        help="Number of trials used in the timed code regions.")

    args = parser.parse_args()

    if myid == 0:
        parser.print_options(args)

    order = args.order
    static_cond = args.static_condensation
    meshfile = expanduser(
        join(os.path.dirname(__file__), '../PyMFEM/', 'data', args.mesh))
    visualization = args.visualization
    device = args.device
    pa = args.partial_assembly
    num_trials = args.trials 


    run(order=order,
        static_cond=static_cond,
        meshfile=meshfile,
        visualization=visualization,
        device=device,
        pa=pa,
        num_trials=num_trials)
