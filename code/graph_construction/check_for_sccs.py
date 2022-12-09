"""
    Code to check for strongly connected components (SCCs) in directed graphs associated with 
    discrete ordinates methods on unstructured FEM meshes.

    This executable takes a collection of files in a directory (assumed to be in .mtx format) and 
    converts them into sparse graph adjacency matrices for subsequent processing with NetworkX tools.
"""

import argparse
import os
from os.path import join

import networkx as nx
import scipy.io as io
import scipy.sparse as sp

# Parse the command line data
parser = argparse.ArgumentParser(description="Checks for strongly connected components (SCCs) in directed graphs using adjacency matrices.")
parser.add_argument("-src_dir", default=None, type=str, help="Source directory containing files to be analyzed.")

args = parser.parse_args()

print("\nOptions used:\n")

for arg in vars(args):
    print(arg, "=", getattr(args, arg))

print("\n")

src_dir = args.src_dir


def mtx_to_sparse(filename):
    """
    Converts a file from Matrix-Market format into a Scipy COO structure.
    """

    # Check that that extension is correct
    file_ext = os.path.splitext(filename)[1]
    assert file_ext == ".mtx", "Error: Check the extension for the source files."

    # To ensure that the matrix is in the format needed by NetworkX, we convert
    # the read output (either dense matrix or coo matrix) to a coo format
    return sp.coo_matrix(io.mmread(filename))


def check_graphs_for_sccs(src_dir):
    """
    Checks all sparse matrix files in a selected source directory for 
    the number of SCCs.
    
    This function can be called on an entire directory at a time.
    """

    # Extract a list of all the files inside the directory src
    # These are the files that will need to be processed
    src_files = os.listdir(src_dir)

    # Next, iterate through this list of files, converting and processing for SCCs
    for f in src_files:

        # The executable is located in the same directory as src and dst
        # so we need to make sure that the full paths are specified
        input_file_path = os.path.join(src_dir, f)

        # Convert the file (specified by its path) to a sparse format
        adj_data = mtx_to_sparse(input_file_path)

        # Construct the directed graph structure using NetworkX
        # We have to explicitly mention the graph type
        # as being directed to set the class constructor internally
        G = nx.from_scipy_sparse_matrix(adj_data, create_using=nx.DiGraph)

        # Determine the subsets of nodes that form SCCs (if they exist)
        # We sort them such that the largest subsets are placed first
        # This step requires G to be a directed graph, or it emits an exception
        scc_list = [scc for scc in sorted(nx.strongly_connected_components(G), key=len, reverse=True)]
    
        # Extract the list of sets of sccs that contain more than one node
        # These are the nodes that contain a cycle
        nontrivial_scc_list = [item for item in scc_list if len(item) > 1 ]

        print("Graph associated with file: %s"%f)
        print("Total number of SCCs found: %d"%len(scc_list))
        print("Number of non-trivial SCCs found: %d"%len(nontrivial_scc_list))

    return None


if __name__ == "__main__":

    assert src_dir is not None, "Error: No source directory was specified."

    check_graphs_for_sccs(src_dir)







