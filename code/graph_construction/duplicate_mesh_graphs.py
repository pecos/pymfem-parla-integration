"""
    Code to read in directed graphs associated with 
    discrete ordinates methods on unstructured FEM meshes and duplicate them.

    This executable takes a collection of files in a directory (assumed to be in .npz format) and 
    reads them into sparse graph adjacency matrices. We duplicate the graphs by shifting the vertices and
    edges in the graph. The duplicates do not connect to each other in any way. The only reason we
    are doing this is that we need to generate the larger graphs for the SC paper.
"""

import argparse
import os
from os.path import join

import numpy as np
import scipy.io as io
import scipy.sparse as sparse

# Parse the command line data
parser = argparse.ArgumentParser(description="Duplicates the directed graphs using adjacency matrices.")
parser.add_argument("-src", default=None, type=str, help="Source directory containing files to be analyzed. Assumed to be in .npz format.")
parser.add_argument("-dst", default=None, type=str, help="Destination that will hold the duplicated files. Assumed to be in .npz format.")
parser.add_argument("-num_duplicates", default=10, type=int, help="Number of duplicates to make of each graph.")

args = parser.parse_args()

print("\nOptions used:\n")

for arg in vars(args):
    print(arg, "=", getattr(args, arg))

print("\n")


src = args.src
dst = args.dst
num_duplicates = args.num_duplicates


def duplicate_graph(coo_data, num_duplicates):
    """
    Replicates a directed graph stored in a coo format
    'num_duplicates' times.

    A graph is duplicated by shifting the vertices and edges.
    Each duplicate is assumed to not be associated with any other graph.
    """
    
    # Extract the attributes of the input coo matrix we want to duplicate
    shape = coo_data.shape
    nnz = coo_data.nnz
    val = coo_data.data
    row = coo_data.row
    col = coo_data.col
    
    # Create the new row, col, and data arrays associated with the duplicated graph
    # We create the duplicated graph by specifying the data in (row,col,val) format
    row_dup = np.zeros((nnz*num_duplicates), dtype=np.int64)
    col_dup = np.zeros((nnz*num_duplicates), dtype=np.int64)
    val_dup = np.zeros((nnz*num_duplicates), dtype=np.int64)

    # Now fill the duplicate data set by applying the appropriate index shifts
    for i in range(num_duplicates):
        for j in range(nnz):

            row_dup[i*nnz + j] = row[j]
            col_dup[i*nnz + j] = col[j]
            val_dup[i*nnz + j] = val[j]

    # Now take the (row,col,val) data and build the coo matrix
    duplicate_coo_data = sparse.coo_matrix((val_dup,(row_dup, col_dup)), dtype=np.int64)

    return duplicate_coo_data


def process_graphs(src, dst, num_duplicates):
    """
    Duplicates a collection of sparse matrix files in a selected source directory.
    
    This function can be called on an entire directory at a time.
    """

    # Extract a list of all the files inside the directory src
    # These are the files that will need to be processed
    src_files = os.listdir(src)

    # Remove any pre-existing destination directory if it already exists and rebuild
    os.system(" ".join(["rm", "-rf", dst]))
    os.makedirs(dst)

    print("src files:", src_files,"\n")

    for f in src_files:

        # The executable is located in the same directory as src and dst
        # so we need to make sure that the full paths are specified
        input_file_path = os.path.join(src, f)

        # 1) Read the data into a sparse matrix format
        # We covert this to a coordinate format for simplicity
        adj_data = sparse.load_npz(input_file_path)
        adj_data_coo = adj_data.tocoo(copy=False)

        # 2) Create a new sparse matrix describing the duplicated graph
        duplicate_adj_data_coo = duplicate_graph(adj_data_coo, num_duplicates)

        # 3) Save the duplicate data to a .npz file
        # We add the tag 'duplicate' to indicate that the data was duplicated
        fname_out = os.path.splitext(f)[0] + "-duplicate"
        output_file_path = os.path.join(dst, fname_out)

        print("Input path:", input_file_path)
        print("Output path:", output_file_path, "\n")

        sparse.save_npz(output_file_path, duplicate_adj_data_coo, compressed=True)

    return None


if __name__ == "__main__":

    assert src is not None, "Error: No source directory was specified."
    assert dst is not None, "Error: No destination directory was specified."
    assert src is not dst, "Error: Source and destination should be distinct."
    assert num_duplicates >= 0, "Error: The number of duplicates should be >= 0."

    process_graphs(src, dst, num_duplicates)


