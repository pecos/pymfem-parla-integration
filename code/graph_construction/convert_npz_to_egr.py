"""
    Code to convert a collection of files to the ECL graph file format.

    Specifically, it calls an executable generated with the conversion tools made
    availble on the webpage: https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/index.html

    For example, a file in Matrix Market format can be converted using the following tools:
        1) https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/ECLgraph.h
        2) https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/mm2ecl.cpp

    This executable takes a collection of files in a directory and 
    converts them to a .ecl files for subsequent processing. Basically, we will pass
    the string containing the directory of the files to be processed and the scipt will
    produce a new directory with files in the proper format.
"""

import argparse
import os
from os.path import join

import scipy.sparse as sparse
import scipy.io as io

# Parse the command line data
parser = argparse.ArgumentParser(description="Converts a collection of sparse matrix files to ECL files.")
parser.add_argument("-src", default="", type=str, help="Source directory containing files to be converted.")
parser.add_argument("-exe", default="", type=str, help="Executable that performs a single file conversion.")
parser.add_argument("-root", default=".", type=str, help="Root directory where the output will be generated.")

args = parser.parse_args()

print("\nOptions used:\n")

for arg in vars(args):
    print(arg, "=", getattr(args, arg))

print("\n")

src = args.src
exe = args.exe
root = args.root

def convert_npz_to_mtx(fname):
    """
    Helper function that converts files in a directory "src" from
    npz to mtx formatting.
    """

    # First load the sparse matrix dataset
    data = sparse.load_npz(fname)

    # Write it to matrix market format
    fname_mtx = os.path.splitext(fname)[0] + ".mtx"
    io.mmwrite(fname_mtx, data, comment='', field="integer")
    
    return fname_mtx

def convert_npz_to_egr(src, exe, root):
    """
    Processes the files contained in the directory "src",
    converting them to ECL format with the executable "exe".

    This function can be called on an entire directory at a time
    and its output will be stored under root.
    """

    assert os.path.exists(exe), "Error: Executable does not exist."

    # Using the source directory "src", create the destination "dst" 
    # where the converted files will be stored
    # We should remove any parts of source which overlap with the root
    dst = root + "/" + "ECL-" + src.replace(root + "/","")

    # Remove any pre-existing destination directory if it already exists
    os.system(" ".join(["rm", "-rf", dst]))
    os.makedirs(dst)

    # Extract a list of all the files inside the directory src
    # These are the files that will need to be processed
    src_files = os.listdir(src)

    # Next, iterate through this list of files, converting and storing in "dst"
    for f in src_files:

        # The executable is located in the same directory as src and dst
        # so we need to make sure that the full paths are specified
        input_file_path = os.path.join(src, f)

        # Convert the file from .npz format to .mtx
        # This is done inside the src directory
        input_file_path_mtx = convert_npz_to_mtx(input_file_path)

        # Strip the .npz extension from the input file and change it to egr
        output_file = os.path.splitext(f)[0] + ".egr"
        output_file_path = os.path.join(dst, output_file)

        # Call the C++ executable that converts .mtx to .egr
        os.system(" ".join([exe, input_file_path_mtx, output_file_path]))    

        # Remove the mtx data file to keep storage reasonable
        os.system(" ".join(["rm", input_file_path_mtx]))

        print("\n")

    return None


if __name__ == "__main__":

    convert_npz_to_egr(src, exe, root)






