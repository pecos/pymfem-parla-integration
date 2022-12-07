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


# Parse the command line data
parser = argparse.ArgumentParser(description="Converts a collection of sparse matrix files to ECL files.")
parser.add_argument("-src", default="", type=str, help="Source directory containing files to be converted.")
parser.add_argument("-exe", default="", type=str, help="Executable that performs a single file conversion.")

args = parser.parse_args()

print("\nOptions used:\n")

for arg in vars(args):
    print(arg, "=", getattr(args, arg))

print("\n")

src = args.src
exe = args.exe


def convert_files(src, exe):
    """
    Processes the files contained in the directory "src",
    converting them to ECL format with the executable "exe".

    This function can be called on an entire directory at a time.
    """

    assert os.path.exists(exe), "Error: Executable does not exist."

    # Using the source directory "src", create the destination "dst" 
    # where the converted files will be stored
    dst = "ECL-" + src

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

        # Strip the extension from the input file and change it to ecl
        # See: https://stackoverflow.com/questions/3548673/how-can-i-replace-or-strip-an-extension-from-a-filename-in-python
        output_file = os.path.splitext(f)[0] + ".ecl"
        output_file_path = os.path.join(dst, output_file)

        # Call the executable on the input and output files
        os.system(" ".join([exe, input_file_path, output_file_path]))    

        print("\n")

    return None


if __name__ == "__main__":

    convert_files(src, exe)






