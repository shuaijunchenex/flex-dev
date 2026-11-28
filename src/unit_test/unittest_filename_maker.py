from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import FileNameMaker, console


def test_fileName_maker():
    console.out("Test FileNameMaker class:")
    console.out("------------- Begin ---------------")
    
    fname = "abc"
    console.out(f"Make file name: {fname}")
    file_names = FileNameMaker.make(fname)

    console.out("Result:")
    console.info("  name: " + file_names.name)          #name - only origin name
    console.info("  path: " + file_names.path)          #path - only path
    console.info("  file name: " + file_names.filename)      #filename - only file name
    console.info("  full name: " + file_names.fullname)      #fullname - combine of path and file name
    console.out("------------- End ----------------\n")

    fname = "deg"
    console.out(f"Make file name with args: {fname}")
    console.out("------------- Begin ---------------")

    file_names_1 = FileNameMaker.with_path("./.results-1/").with_prefix("prefix").make(fname)
    console.info("  name: " + file_names_1.name)
    console.info("  path: " + file_names_1.path)
    console.info("  file name: " + file_names_1.filename)
    console.info("  full name: " + file_names_1.fullname)
    console.out("------------- End ----------------\n")
    return


def main():
    test_fileName_maker()
    return


if __name__ == "__main__":
    main()
