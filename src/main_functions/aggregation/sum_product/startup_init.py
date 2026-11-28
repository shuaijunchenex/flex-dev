import sys, os

def startup_init_path(startup_file_path, search_paths: list = ["..", "../.."]):
    """
    Initial execute startup path.

    Notice: Copy this file to program source code folder, then import it, and
            add follow code into startup file, like main.py

    ``` pyhton

    from __future__ import annotations

    # Init startup path, change current path to test py file folder 
    #-----------------------------------------------------------------
    import os
    from startup_init import startup_init_path
    startup_init_path(os.path.dirname(os.path.abspath(__file__)))
    #-----------------------------------------------------------------

    Args:
        startup_file_path: program run starup path
        search_paths: append search paths, default append '..' and '../..' folder

    ```
    """

    # For Visual Studio startup path at <project folder>\test
    # like D:\Project.ML\neat_torch_ml\src-libs\usyd-learning\src\test
    # You must add search path as follows
    for path in search_paths:
        sys.path.append(path)

    # For VS Code startup path at <project folder> where VS open "src" folder as workspace(via File -> Open Folder)
    # like D:\Project.ML\neat_torch_ml\src-libs\usyd-learning\src
    # You must change running path to test file folder
    os.chdir(startup_file_path)
    
    # print(f"\nStartup path: {os.getcwd()}\n")
    return