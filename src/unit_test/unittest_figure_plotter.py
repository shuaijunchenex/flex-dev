from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import FigurePlotter, ConfigLoader, console


def main():

    files_dict={
        "full":"F:\\Torchly\\results\\full_model_cx_20250611_204115_9728ab4e.csv",
        "svd_ab":"F:\\Torchly\\results\\svd_ab_cx_20250611_173038_9728ab4e.csv",
        "init_ab":"F:\\Torchly\\results\\w_init_ab_cx_20250611_180809_9728ab4e.csv",
        "w_svd_ab": "F:\\Torchly\\results\\w_svd_ab_cx_20250612_192843_9728ab4e.csv"}

    FigurePlotter.plot_csv_files(files_dict=files_dict, x_column="round", y_column="accuracy") 
    return

if __name__ == "__main__":
    main()
