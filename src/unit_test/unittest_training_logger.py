from __future__ import annotations


# Init startup path, change current path to test py file folder 
#-----------------------------------------------------------------
import os
from startup_init import startup_init_path
startup_init_path(os.path.dirname(os.path.abspath(__file__)))
#-----------------------------------------------------------------

from flex.ml_utils import ConfigLoader, CsvDataRecorder,FileNameMaker,TrainingLogger, console


header_data = {"round": "10", "accuracy" : "20", "precision": "30", "recall" : "40", "f1_score" : "50"}
log_data = {"round": "1", "accuracy" : "2", "precision": "3", "recall" : "4", "f1_score" : "5"}

def test_csv_data_recorder():
    #log file name
    file_names = FileNameMaker.make("csv-data-record")
    print("Generated file name: " + file_names.fullname)
     
    #training logger
    console.info(f"Create csv data recorder object")
    csv_recorder = CsvDataRecorder(file_names.fullname)

    try:
        #begin log
        console.info("Begin csv record...")
        csv_recorder.begin(header_data)

        #record log
        console.info("Write csv record...")
        for i in range(0, 9):
            csv_recorder.record(log_data)

    except Exception as e:
        console.error(f"Something csv ERROR {e}")

    finally:
        #log end
        csv_recorder.end()
        console.info("End csv record")

    return


def test_training_logger():
    yaml_file_name = './test_data/fed_runner_template.yaml'
    console.info(f"Load train log yaml file {yaml_file_name}...")
    yaml = ConfigLoader.load(yaml_file_name)

    #training logger
    console.info(f"Create training logger object")
    training_logger = TrainingLogger(yaml["logger"])

    try:
        #begin log
        console.info("Begin training log...")
        training_logger.begin()

        #record log
        console.info("Write training log...")
        for i in range(0, 9):
            training_logger.record(log_data)

    except Exception as e:
        console.error(f"Something training ERROR {e}")

    finally:
        #log end
        training_logger.end()
        console.info("End training log")

    return


def main():
    print("test CsvDataRecorder")
    console.out("------------- Begin ---------------")
    test_csv_data_recorder()
    console.out("------------- End -----------------\n")

    print("test TrainingLogger")
    console.out("------------- Begin ---------------")
    test_training_logger()
    console.out("------------- End -----------------")
    return

if __name__ == "__main__":
    main()
