from __future__ import annotations
import json, csv, os


class CsvDataRecorder:
    """
    Record config(json) and dictionary data to CSV file
    """

    def __init__(self, filename: str):
        self.__filename: str = filename     # file name (private)
        self.__is_header_written: bool = False      # Indicate whether header is written (private)
        self.__csv_writer = None
        self.__csv_stream = None
        return

    # Indicate data recorder begin
    @property    
    def is_begin(self): return not (self.__csv_stream is None or self.__csv_stream.closed)

    def begin(self, head_config=None):
        """
        Begin write with header config json
        """

        if self.is_begin:
            return

        (path, _) = os.path.split(self.__filename)

        os.makedirs(path, exist_ok=True)
        self.__csv_stream = open(self.__filename, "a", newline = "", encoding = "utf-8")
        if head_config is not None:
            self.__csv_stream.write(f"Config,{json.dumps(head_config, ensure_ascii=False)}\n\n")
            self.__csv_stream.flush()
        return

    def end(self):
        """
        End write log
        """
        if not self.is_begin:
            return

        if self.__csv_stream is not None:
            self.__csv_stream.close()
        self.__csv_stream = None
        return

    def record(self, result_dict: dict) -> CsvDataRecorder:
        """
        Write record to CSV
        """
        if not self.is_begin:
            raise Exception("Must call begin() first")

        if not self.__is_header_written:
            self.__csv_writer = csv.DictWriter(
                self.__csv_stream, fieldnames=result_dict.keys(),
                extrasaction='ignore',       # silently drop keys not in header
                restval='',                   # fill missing columns with empty string
            )
            self.__csv_writer.writeheader()
            self.__is_header_written = True
            
        if self.__csv_writer is not None:
            self.__csv_writer.writerow(result_dict)
            self.__csv_stream.flush()
        return self

    def __del__(self):
        self.end()
