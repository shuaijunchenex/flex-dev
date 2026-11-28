from __future__ import annotations

from collections import namedtuple
import os

"""
Declare FileName part named tuple
"""
FileNameParts = namedtuple("FileNameParts", ["fullname", "folder", "file", "name", "extension"])


class FileNameHelper:
    """
    Get file's part: "fullname", "folder", "file", "name", "extension"
    """

    @staticmethod
    def split(file_path):
        """
        Split path into parts, include "fullname", "folder", "file", "name", "extension"
        """
        path = os.path.dirname(file_path)
        file = os.path.basename(file_path)
        name, extension = os.path.splitext(file)
        return FileNameParts(file_path, path, file, name, extension)

    @staticmethod
    def combine(*paths):
        """
        Combine paths use os.path.join method
        """
        return os.path.join(*paths).replace("\\", "/")

    @staticmethod
    def exists(file):
        """
        Check if file exists
        """
        return os.path.exists(file)