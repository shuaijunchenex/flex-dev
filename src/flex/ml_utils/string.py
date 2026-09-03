from __future__ import annotations


class String:
    """
    Static class for string operation
    """

    @staticmethod
    def is_none_or_empty(string) -> bool:
        """
        Test string is None or len == 0 or is space
        """
        if string is None:      # string is None
            return True
        if len(string) == 0:    # string len() == 0
            return True
        if string.isspace():    # isspace()
            return True
        return False
