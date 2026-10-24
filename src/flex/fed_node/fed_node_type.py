from enum import Enum, auto

try:
    from enum import StrEnum
except ImportError:  # Python 3.10 compatibility
    class StrEnum(str, Enum):
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            return name.lower()

        __str__ = str.__str__

'''
' Node enumerate type
'''

class EFedNodeType(str, Enum):

    """
    " Unknown
    """
    unknown = "unknown"

    """
    " Server Node
    """
    server = "server"

    """
    " Edge Node
    """
    edge = "edge"
    
    """
    " Client node
    """
    client = "client"

    # Preserve StrEnum's string behavior while supporting Python 3.10.
    __str__ = str.__str__
    __format__ = str.__format__
