from enum import StrEnum, auto

'''
' Node enumerate type
'''

class EFedNodeType(StrEnum):    

    """
    " Unknown
    """
    unknown = auto()

    """
    " Server Node
    """
    server = auto()

    """
    " Edge Node
    """
    edge = auto()
    
    """
    " Client node
    """
    client = auto()
