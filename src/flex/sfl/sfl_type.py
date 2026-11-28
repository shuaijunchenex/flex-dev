from enum import Enum

'''
' How the rear (server side) model state evolves across client visits
'''

class ERearMode(str, Enum):

    shared = "shared"
    private_adapter = "private_adapter"
    round_average = "round_average"

    __str__ = str.__str__
    __format__ = str.__format__


'''
' How the rear optimizer slots are kept across client visits
'''

class ERearHistory(str, Enum):

    shared = "shared"
    per_client = "per_client"
    reset_on_switch = "reset_on_switch"

    __str__ = str.__str__
    __format__ = str.__format__


'''
' How the client front weights are synchronized after a round
'''

class EFrontSyncMethod(str, Enum):

    local = "local"
    fedavg = "fedavg"
    factor_avg = "factor_avg"
    rbla_plus = "rbla_plus"

    __str__ = str.__str__
    __format__ = str.__format__
