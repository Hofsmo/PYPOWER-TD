"""
The main module for PyPiW
"""


class PyPiW(object):
    """
    Class responsible for performing the identification.
    """
    def __init__(self, in_data, out_data, time, alg):
        """
        Constructor of PyPiW
        Input:
            in_data: Array of in data
            out_data: Array of response data
            ts: Time step
            alg: Algorithm object that should an identify method.
                Algorithms can be found in algorithms.py
        """
        self.in_data = in_data
        self.out_data = out_data
        self.time = time
        self.alg = alg

    def identify(self):
        """
        Function to perform the identification
        """
        self.alg.identify()
