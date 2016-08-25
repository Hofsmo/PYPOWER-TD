from abc import ABCMeta, abstractmethod
import six


@six.add_metaclass(ABCMeta)
class SystemBase():
    """
    Base class for system representations
    """

    @abstractmethod
    def time_response(self):
        """
        This method calculates the time response of the system
        """
        pass


class Tf(SystemBase):
    """
    Class for transfer function representation
    """
    def __init__(self, sys):
        self.sys = sys

    def time_response(self):
        pass
