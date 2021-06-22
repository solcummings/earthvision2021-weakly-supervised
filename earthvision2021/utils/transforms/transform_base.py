import numpy as np
import typing


def _unimplemented_calls(self, *input: typing.Any) -> None:
    """
    Defines the computation performed at every call.
    Should be overridden by all subclasses.
    """
    raise NotImplementedError


class Transform:
    def __init__(self, p: float = 0.5, disable: bool = False):
        self.p = p
        self.disable = disable

    # forward (for normal calls) and backward (for reversing normal calls during tta) must
    # be implemented
    forward: typing.Callable[..., typing.Any] = _unimplemented_calls
    backward: typing.Callable[..., typing.Any] = _unimplemented_calls

    def _call_impl(self, array_list, reverse=False, *input, **kwargs):
        if np.random.rand() <= self.p and not self.disable:
            if reverse:
                array_list = self.backward(array_list, *input, **kwargs)
            else:
                array_list = self.forward(array_list, *input, **kwargs)
        return array_list

    __call__ : typing.Callable[..., typing.Any] = _call_impl

    def __repr__(self):
        return self.__class__.__name__ + '(p={}, disable={})'.format(self.p, self.disable)

