import numpy as np
from numpy.random import default_rng

rng = default_rng()

class DiscreteVariable:
    def __init__(self, low=np.NINF, high=np.Inf, value=None):
        self.low = low
        self.high = high
        self.value = rng.integers(low=low, high=high, endpoint=True) if value is None else value

        if self.value > high:
            self.value = high
        elif self.value < low:
            self.low = low

    def draw_random(self, low=None, high=None):

        if low is None:
            low = -np.abs(self.high - self.low)
        if high is None:
            high = np.abs(self.high - self.low)

        return DiscreteVariable(value=rng.uniform(low=low, high=high))

    def __add__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return DiscreteVariable(low=self.low, high=self.high, value=round(self.value + other))
        else:
            return DiscreteVariable(low=self.low, high=self.high, value=np.round(self.value + other.value))
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return DiscreteVariable(low=self.low, high=self.high, value=round(self.value - other))
        else:
            return DiscreteVariable(low=self.low, high=self.high, value=round(self.value - other.value))
    __rsub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return DiscreteVariable(low=self.low, high=self.high, value=round(self.value * other))
        else:
            return DiscreteVariable(low=self.low, high=self.high, value=round(self.value * other.value))
    __rmul__ = __mul__

    def __pow__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return DiscreteVariable(low=self.low, high=self.high, value=round(self.value ** other))
        else:
            return DiscreteVariable(low=self.low, high=self.high, value=round(self.value ** other.value))
    __rpow__ = __pow__

    def __truediv__(self, other):
        # return self.value // other.value
        return DiscreteVariable(self.value // other.value)
    __rtruediv__ = __truediv__


class ContinuousVariable:
    def __init__(self, low=np.NINF, high=np.Inf, value=None):
        self.low = low
        self.high = high
        self.value = rng.uniform(low=low, high=high) if value is None else value

        if self.value > high:
            self.value = high
        elif self.value < low:
            self.value = low

    def draw_random(self, low=None, high=None):

        if low is None:
            low = -np.abs(self.high - self.low)
        if high is None:
            high = np.abs(self.high - self.low)

        return ContinuousVariable(value=rng.uniform(low=low, high=high))

    def __add__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return ContinuousVariable(low=self.low, high=self.high, value=self.value + other)
        else:
            return ContinuousVariable(low=self.low, high=self.high, value=self.value + other.value)
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return ContinuousVariable(low=self.low, high=self.high, value=self.value - other)
        else:
            return ContinuousVariable(low=self.low, high=self.high, value=self.value - other.value)
    __rsub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return ContinuousVariable(low=self.low, high=self.high, value=self.value * other)
        else:
            return ContinuousVariable(low=self.low, high=self.high, value=self.value * other.value)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return ContinuousVariable(low=self.low, high=self.high, value=self.value / other)
        else:
            return ContinuousVariable(low=self.low, high=self.high, value=self.value / other.value)
    __rtruediv__ = __truediv__

    def __pow__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return ContinuousVariable(low=self.low, high=self.high, value=self.value ** other)
        else:
            return ContinuousVariable(low=self.low, high=self.high, value=self.value ** other.value)
    __rpow__ = __pow__

    def __lt__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.value < other
        else:
            return self.value < other.value

    def __le__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.value <= other
        else:
            return self.value <= other.value

    def __gt__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.value > other
        else:
            return self.value > other.value

    def __ge__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.value >= other
        else:
            return self.value >= other.value

    def __eq__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.value == other
        else:
            return self.value == other.value

    def __ne__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            return self.value != other
        else:
            return self.value != other.value


class Constant:
    def __init__(self, value):
        self.value = value
        self.low = value
        self.high = value

    def draw_random(self, low=None, high=None):

        if low is None:
            low = -np.abs(self.high - self.low)
        if high is None:
            high = np.abs(self.high - self.low)

        return Constant(value=rng.uniform(low=low, high=high))

    def __add__(self, other):
        return Constant(self.value)
    __radd__ = __add__

    def __sub__(self, other):
        return Constant(self.value)
    __rsub__ = __sub__

    def __mul__(self, other):
        return Constant(self.value)
    __rmul__ = __mul__

    def __truediv__(self, other):
        return Constant(self.value)
    __rtruediv__ = __truediv__

    def __pow__(self, power, modulo=None):
        return Constant(self.value)
    __rpow__ = __pow__

    def __call__(self):
        return Constant(self.value)