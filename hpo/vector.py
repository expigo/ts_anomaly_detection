import numpy as np
from numpy.random import default_rng

rng = default_rng(42)


class DiscreteVariable:
    def __init__(self, low=np.NINF, high=np.Inf, value=None):
        self.low = low
        self.high = high
        self._value = int(rng.integers(low=low, high=high, endpoint=True)) if value is None else int(value)
        self.__bound_check__()

    def __bound_check__(self):
        if self.value > self.high:
            self.value = self.high
        elif self.value < self.low:
            self.value = self.low

    def draw_random(self, low=None, high=None):
        if low is None:
            low = -np.abs(self.high - self.low)
        if high is None:
            high = np.abs(self.high - self.low)

        return DiscreteVariable(value=round(rng.uniform(low=low, high=high)))

    @property
    def value(self):
        integer = int(self._value)
        return integer

    @value.setter
    def value(self, value):
        self._value = int(value)


    def __add__(self, other):
        value = None
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            value = round(self.value + other)
        else:
            value = round(self.value + other.value)
        # self.__bound_check__()
        # return self
        return DiscreteVariable(low=self.low, high=self.high, value=value)
    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            self.value = round(self.value - other)
        else:
            self.value = round(self.value - other.value)
        self.__bound_check__()
        return self

    __rsub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            self.value = round(self.value * other)
        else:
            self.value = round(self.value * other.value)
        self.__bound_check__()
        return self

    __rmul__ = __mul__

    def __pow__(self, other):
        value = None
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            value = round(self.value ** other)
        else:
            value = round(self.value ** other.value)
        # self.__bound_check__()
        # return self
        return DiscreteVariable(low=self.low, high=self.high, value=value)
    __rpow__ = __pow__

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            self.value = self.value // other
        else:
            self.value = self.value // other.value
        self.__bound_check__()
        return self

    __rtruediv__ = __truediv__

    @value.setter
    def value(self, value):
        self._value = value


class ContinuousVariable:
    def __init__(self, low=np.NINF, high=np.Inf, value=None):
        self.low = low
        self.high = high
        self.value = rng.uniform(low=low, high=high) if value is None else value

        self.__bound_check__()

    def __bound_check__(self):
        if self.value > self.high:
            self.value = self.high
        elif self.value < self.low:
            self.value = self.low

    def draw_random(self, low=None, high=None):

        if low is None:
            low = -np.abs(self.high - self.low)
        if high is None:
            high = np.abs(self.high - self.low)

        return ContinuousVariable(value=rng.uniform(low=low, high=high))

    def __add__(self, other):
        value = None
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            value = self.value + other
        else:
            value = self.value + other.value
        # self.__bound_check__()
        return ContinuousVariable(low=self.low, high=self.high, value=value)

    __radd__ = __add__

    def __sub__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            self.value = self.value - other
        else:
            self.value = self.value - other.value
        self.__bound_check__()
        return self

    __rsub__ = __sub__

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            self.value = self.value * other
        else:
            self.value = self.value * other.value
        self.__bound_check__()
        return self

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            self.value = self.value / other
        else:
            self.value = self.value / other.value
        self.__bound_check__()
        return self

    __rtruediv__ = __truediv__

    def __pow__(self, other):
        value = None
        if isinstance(other, (int, float, complex)) and not isinstance(other, bool):
            value = self.value ** other
        else:
            value = self.value ** other.value
        self.__bound_check__()
        return ContinuousVariable(low=self.low, high=self.high, value=value)

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
        #
        # if low is None:
        #     low = -np.abs(self.high - self.low)
        # if high is None:
        #     high = np.abs(self.high - self.low)
        #
        # return Constant(value=rng.uniform(low=low, high=high))
        return self

    def __add__(self, other):
        return self

    __radd__ = __add__

    def __sub__(self, other):
        return self

    __rsub__ = __sub__

    def __mul__(self, other):
        return self

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self

    __rtruediv__ = __truediv__

    def __pow__(self, power, modulo=None):
        return self

    __rpow__ = __pow__

    def __call__(self):
        return self


class Vector:
    def __init__(self, elements):
        self.elements = np.array(elements)
        self._shape = None
        self._raw = None

    @classmethod
    def CONTINUOUS(cls, low=None, high=None, value=None):
        return (low, high, value, float)

    @classmethod
    def DISCRETE(cls, low=None, high=None, value=None):
        return (low, high, value, int)

    @classmethod
    def CONSTANT(cls, value=None):
        return Constant(value=value)

    @classmethod
    def from_definition(cls, definition: list):
        v = []
        for d in definition:
            # low, high, type = d
            n = d["repeat"] if "repeat" in d else 1
            if d["type"] == "continuous":
                low = d["low"]
                high = d["high"]
                for i in range(n):
                    v.append(ContinuousVariable(low=low, high=high))
            elif d["type"] == "discrete":
                low = d["low"]
                high = d["high"]
                for i in range(n):
                    v.append(DiscreteVariable(low=low, high=high))
            elif d["type"] == "constant":
                value = d["value"]
                for i in range(n):
                    v.append(Constant(value))

        return cls(elements=v)

    @classmethod
    def two_dims(cls):
        return cls.from_definition(definition=[
            {
                "low": -100,
                "high": 100,
                "type": "continuous",
            },
            {
                "low": -100,
                "high": 100,
                "type": "continuous",
            },
        ])

    @property
    def raw(self):
        values = np.array([e.value for e in self.elements])
        return values

    @property
    def shape(self):
        return self.elements.shape

    def draw_random_like(self, array=None, low=None, high=None):
        if array is None:
            array = self.elements
        return Vector(np.array([e.draw_random(low, high) for e in array]))

    def __getitem__(self, item):
        return self.elements[item]

    def __len__(self):
        return len(self.elements)

    def __add__(self, other):
        new_value = self.elements + other
        return Vector(elements=new_value)
        # return self
    __radd__ = __add__

    def __sub__(self, other):
        return Vector(elements=self.elements - other)
        # self.elements = self.elements - other
        # return self
    __rsub__ = __sub__

    def __mul__(self, other):
        return Vector(elements=self.elements * other)
        # self.elements = self.elements * other
        # return self
    __rmul__ = __mul__

    def __pow__(self, other):
        return Vector(elements=self.elements + other)
        # self.elements = self.elements ** other
        # return self
    __rpow__ = __pow__

    def __truediv__(self, other):
        return Vector(elements=self.elements / other)
        # self.elements = self.elements / other
        # return self
    __rtruediv__ = __truediv__

    def __eq__(self, other):
        if (self.raw == other.raw).all():
            return True
        else:
            return False
