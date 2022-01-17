import numpy as np
from enum import Enum


def buildermethod(func):
    def wrapper(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return wrapper


class HP_TYPE(Enum):
    MODEL = 0,
    TRAINING = 1


class Hyperparams:
    def __init__(self, defs=dict(), values=None):
        self._definitions = defs
        self.vector = values

    @buildermethod
    def add_cont(self, name, low, high, kind=HP_TYPE.MODEL):
        self._definitions[name] = {
            "low": low,
            "high": high,
            "space": "continuous",
            "type": kind,
        }

    @buildermethod
    def add_discrete(self, name, low, high, kind=HP_TYPE.MODEL):
        self._definitions[name] = {
            "low": low,
            "high": high,
            "space": "discrete",
            "type": kind,
        }

    @buildermethod
    def add_constant(self, name, value, kind=HP_TYPE.MODEL):
        self._definitions[name] = {
            "low": value,
            "high": value,
            'value': value,
            "space": "constant",
            "type": kind,
        }

    def build(self):
        values = list(self._definitions.values())
        values = [dict(map(lambda key: (key, ld.get(key, None)), ['low', 'high', 'space'])) for ld in values]
        values = [v if v['space'] != 'constant' else {'value': v['low'], 'space': v['space']} for v in values]
        self.vector = [np.inf for _ in range(len(values))]
        return values

    def match(self, new_vals: list):
        if len(self._definitions.keys()) != len(new_vals):
            raise ValueError("Cannot update values: wrong argument length!")

        self.vector = new_vals

        return Hyperparams(defs=self._definitions, values=new_vals)


    def get_params_by_type(self, kind=HP_TYPE.MODEL):

        d = dict(zip(self._definitions.keys(), self.vector))

        model_param_names = [name for name, constraints in self._definitions.items() if constraints['type'] == kind]

        model_params = {k: d[k] for k in model_param_names}

        return model_params

    def get_as_dict(self):
        return dict(zip(self._definitions.keys(), self.vector))

    def get_model_params(self):
        return self.get_params_by_type(kind=HP_TYPE.MODEL)


    def __getitem__(self, item):
        d = dict(zip(self._definitions.keys(), self.vector))
        return d[item]


# hps = Hyperparams().add_cont('a', 10, 20).add_discrete('b', 20, 30)
# raw = hps.build()
# hp = hps.get_hp_dict([15, 23])
# print()
