from dataclasses import dataclass


@dataclass
class TensorShape:
    num: int  # number of separate instances
    dim: int  # dimensionality of a single instance (needs to be flat)


@dataclass
class DatasetShape:
    x: TensorShape
    u: TensorShape
    y: TensorShape
    v: TensorShape
