# %% Load the dependencies

import jax
import jax.numpy as jnp
from memo import memo
from enum import IntEnum, auto
from matplotlib import pyplot as plt

# %% Run a `memo` model

class Card(IntEnum):
    ONE = auto()
    TWO = auto()

@memo
def game[_c: Card]():
    player: chooses(c in Card, wpp=1)
    return Pr[player.c == _c]

game(print_table=True)

# %% Check if `pandas` is installed

game(return_aux=True, return_pandas=True).aux.pandas

# %% Check if `xarray` is installed

game(return_aux=True, return_xarray=True).aux.xarray
