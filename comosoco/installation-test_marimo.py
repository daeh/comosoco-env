import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    ## Load the dependencies
    """)
    return


@app.cell
def _():
    import jax
    import jax.numpy as jnp
    from memo import memo
    from enum import IntEnum, auto
    from matplotlib import pyplot as plt
    return IntEnum, auto, memo


@app.cell
def _(mo):
    mo.md(r"""
    ## Run a `memo` model
    """)
    return


@app.cell
def _(IntEnum, Pr, auto, c, chooses, memo):
    class Card(IntEnum):
        ONE = auto()
        TWO = auto()

    @memo
    def game[_c: Card]():
        player: chooses(c in Card, wpp=1)
        return Pr[player.c == _c]

    game(print_table=True)
    return (game,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Check if `pandas` is installed
    """)
    return


@app.cell
def _(game):
    game(return_aux=True, return_pandas=True).aux.pandas
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Check if `xarray` is installed
    """)
    return


@app.cell
def _(game):
    game(return_aux=True, return_xarray=True).aux.xarray
    return


if __name__ == "__main__":
    app.run()
