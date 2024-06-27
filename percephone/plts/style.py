"""
Théo Gauvrit 18/01/2024
Style and aesthetics for plots
"""


def boxplot_style(func):
    def wrapper(ax, *args, **kwargs):
        ax.grid(False)
        ax.set_facecolor("white")
        ax.spines[["right", "top", "bottom"]].set_visible(False)
        ax.spines["left"].set_color("black")
        ax.tick_params(axis='both', which='major', length=8, width=3, color="black", left=True)
        func(ax, *args, **kwargs)

    return wrapper

