import seaborn as sns


def cubehelix_cmap(color: str = "red", dark: float = 0.20):
    kws = dict(
        n_colors=12,
        rot=0.0,
        gamma=1.0,
        hue=1.0,
        light=1.0,
        dark=dark,
        as_cmap=True,
    )

    cmap = None
    if color == "red":
        cmap = sns.cubehelix_palette(start=0.9, **kws)
    elif color == "pink":
        cmap = sns.cubehelix_palette(start=0.8, **kws)
    elif color == "blue":
        cmap = sns.cubehelix_palette(start=2.8, **kws)
    else:
        raise ValueError
    return cmap