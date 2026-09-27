import click
import logging
import os
import uuid
import sys
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from colortools import *


@click.command()
@click.option('--weights', type=int, nargs=31,
              default=(500, 100, 75, 50, 25, 100, 75, 50, 25, 10, 25, 10, 20, 10, 1000, 100,
                       50, 25, 10, 30, 10, 350, 150, 100, 50, 25, 200, 100, 50, 25, 10),
              help='31 objective weights')
@click.option('--hues', type=str, default=None,
              help='comma-separated initial hues, one per color (default: random, see --n_colors)')
@click.option('--n_colors', type=int, default=6,
              help='number of colors when --hues is not given')
@click.option('--seed', type=int, default=None,
              help='random seed for the initial hues (default: random, logged)')
@click.option('--c_from', type=float, default=50, help='a')
@click.option('--c_to', type=float, default=75, help='b')
@click.option('--h_from', type=float, default=0, help='c')
@click.option('--h_to', type=float, default=360, help='d')
@click.option('--l_from', type=float, default=40, help='a')
@click.option('--l_to', type=float, default=75, help='b')
@click.option('--logname', type=str, default=None, help='log file name in logs/, without .log (default: random run id)')
def main(weights, hues, n_colors, seed, c_from, c_to, h_from, h_to, l_from, l_to, logname):
    weights = np.array(weights)
    if hues is not None:
        hues = np.array([int(h) for h in hues.split(',')])
    else:
        # Random permutation of evenly spaced hues covering the whole hue range, with a random offset.
        if seed is None:
            seed = int(np.random.default_rng().integers(2**32))
        rng = np.random.default_rng(seed)
        spacing = (h_to - h_from) / n_colors
        hues = h_from + spacing * (rng.random() + rng.permutation(n_colors))

    
    def cost_function(x):
        colors = []
        for l, c, h in zip(*[iter(x)]*3):
            if h > h_to or h < h_from:
                return 0
            if l > l_to or l < l_from:
                return 0
            if c > c_to or c < c_from:
                return 0
            colors.append(clamp(lch_to_srgb(l, c, h)))
        return -multicolor_cost(colors, weights)

    #  set up logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if logname is None:
        logname = uuid.uuid4().hex[:16]

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'{logname}.log')
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.handlers = [file_handler, console_handler]

    root.info(f'Starting with parameters: weights={weights} hues={hues} seed={seed} c_from={c_from} c_to={c_to} h_from={h_from} h_to={h_to} l_from={l_from} l_to={l_to}')
    x0 = []
    for h in hues:
        x0.extend([(l_from+l_to)/2, (c_from+c_to)/2, h])
    res = minimize(cost_function, x0, method='Powell', tol=1e-9, options={'maxfev': len(x0)*10000, 'disp': True})
    colors = []
    for l, c, h in zip(*[iter(res.x)]*3):
        colors.append(clamp(lch_to_srgb(l, c, h)))
    seeds = [clamp(lch_to_srgb(l, c, h)) for l, c, h in zip(*[iter(x0)]*3)]
    order = subpalette_order(colors, weights)
    colors = [colors[i] for i in order]
    seeds = [seeds[i] for i in order]
    root.info(f'Score={multicolor_cost(colors, weights)} colors: {list(map(lambda x: tuple(int(v) for v in np.floor(0.5 + x*255)), colors))}')

    # Palette figure in the style of the blog post: the seed colors the optimizer started from, the
    # colors, then how they look in grayscale and to the two colorblind simulations.
    grid = np.array([seeds,
                     colors,
                     [to_grayscale(c) for c in colors],
                     [to_colorblind_g(c) for c in colors],
                     [to_colorblind_r(c) for c in colors]])
    fig, ax = plt.subplots(figsize=(0.8*len(colors) + 1.5, 4.2))
    ax.imshow(grid, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks(range(5), ['Seed', 'Normal', 'Grayscale', 'Colorblind (g)', 'Colorblind (r)'])
    ax.tick_params(left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(f'{len(colors)} colors')
    fig.tight_layout()
    fig_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'{logname}.png')
    fig.savefig(fig_filename, dpi=150)
    root.info(f'Saved palette figure to {fig_filename}')
    plt.show()

    file_handler.close()
    console_handler.close()


if __name__ == '__main__':
    main()
