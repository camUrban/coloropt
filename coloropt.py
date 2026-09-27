import click
import colour
import logging
import os
import uuid
import sys
import numpy as np
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.optimize import minimize
from colortools import *


# The cost function and single-start optimizer live at module level so worker processes can use them.
def cost_function(x, weights, bounds):
    l_from, l_to, c_from, c_to, h_from, h_to = bounds
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


def run_start(start, weights, bounds):
    res = minimize(cost_function, start, args=(weights, bounds), method='Powell', tol=1e-9,
                   options={'maxfev': len(start)*10000})
    return res.x, res.fun, res.nfev


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
              help='random seed for the initial hues and restarts (default: random, logged)')
@click.option('--restarts', type=int, default=1,
              help='number of optimizer starts; starts after the first use random hues, lightness, and chroma (default: 1)')
@click.option('--workers', type=int, default=None,
              help='parallel processes for the starts (default: one per start, up to the CPU count)')
@click.option('--reorder/--no_reorder', default=True,
              help='reorder the result so each leading sub-palette scores well, or sort it by hue (default: reorder)')
@click.option('--c_from', type=float, default=50, help='a')
@click.option('--c_to', type=float, default=75, help='b')
@click.option('--h_from', type=float, default=0, help='c')
@click.option('--h_to', type=float, default=360, help='d')
@click.option('--l_from', type=float, default=40, help='a')
@click.option('--l_to', type=float, default=75, help='b')
@click.option('--logname', type=str, default=None, help='log file name in logs/, without .log (default: random run id)')
def main(weights, hues, n_colors, seed, restarts, workers, reorder, c_from, c_to, h_from, h_to, l_from, l_to, logname):
    weights = np.array(weights)
    if seed is None:
        seed = int(np.random.default_rng().integers(2**32))
    rng = np.random.default_rng(seed)
    if hues is not None:
        hues = np.array([int(h) for h in hues.split(',')])
    else:
        # Random permutation of evenly spaced hues covering the whole hue range, with a random offset.
        spacing = (h_to - h_from) / n_colors
        hues = h_from + spacing * (rng.random() + rng.permutation(n_colors))
    bounds = (l_from, l_to, c_from, c_to, h_from, h_to)
    if workers is None:
        workers = min(restarts, os.cpu_count() or 1)

    #  set up logging
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    if logname is None:
        logname = uuid.uuid4().hex[:16]

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'{logname}.log')
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    root.handlers = [file_handler, console_handler]

    root.info(f'Starting with parameters: weights={weights} hues={hues} seed={seed} restarts={restarts} workers={workers} reorder={reorder} c_from={c_from} c_to={c_to} h_from={h_from} h_to={h_to} l_from={l_from} l_to={l_to}')
    # The first start uses the given (or random) hues at mid lightness and chroma. Later starts draw
    # new evenly spaced hues plus random lightness and chroma per color, to escape local optima.
    n = len(hues)
    starts = [np.column_stack([np.full(n, (l_from+l_to)/2), np.full(n, (c_from+c_to)/2), hues]).ravel()]
    for _ in range(restarts - 1):
        spacing = (h_to - h_from) / n
        start_hues = h_from + spacing * (rng.random() + rng.permutation(n))
        starts.append(np.column_stack([rng.uniform(l_from, l_to, n), rng.uniform(c_from, c_to, n), start_hues]).ravel())
    results = [None] * len(starts)

    def log_start(i):
        x, fun, nfev = results[i]
        start_colors = [tuple(int(v) for v in np.floor(0.5 + clamp(lch_to_srgb(l, c, h))*255))
                        for l, c, h in zip(*[iter(x)]*3)]
        root.info(f'Start {i+1}/{len(starts)}: score={-fun} evaluations={nfev} colors: {start_colors}')

    if workers <= 1:
        for i, start in enumerate(starts):
            results[i] = run_start(start, weights, bounds)
            log_start(i)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(run_start, start, weights, bounds): i for i, start in enumerate(starts)}
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                log_start(i)
    ranking = np.argsort([fun for _, fun, _ in results])
    root.info(f'Best start: {ranking[0]+1}')

    # Palette figure in the style of the blog post, one panel per start from best to worst (top 10
    # only): the seed colors the optimizer started from, the colors, then how they look in grayscale
    # and to the two colorblind simulations.
    shown = ranking[:10]
    fig, axes = plt.subplots(len(shown), 1, figsize=(0.8*n + 1.5, 4.2*len(shown)), squeeze=False)
    for rank, (i, ax) in enumerate(zip(shown, axes[:, 0])):
        colors = [clamp(lch_to_srgb(l, c, h)) for l, c, h in zip(*[iter(results[i][0])]*3)]
        seeds = [clamp(lch_to_srgb(l, c, h)) for l, c, h in zip(*[iter(starts[i])]*3)]
        if reorder:
            order = subpalette_order(colors, weights)
        else:
            order = np.argsort(colour.Lab_to_LCHab(srgb_to_lab(np.array(colors)))[:, 2])
        colors = [colors[j] for j in order]
        seeds = [seeds[j] for j in order]
        score = multicolor_cost(colors, weights)
        if rank == 0:
            root.info(f'Score={score} colors: {list(map(lambda x: tuple(int(v) for v in np.floor(0.5 + x*255)), colors))}')
        grid = np.array([seeds,
                         colors,
                         [to_grayscale(c) for c in colors],
                         [to_colorblind_g(c) for c in colors],
                         [to_colorblind_r(c) for c in colors]])
        ax.imshow(grid, interpolation='nearest')
        ax.set_xticks([])
        ax.set_yticks(range(5), ['Seed', 'Normal', 'Grayscale', 'Colorblind (g)', 'Colorblind (r)'])
        ax.tick_params(left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f'Start {i+1} (rank {rank+1}, score {score:.4f})')
    axes[-1, 0].set_xlabel(f'{n} colors')
    fig.tight_layout()
    fig_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs', f'{logname}.png')
    fig.savefig(fig_filename, dpi=150)
    root.info(f'Saved palette figure to {fig_filename}')
    plt.show()

    file_handler.close()
    console_handler.close()


if __name__ == '__main__':
    main()
