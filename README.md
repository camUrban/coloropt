# coloropt

This code accompanies [my blogpost](http://tsitsul.in/blog/coloropt/) on color optimization.

### Basic usage

First, install the dependencies:

    uv sync

Then, you can use the command-line tool as follows:

    uv run coloropt.py --weights WEIGHT_VECTOR --hues INITIAL_HUES --c_from 0 --c_to 100 --l_from 0 --l_to 100 --logname LOG_NAME

where ``WEIGHT_VECTOR`` is a parameters for the objective function and ``INITIAL_HUES`` are initial guesses for the hue values for the optimal colors, given as a comma-separated list. Every option has a default, so ``uv run coloropt.py`` on its own also works. If ``--hues`` is omitted, the initial hues are a random permutation of ``--n_colors`` evenly spaced hues (6 by default), and ``--seed`` makes them reproducible. The seed used is always written to the log.

After optimizing, the colors are reordered so that each leading sub-palette (the first 2, first 3, and so on) scores well. The log and a palette figure are saved to ``logs/LOG_NAME.log`` and ``logs/LOG_NAME.png``, and the figure is also shown in a window.

Example usage with fixed initial hues:

    uv run coloropt.py --weights 500 100 75 50 25 100 75 50 25 10 25 10 20 10 1000 100 50 25 10 30 10 350 150 100 50 25 200 100 50 25 10 --hues 40,200,240,360 --c_from 50 --c_to 75 --l_from 40 --l_to 75 --logname example

Example usage with 8 random initial hues:

    uv run coloropt.py --n_colors 8 --seed 1 --logname random8
    
    
### License

You are free to use the pallettes and the code without attribution in commercial projects (specifically, assume [public domain](https://creativecommons.org/share-your-work/public-domain/cc0/) license on the colors). I would love if you drop an email if you find them useful.
