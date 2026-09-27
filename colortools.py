import colour
import numpy as np
import itertools

# Colors are sRGB arrays in [0, 1]. Lab and LCh are relative to D65, sRGB's white point.
D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']

def srgb_to_lab(rgb):
    return colour.XYZ_to_Lab(colour.sRGB_to_XYZ(rgb), illuminant=D65)

def lch_to_srgb(l, c, h):
    xyz = colour.Lab_to_XYZ(colour.LCHab_to_Lab([l, c, h]), illuminant=D65)
    return colour.XYZ_to_sRGB(xyz, illuminant=D65)

def pairwise_delta_e(labs):
    n = len(labs)
    i, j = np.triu_indices(n, k=1)
    dist = colour.delta_E(labs[i], labs[j], method='CIE 2000')
    cdists = np.zeros((n, n))
    cdists[i, j] = dist
    cdists[j, i] = dist
    return cdists

black_lab = srgb_to_lab(np.zeros(3))
white_lab = srgb_to_lab(np.ones(3))

def to_grayscale(rgb):
    # The neutral gray with the same perceptual lightness L*.
    lab = srgb_to_lab(rgb)
    lab[..., 1:] = 0
    return colour.XYZ_to_sRGB(colour.Lab_to_XYZ(lab, illuminant=D65), illuminant=D65)

def clamp(rgb):
    return np.clip(rgb, 0, 1)

def to_colorblind_g(rgb):
    r, g, b = np.floor(0.5 + np.asarray(rgb)*255)
    r_ = np.power(4211.106+0.6770*(g**2.2)+0.2802*(r**2.2), 1/2.2)
    g_ = np.power(4211.106+0.6770*(g**2.2)+0.2802*(r**2.2), 1/2.2)
    b_ = np.power(4211.106+0.95724*(b**2.2)+0.02138*(g**2.2)-0.02138*(r**2.2), 1/2.2)
    return np.array([r_, g_, b_]) / 255

def to_colorblind_r(rgb):
    r, g, b = np.floor(0.5 + np.asarray(rgb)*255)
    r_ = np.power(782.74+0.8806*(g**2.2)+0.1115*(r**2.2), 1/2.2)
    g_ = np.power(782.74+0.8806*(g**2.2)+0.1115*(r**2.2), 1/2.2)
    b_ = np.power(782.74+0.992052*(b**2.2)-0.003974*(g**2.2)+0.003974*(r**2.2), 1/2.2)
    return np.array([r_, g_, b_]) / 255

def anglediff(h1, h2):
    x, y = h1*np.pi/180, h2*np.pi/180
    return np.abs(np.arctan2(np.sin(x-y), np.cos(x-y))) * 180 / np.pi

def multicolor_cost(colors, weights):
    return np.sum(multicolor_cost_debug(colors, weights))/np.sum(weights)

def subpalette_order(colors, weights):
    # Greedy order so that every leading sub-palette (first 2, first 3, ...) scores well: start
    # from the best pair, then repeatedly append the color that gives the best next sub-palette.
    n = len(colors)
    order = list(max(itertools.combinations(range(n), 2),
                     key=lambda p: multicolor_cost([colors[i] for i in p], weights)))
    while len(order) < n:
        rest = [i for i in range(n) if i not in order]
        order.append(max(rest, key=lambda i: multicolor_cost([colors[j] for j in order + [i]], weights)))
    return order

def multicolor_cost_debug(colors, weights):
    scores = np.zeros(31)
    colors = np.asarray(colors)
    ncolors = len(colors)
    weights = np.array(weights)

    colors_lab = srgb_to_lab(colors)
    cdists = pairwise_delta_e(colors_lab) / 116

    quantiles = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
    scores[0:5] = weights[0:5]*quantiles

    colors_lch = colour.Lab_to_LCHab(colors_lab)

    cdists = np.zeros((ncolors, ncolors))
    for i in range(ncolors):
        for j in range(i+1, ncolors):
            dist = anglediff(colors_lch[i, 2], colors_lch[j, 2])
            cdists[i, j] = dist
            cdists[j, i] = dist

    reals = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
    opts = np.array([2/ncolors, 0.25, 0.5, 0.75, 1])*360/2
    scores[5:10] = weights[5:10]*(1-np.abs(opts-reals)/opts)

    if weights[10] > 0 or weights[11] > 0:
        dists = colour.delta_E(colors_lab, white_lab, method='CIE 2000') / 100
        scores[11] = weights[11] * np.sum(dists) / ncolors
        scores[10] = weights[10] * np.min(dists)

    if weights[12] > 0 or weights[13] > 0:
        dists = colour.delta_E(colors_lab, black_lab, method='CIE 2000') / 100
        scores[13] = weights[13] * np.sum(dists) / ncolors
        scores[12] = weights[12] * np.min(dists)

    colors_gray = srgb_to_lab(np.array([to_grayscale(rgb) for rgb in colors]))

    if np.any(weights[14:19]>0):
        cdists = pairwise_delta_e(colors_gray) / 116

        quantiles = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
        scores[14:19] = weights[14:19]*quantiles

    if weights[19] > 0 or weights[20] > 0:
        dists = colour.delta_E(colors_gray, white_lab, method='CIE 2000') / 100
        scores[20] = weights[20] * np.sum(dists) / ncolors
        scores[19] = weights[19] * np.min(dists)

    if np.any(weights[21:26]>0):
        colors_cb_g = srgb_to_lab(np.array([to_colorblind_g(rgb) for rgb in colors]))
        cdists = pairwise_delta_e(colors_cb_g) / 116

        quantiles = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
        scores[21:26] = weights[21:26]*quantiles

    if np.any(weights[26:31]>0):
        colors_cb_r = srgb_to_lab(np.array([to_colorblind_r(rgb) for rgb in colors]))
        cdists = pairwise_delta_e(colors_cb_r) / 116

        quantiles = np.quantile(cdists[~np.eye(ncolors, dtype=bool)], [0, 0.25, 0.5, 0.75, 1])
        scores[26:31] = weights[26:31]*quantiles
    
    return scores