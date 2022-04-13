import scipy.signal as sp
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt
import math
from matplotlib.widgets import Slider, Button
from scipy.interpolate import interp2d
import cv2

def generate_stimuli(pos = [0, 0], size=20):
    if type(pos) == list:
        pos = np.asarray(pos).reshape(1, len(pos))
        
    stims = np.zeros([size, size, pos.shape[0]])
    
    for i in range(pos.shape[0]):
        center = pos[i]
        x = round(pos[i, 0]) + int(size/2)
        y = round(pos[i, 1]) + int(size/2)
        
        stims[y, x, i] = 1
    
    return np.squeeze(stims), pos


def generate_prf(center = [0, 0], size=20, sd=10):
    x = np.arange(-int(size/2), int(size/2), 1, float)
    y = x[:,np.newaxis]
    
    prf =  np.exp(- ((x-center[0])**2 + (y-center[1])**2) / (2*sd**2))
    
    return prf


def generate_prf_warp(center = [0, 0], size=20, sd=10, y_warp = 1, x_warp = 1):
    x = np.arange(-int(size/2), int(size/2), 1, float)
    y = x[:,np.newaxis] * y_warp
    x = x * x_warp
    
    prf =  np.exp(- ((x-center[0])**2 + (y-center[1])**2) / (2*sd**2))
    
    prf /= prf.sum()
    
    return prf


def generate_voxels(deg_radius=20, n_eccentricities=10, prf_slope=0.2, prf_intercept = 0.6):
    # v1, v2 slope = 0.2
    # v3 = 0.25
    # hV4 = 0.5
    pos = np.linspace(0, deg_radius, n_eccentricities)
    xs = np.concatenate([-pos[:0:-1], pos])
    ys = xs[:, np.newaxis][::-1]

    voxels = []

    for x in xs.flatten():
        for y in ys.flatten():
            ecc = np.linalg.norm(np.array([x, y]))
            prf_size = ecc * prf_slope + prf_intercept

            voxels.append([x, y, ecc, prf_size])

    voxels = np.asarray(voxels)
    
    return voxels


def generate_isostim(eccentricity, n_pos, size):
    xstart = np.arange(0, 2*np.pi, 2*np.pi / n_pos)
    xstop = xstart + 2*np.pi / n_pos
    x = np.random.uniform(xstart, xstop)
    
    positions = eccentricity * np.array([np.sin(x), np.cos(x)]).T 
    
    stims, pos = generate_stimuli(positions, size = size)
    
    return stims, positions, x


def gen_pRFresponses(near_ecc, target_ecc, stims, prf_maps, voxels, positions, all_vox = False):
    # near_ecc = 2
    # stim = stims[:, :, slice]
    # angle_s = angles[slice]
    # pos = positions[slice]
    # angle = np.arctan2(pos[1], pos[0])
    # coords = np.where(stim == 1)
    # threshold = 0.01
    target_idxs = np.where((voxels[:, 2] <= target_ecc + near_ecc) & (voxels[:, 2] >= target_ecc - near_ecc))
    target_voxels = voxels[target_idxs]
    target_prf_maps = prf_maps[:, :, target_idxs].squeeze()
    
    if all_vox:
        target_voxels = voxels
        target_prf_maps = prf_maps[:, :, :].squeeze()
        
    prf_responses = np.zeros([len(target_voxels), 6, stims.shape[-1]])     
    
    for s in range(stims.shape[-1]):
        stim = stims[:, :, s]
        prf_resps = []
        pos = positions[s]
        angle = np.arctan2(pos[1], pos[0])
        
        for i, v in enumerate(target_voxels):
            x, y, ecc, prf_size = v
            prf = target_prf_maps[:, :, i]
            prf_resp = stim.flatten() @ prf.flatten()
            vox_deg_in_rads = np.arctan2(y, x)

            vox_diff = angle - vox_deg_in_rads
            if vox_diff >= np.pi:
                vox_diff -= 2*np.pi
            elif vox_diff <= -np.pi:
                vox_diff += 2*np.pi
            
            prf_resps.append([x, y, ecc, vox_deg_in_rads, vox_diff, prf_resp])

        prf_responses[:, :, s] = np.asarray(prf_resps)
    
    return prf_responses


def gen_LTMstims(stims, size, filt_size = 20, sd = 4.5):
    filt = generate_prf(size = filt_size, sd = sd)
    ltm_stims = np.zeros([size, size, stims.shape[-1]])
    
    for s in range(stims.shape[-1]):
        stim = stims[:, :, s]
        stim_conv = sp.convolve2d(stim, filt, mode = 'same')
        #edge = int(stim_conv.shape[0]/2 - stim.shape[0]/2)

        #stim_conv = stim_conv[edge+1:-edge, edge+1:-edge]
        stim_conv /= stim_conv.sum()
        ltm_stims[:, :, s] = stim_conv
    
    ltm_stims = np.asarray(ltm_stims)
    
    return ltm_stims


def gen_radialgauss(stim, filt, distnorm = False):
    size = stims.shape[0]
    rad_gaussians = np.zeros([size, size, stim.shape[-1]])
    
    for s in range(stims.shape[-1]):
        stim = stims[:, :, s]
        pos = np.where(stim == 1)
        line = np.array([np.round(np.linspace(size/2, pos[0])), np.round(np.linspace(size/2, pos[1]))]).T.squeeze()
        source = np.zeros(stim.shape)

        for coord in line:
            x = int(coord[0]) - 1
            y = int(coord[1]) - 1
            if distnorm:
                source[x, y] = np.linalg.norm(np.array(coord) - np.array([size/2, size/2]))
            else:
                source[x, y] = 1
        rg = sp.convolve2d(source, filt, mode = 'same')
        rg /= np.sum(rg)
        rad_gaussians[:, :, s] = rg
    
    
    return line, source, rad_gaussians


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def gen_polarstims(stims, filt, origin, radius, space = "both"):
    polars = np.zeros(stims.shape)
    logpolars = np.zeros(stims.shape)
    size = stims.shape[0]
    
    for s in range(stims.shape[-1]):
        stim = stims[:, :, s]
        # Linear Polar
        stim_polar = cv2.warpPolar(stim, stim.shape, origin, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LINEAR)
        angle_change = size/2 - np.where(stim_polar == np.amax(stim_polar))[0][0]
        stim_polar = np.roll(stim_polar, int(angle_change) , axis = 0)
        stim_pconv = sp.convolve2d(stim_polar, filt, mode = 'same', boundary = 'symmetric')
        stim_pconv = np.roll(stim_pconv, -int(angle_change) , axis = 0)
        stim_pconv_cart = cv2.warpPolar(stim_pconv, stim.shape, origin, radius, cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LINEAR + cv2.WARP_FILL_OUTLIERS)
        stim_pconv_cart /= stim_pconv_cart.flatten().sum()
        
        polars[:, :, s] = stim_pconv_cart

        # Log Polar warp
        stim_log = cv2.warpPolar(stim, stim.shape, origin, radius, cv2.WARP_FILL_OUTLIERS + cv2.WARP_POLAR_LOG)
        stim_log = np.roll(stim_log, int(angle_change) , axis = 0)
        stim_lconv = sp.convolve2d(stim_log, filt, mode = 'same', boundary = 'symmetric')
        stim_lconv = np.roll(stim_lconv, -int(angle_change) , axis = 0)
        stim_lconv_cart = cv2.warpPolar(stim_lconv, stim.shape, origin, radius, cv2.WARP_INVERSE_MAP + cv2.WARP_POLAR_LOG + cv2.WARP_FILL_OUTLIERS)
        stim_lconv_cart /= stim_lconv_cart.flatten().sum()
        
        logpolars[:, :, s] = stim_lconv_cart
        
    if space == 'both':
        return polars, logpolars
    elif space == 'linear':
        return polars
    elif space == 'log':
        return logpolars

    
def parf_tuningcurve(n_bins, prf_responses):
    pct_signal = prf_responses[:, -1, :].flatten()
    angle_diff = prf_responses[:, -2, :].flatten() 
    
    bins_idxs = np.linspace(-np.pi, np.pi, n_bins)
    bins = np.stack([bins_idxs[:-1], bins_idxs[1:]], axis = 1)
    
    parf = np.zeros([len(bins), 2])
    
    for i, b in enumerate(bins):
        idxs = np.where((angle_diff >= b[0]) & (angle_diff <= b[1]))
        parf[i] = np.array([np.mean(angle_diff[idxs]), np.nanmean(pct_signal[idxs])])
        
    return parf
        