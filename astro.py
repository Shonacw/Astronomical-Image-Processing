#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:47:04 2019

@author: gennadiigorbun
"""
from tqdm import tqdm
from astropy.io import fits
from astropy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm
import numpy.ma as ma
import random
import pdb
import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

path_mos = '/Users/gennadiigorbun/Documents/Imperial/AstroLab/A1_mosaic.fits'
mos = fits.open(path_mos)
#  Header:
data = mos[0].data
mos.close()
data_ma = ma.masked_array(data)

msk = np.zeros([len(data[0]), len(data)])

def test_1():
    #img = ma.masked_array(create_test_image())
    #img = ma.masked_array(data[3840:3853, 2260:2300])
    img = ma.masked_array(data[3820:3873, 2240:2310])
    img = ma.masked_array(data[3800:3900, 2200:2400])
    plot_2d(img)
    df = stats.sigma_clip(img, sigma=5)
    noise_local = np.mean(df)
    variance = np.var(df)
    sigma_noise = np.sqrt(variance)
    for i in range(30):
        centre = find_max_pxl(img)
        points_to_mask, r_star = extend(centre, img, noise_local, sigma_noise)
        for pt in points_to_mask:
            img[pt[0], pt[1]] = ma.masked
        plot_2d(img)
    print('Noise start: ', noise_local)
    noise_local = np.mean(stats.sigma_clip(img, sigma=5))
    print('Noise end: ', noise_local)

def maskit(i, j):
    msk[i, j] = 1
    return


def unmaskit(i, j):
    msk[i, j] = 0
    return


data_flat = data.flatten()
mean = np.mean(data_flat)
variance = np.var(data_flat)
sigma = np.sqrt(variance)
print(f'Before clipping: Mean = {mean}, std = {sigma}')


def points_in_circle_np(radius, x0=0, y0=0, ):
    x_ = np.arange(x0 - radius - 1, x0 + radius + 1, dtype=int)
    y_ = np.arange(y0 - radius - 1, y0 + radius + 1, dtype=int)
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2)
    # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
    for x, y in zip(x_[x], y_[y]):
        yield x, y


def create_test_image(size=[100, 100], std=20, centre=3421):
    gauss = np.random.normal(loc=centre, size=size, scale=std).astype(int)
    star_coords = list(points_in_circle_np(20, x0=50, y0=50))
#    x, y = np.meshgrid(np.arange(45, 55), np.arange(45, 55))
#    d = np.sqrt(x*x+y*y) 
#    sigma, mu = 20, 3500 
#    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )     
    for pt in star_coords:
        gauss[pt[0], pt[1]] = 3500 + random.randint(0, 20)
    return gauss


def clip():
    for n_sig in range(1, 7):
        # plt.hist(data_flat, bins=400, density=True, range=[3300, 3600])
        df = stats.sigma_clip(data, sigma=n_sig)
        mean_2 = np.mean(df)
        variance_2 = np.var(df)
        sigma_2 = np.sqrt(variance_2)
        print(f'Clipped to {n_sig} sigma: Mean = {mean_2}, std = {sigma_2}')
    return


def mask_edges():
    for i in [-1, 1]:
        while np.all(data[:, i] == 3421):
            data_ma[:, i] = ma.masked
            if i < 0:
                i -= 1
            else:
                i += 1
        while np.all(data[i, 3421] == 3421):
            data_ma[i, :] = ma.masked
            if i < 0:
                i -= 1
            else:
                i += 1


def plot_2d(arr):
    plt.figure()
    plt.imshow(arr, cmap=plt.get_cmap('plasma'))
    plt.colorbar()
    return


def extend(centre, data, mean_noise, noise_std):
    def within_frame(x, y, x_max, y_max):
        if x >= 0 and y >= 0 and x < x_max - 1 and y < y_max - 1:
            return True
        else:
            return False
    r = 1
    x0, y0 = centre[0], centre[1]
    x_m, y_m = len(data), len(data[0])
    pts = list(points_in_circle_np(r, x0=x0, y0=y0))
    r += 1
    pts2 = list(points_in_circle_np(r, x0=x0, y0=y0))
    #   ;  and 
    diff_pts = list(set(tuple(i) for i in pts2 if within_frame(i[0], i[1], x_m, y_m) and data[i] is not ma.masked) - set(tuple(i) for i in pts if within_frame(i[0], i[1], x_m, y_m) and data[i] is not ma.masked))
    flux_diff = sum([data[i] for i in diff_pts])
    noise_lim = len(diff_pts) * (mean_noise - 5 * noise_std)
    
    while flux_diff > noise_lim:
        pts = pts2
        r += 1
        pts2 = list(points_in_circle_np(r, x0=x0, y0=y0))
        diff_pts = list(set(tuple(i) for i in pts2 if within_frame(i[0], i[1], x_m, y_m) and data[i] is not ma.masked) - set(tuple(i) for i in pts if within_frame(i[0], i[1], x_m, y_m) and data[i] is not ma.masked))
        flux_diff = sum([data[x, y] for x, y in diff_pts])
        noise_lim = len(diff_pts) * (mean_noise + noise_std)
    flux = sum([int(data[i]) for i in pts if within_frame(i[0], i[1], x_m, y_m) and data[i] is not ma.masked])

    return pts, r, flux


def find_max_pxl(img):
    return np.unravel_index(img.argmax(), img.shape)

def get_local_noise(centre, data):
    def within_frame(x0, y0):
        x_max, y_max = len(data), len(data[0])
        if x0 >= 0 and y0 >= 0 and x0 < x_max - 1 and y0 < y_max - 1:
            return True
        else:
            return False
    x0, y0 = centre[0], centre[1]
    aperture = 6
    #pdb.set_trace()
    pts_in_apt = [data[pt[0], pt[1]] for pt in points_in_circle_np(aperture, x0=x0, y0=y0) if within_frame(pt[0], pt[1]) and data[pt[0], pt[1]] is not ma.masked]
    _df = stats.sigma_clip(pts_in_apt, sigma=5)
    noise_local, noise_local_sigma = np.mean(_df), np.std(_df)
    df = stats.sigma_clip(data, sigma=5)
    av_noise, av_noise_sigma = np.mean(df), np.std(df)
    #pdb.set_trace()
    while noise_local > av_noise + av_noise_sigma:
        aperture += 3
        pts_in_apt = [data[x, y] for x, y in points_in_circle_np(aperture, x0=x0, y0=y0)]
        _df = stats.sigma_clip(pts_in_apt, sigma=5)
        noise_local, noise_local_sigma = np.mean(_df), np.std(_df)
    return noise_local, noise_local_sigma, aperture
    
def run(img):
    res_path = '/Users/gennadiigorbun/Documents/Imperial/AstroLab/catalogue_{}.txt'
    num_file = 0
    while os.path.exists(res_path.format(num_file)):
        num_file += 1  
    f = open(res_path.format(num_file), 'w')
    #img = ma.masked_array(data[3000:3500, 2000:2400])
    plot_2d(img)
    
    df = stats.sigma_clip(img, sigma=5)
    noise_local = np.mean(df)
    sigma_noise = np.std(df)
    
    #pdb.set_trace()
    centre = find_max_pxl(img)
    noise_local, sigma_noise, apt = get_local_noise(centre, img)
    points_to_mask, r_star, flux_star = extend(centre, img, noise_local, sigma_noise)
    noise_flux = len(points_to_mask) * (noise_local + 2 * sigma_noise)
    count = 0
    while img[centre] > noise_local + sigma_noise and flux_star > noise_flux: 
        count+=1
        f.write(f'{count}\t{centre[0]}\t{centre[1]}\t{img[centre[0], centre[1]]}\t{noise_local}\t{apt}\t{r_star}\t{flux_star}\n')
        for x, y in points_to_mask:
            if x >= 0 and y >= 0 and y < len(img[0]) - 1 and x < len(img) - 1:
                img[x, y] = ma.masked
        print(count, flush=True)
        centre = find_max_pxl(img)
        noise_local, sigma_noise, apt = get_local_noise(centre, img)
        points_to_mask, r_star, flux_star = extend(centre, img, noise_local, sigma_noise)
        noise_flux = len([i for i in points_to_mask if img[i] is not ma.masked]) * (noise_local + 2 * sigma_noise)
#        df = stats.sigma_clip(img, sigma=5)
#        noise_local = np.mean(df)
#        sigma_noise = np.std(df)
    plot_2d(img)
    f.close()
    df = pd.read_csv(res_path.format(num_file), sep='\t', header=None)
    return img, df


def find_approx_bound():
    # img = ma.masked_array(data[3840:3853, 2260:2300])
    img = ma.masked_array(data[3820:3873, 2240:2310])
    plot_2d(img)
    centre = find_max_pxl(img)
    df = stats.sigma_clip(img, sigma=5)
    # need to calc local noise over a bigger area
    noise_local = np.mean(df)    
    t = b = centre[0]
    l = r = centre[1]    
    
    while img[t, centre[1]] > noise_local and t < len(img) - 1:
        t += 1
        
    while img[b, centre[1]] > noise_local and b > 0:
        b -= 1
    
    while img[centre[0], l] > noise_local and l > 0:
        l -= 1
        
    while img[centre[0], r] > noise_local and r < len(img[0]):
        r += 1


def premask(img):
    plot_2d(img)
    # middle star
    msk = [[y, x] for x, y in points_in_circle_np(325, x0=1435, y0=3201)]
    for x, y in msk:
        img[x, y] = ma.masked
    # bleeding of that star
    img.mask[:, 1420:1445] = True
    # another star
    msk = [[y, x] for x, y in points_in_circle_np(50, x0=777, y0=3320)]
    for x, y in msk:
        img[x, y] = ma.masked
        # another star
    msk = [[y, x] for x, y in points_in_circle_np(52, x0=907, y0=2283)]
    for x, y in msk:
        img[x, y] = ma.masked
    # another star
    msk = [[y, x] for x, y in points_in_circle_np(50, x0=2135, y0=3757)]
    for x, y in msk:
        img[x, y] = ma.masked
    # another star
    msk = [[y, x] for x, y in points_in_circle_np(50, x0=974, y0=2771)]
    for x, y in msk:
        img[x, y] = ma.masked
    # left top corner
    img.mask[4514:, :117] = True
    # left bottom corner
    img.mask[:423, :121] = True
    img.mask[:120, :432] = True
    # right top corner
    img.mask[4510:, 2162:] = True
    img.mask[4208:, 2477:] = True
    # right bottom corner
    img.mask[:115, 2470:] = True
    # horizontal bleeding
    img.mask[423:464, 1102:1654] = True
    img.mask[312:358, 1079:1680] = True
    img.mask[217:262, 1391:1477] = True
    img.mask[114:152, 1286:1546] = True
    img.mask[:300, 1410:1460] = True
    # edges
    img.mask[:144, :] = True
    img.mask[:, :144] = True
    img.mask[:, -144:] = True
    img.mask[-144:, :] = True
    return img


def calc_mag(counts):
    """
    This function converts the number count to a calibrated magnitude.
    """
    ZP = 2.530E+01
    m = ZP - (2.5 * math.log10(counts))
    return m
 

def do_the_thing():
    img = premask(data_ma)
    plot_2d(img)
    # final_img, df = run(img)
    # plot_2d(final_img)

#plt.figure(1)
#plt.hist(df, density=True, bins=4000)
## plt.xlim((min(data_flat), max(data_flat)))
#
#mean = np.mean(df)
#variance = np.var(df)
#sigma = np.sqrt(variance)
#x = np.linspace(3300, 3600, 1000)
#plt.plot(x, norm.pdf(x, mean, sigma))
#
#plt.show()


