#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 11:54:02 2019

@author: ShonaCW
"""
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.colors import LogNorm


hdulist = fits.open("/Users/ShonaCW/Desktop/YEAR 3/Labs/Astro/A1_mosaic.fits")
print(hdulist[0].header)
#print(hdulist.info())
print(type(hdulist))
image_data = hdulist[0].data
#plt.imshow(image_data, cmap='gray', norm=LogNorm())
#plt.colorbar()

#Getting some basic info about our image
print('Min:', np.min(image_data))
print('Max:', np.max(image_data)) #there is a pixel which maxes out the 16 bits available for a single pixel
print('Mean:', np.mean(image_data))
print('Stdev:', np.std(image_data))
NBINS = 1000
#histogram = plt.hist(image_data.flatten(), NBINS, density=True, range = [3300, 3600])

"""

i,j = np.unravel_index(image_data.argmax(), image_data.shape)
print('max index', i, j)
print('max', image_data[657][2530])

#image_data[i][j].masked

from tqdm import tqdm
from astropy.io import fits
from astropy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm

mos = fits.open(path_mos)
#  Header:
data = mos[0].data

msk = np.zeros([len(data[0]), len(data)])

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
for n_sig in range(1, 7):
    # plt.hist(data_flat, bins=400, density=True, range=[3300, 3600])
    df = stats.sigma_clip(data, sigma=n_sig)
    mean_2 = np.mean(df)
    variance_2 = np.var(df)
    sigma_2 = np.sqrt(variance_2)
    print(f'Clipped to {n_sig} sigma: Mean = {mean_2}, std = {sigma_2}')
mos.close()
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
"""
#---------------------------------------------------------------------------
from tqdm import tqdm
from astropy.io import fits
from astropy import stats
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
from scipy.stats import norm
import numpy.ma as ma
import math


path_mos = '/Users/ShonaCW/Desktop/YEAR 3/Labs/Astro/A1_mosaic.fits'
mos = fits.open(path_mos)
#  Header:
data = mos[0].data
mos.close()
data_ma = ma.masked_array(data)

msk = np.zeros([len(data[0]), len(data)])

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
    x, y = np.where((x_[:,np.newaxis] - x0)**2 + (y_ - y0)**2 <= radius**2) #Returns array with elements depending on condition.
    # x, y = np.where((np.hypot((x_-x0)[:,np.newaxis], y_-y0)<= radius)) # alternative implementation
    for x, y in zip(x_[x], y_[y]):
        yield x, y


def create_test_image(size=[100, 100], std=20, centre=3421):
    gauss = np.random.normal(loc=centre, size=size, scale=std).astype(int)
    star_coords = lsit(points_in_circle_np(10, x0=50, y0=50))
    for pt in star_coords: 
        
        gauss[pt[0], pt[1]] = 3500 
    return gauss


def clip():
    for n_sig in range(1, 7):
        # plt.hist(data_flat, bins=400, density=True, range=[3300, 3600])
        df = stats.sigma_clip(data, sigma=n_sig)
        mean_2 = np.mean(df)
        variance_2 = np.var(df)
        sigma_2 = np.sqrt(variance_2)
        print(f'Clipped to {n_sig} sigma: Mean = {mean_2}, std = {sigma_2}')
    mos.close()
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
    plt.imshow(arr)
    plt.colorbar()
    return


def extend(centre, data, mean_noise, noise_std, plot=True):
    vals = [] #will contain the mean count at each radius around object
    r = 1 #initialise radius 
    x0, y0 = centre[0], centre[1]
    vals.append(data[x0][y0]) #append count of center point
    pts = list(points_in_circle_np(r, x0=x0, y0=y0))
    r += 1
    pts2 = list(points_in_circle_np(r, x0=x0, y0=y0))
    diff_pts = list(set(tuple(i) for i in pts2) - set(tuple(i) for i in pts))
    
    flux_diff = sum([data[x, y] for x, y in diff_pts])
    noise_lim = len(diff_pts) * (mean_noise + noise_std)
    # pdb.set_trace()
    while flux_diff > noise_lim:
        pts = pts2
        r += 1
        pts2 = list(points_in_circle_np(r, x0=x0, y0=y0))
        diff_pts = list(set(tuple(i) for i in pts2) - set(tuple(i) for i in pts))
        vals.append(sum([data[x, y] for x, y in diff_pts]) /len(diff_pts))
        noise_lim = len(diff_pts) * (mean_noise + noise_std)
        #print(flux_diff, noise_lim)
    print('vals final', vals)
    print('accessing', vals[0])
    vals_arr = np.asarray(vals)
    if plot==True:
        plt.hist(vals_arr)

        
    return pts, r, vals[0]


def find_max_pxl(img):
    return np.unravel_index(img.argmax(), img.shape)

def Cal_Mag(counts):
    """
    This function converts the number count to a calibrated magnitude.
    """
    ZP = 2.530E+01
    m = ZP - (2.5 * math.log10(counts))
    return m

#def distinguish():


print('list of counts around certain object', extend(centre, img, noise_local, sigma_noise)[2])   


#img = ma.masked_array(create_test_image())
img = ma.masked_array(data[3840:3853, 2260:2300])
plot_2d(img)
centre = find_max_pxl(img)
df = stats.sigma_clip(img, sigma=5)
noise_local = np.mean(df)
variance = np.var(df)
sigma_noise = np.sqrt(variance)


points_to_mask = extend(centre, img, noise_local, sigma_noise)[0]
Count = 0
for pt in points_to_mask:
    i = pt[0]
    j = pt[1]
    img[i, j] = ma.masked
    Count+= data[i][j] #original data - need clipped version
   
print('---------------Count per object-------------------')
print('COUNT', Count)
print('MAG FROM COUNT', Cal_Mag(Count))
    
plot_2d(img)

#-----------------------------------------------------------------------------
print()
print("--------------------------------------------------")
print()

def edges():
    #initial dimensions chosen so inside&out have roughly the same area
    #x index starting points (from photo dimensions)
    #a = 2
    a = 632
    b = 1287
    
    #y index starting points (from photo dimensions)
    #c = 2
    c = 1523
    d = 3458
    
    #clip the masked data in order to rid of brightest/ darkest noise
    data_msd = stats.sigma_clip(data_ma, sigma=5, masked=True)
    data_whole = stats.sigma_clip(data, sigma=5, masked=True)
    
    #mask all pixels within the parameters 
    data_msd[a:b, c:d] = ma.masked
    #calculate standard deviation of inside of masked area (data is original set)
    std_dev_in = np.std(data_whole[a:b, c:d])
    #calculate standard deviation of area outside of mask
    std_dev_out = np.std(data_msd)
    
    #initialise loop by calculating 15% of the inner-area std_dev
    print('STD DEV IN', std_dev_in)
    print('STD DEV OUT', std_dev_out)
    prc = (1.15 * std_dev_in) #- std_dev_in
    print('PRC', prc)
    #once outside is 15 percent greater than std of inside, break loop
    while std_dev_out > prc:
        a -= 1
        b += 1
        c -= 1
        d += 1
        
        if a==0 or c==0 or b==2570 or d==4611: #the limiting pixels 
            raise ValueError('No convergence of standard deviation.')
        
        data_msd[a:b, c:d] = ma.masked #update mask
        std_dev_in = np.std(data_whole[a:b, c:d])
        std_dev_out = np.std(data_msd)
        prc = 1.15 * std_dev_in 
    
    print("Final x width is: ", a, ". Final y width is: ", c)
    print("Final standard dev is:", std_dev_out)

    return

print(edges())
