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
import pandas as pd
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable


hdulist = fits.open("/Users/ShonaCW/Desktop/YEAR 3/Labs/Astro/A1_mosaic.fits")
print(hdulist[0].header)
#print(hdulist.info())
print(type(hdulist))
image_data = hdulist[0].data
plt.imshow(image_data, cmap='gray', norm=LogNorm())
plt.colorbar()

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


def points_in_circle_np(radius, x0=0, y0=0 ):
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

def Cal_Mag(counts):
    """
    This function converts the number count to a calibrated magnitude.
    """
    ZP = 2.530E+01
    m = ZP - (2.5 * math.log10(counts))
    return m

#%%
def extend_Shona(centre, data, mean_noise, noise_std, plot=False):
    """
    Returns a list of points within 
    """
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
    #print('vals final', vals)
    #print('accessing', vals[0])
    
    if plot==True:
        x_ax = []
        for i in range(len(vals)):
            x_ax.append(i)
        plt.figure(5)
        plt.xlabel('Distance from Object Center (pixel number')
        plt.ylabel('Count')
        plt.bar(x_ax, vals)
        
        plt.figure(6)
        y = []
        plt.xlabel('Distance from Object Center (pixel number')
        plt.ylabel('Calibrated Magnitude')
        vals.reverse()
        print([Cal_Mag(count) for count in vals])
        y.extend([Cal_Mag(count) for count in vals])
        plt.bar(x_ax, y)
    
    #at this point we have radius of star in r, and list of points within the
    #circular area of the star in pts. going to add to pts and work with r.
    r = r_circ
    
    rect = []
    pts_s = []
    pts2_s = []
    pts_s.append(data[x0][y0])
    pts2_s.ppend(data[x0][y0 + 1])
    flux_diff_s = pts2_s - pts_s
    
    while flux_diff_s > noise_lim:
        pts_s = data[x0][y0 + r]
        r_circ = r_rec
        r_rec += 1
        pts2_s = data[x0][y0 + r]
        flux_diff_s = pts2_s - pts_s
        noise_lim = len(diff_pts_s) * (mean_noise + noise_std)
    
    
    return pts, r_circ, r_rec, vals[0]


def find_max_pxl(img):
    return np.unravel_index(img.argmax(), img.shape)

#def distinguish():

 


#img = ma.masked_array(create_test_image())
img = ma.masked_array(data[747:816, 3190:3440])

plot_2d(img)
centre = find_max_pxl(img)
df = stats.sigma_clip(img, sigma=5)
noise_local = np.mean(df)
variance = np.var(df)
sigma_noise = np.sqrt(variance)


points_to_mask = extend_Shona(centre, img, noise_local, sigma_noise, plot=True)[0] #note also setting plot to true
Count = 0
for pt in points_to_mask:
    i = pt[0]
    j = pt[1]
    img[i, j] = ma.masked
    Count+= data[i][j] #original data - need clipped version
   
print('---------------Count per object-------------------')
print('COUNT', Count)
print('MAG FROM COUNT', Cal_Mag(Count))
    
#plot_2d(img)
#%%
img = ma.masked_array(data[3440:3190, 747:816])
plt.imshow(img, cmap='gray', norm=LogNorm())

#%%
#-----------------------------------------------------------------------------
print()
print("--------------------------------------------------")
print()

def edges():
    x =  np.linspace(632, 0, 80) #width of horizontal thickness 
    #initial dimensions chosen so inside&out have roughly the same area
    #x index starting points (from photo dimensions)
    a = 632
    b = 1938
    
    #y index starting points (from photo dimensions)
    c = 632
    d = 3979
    
    #clip the masked data in order to rid of brightest/ darkest noise
    #data_msd = stats.sigma_clip(data_ma, masked=True)
    
    #data_whole = stats.sigma_clip(data, masked=True)
    #data_whole = stats.sigma_clip(data, sigma=5, masked=True)
    
    #data_msd= data_ma
    data_whole= data
    
    
    #mask all pixels within the parameters 
    #data_msd[a:b, c:d] = ma.masked
    #calculate standard deviation of inside of masked area (data is original set)
    std_dev_in = []
    std_dev_in.append(np.std(data_whole[a:b, c:d]))
    #calculate standard deviation of area outside of mask
    #std_dev_out = []
    #std_dev_out.append(np.std(data_msd))
    
    print('STD DEV IN', std_dev_in)
    #print('STD DEV OUT', std_dev_out)
    #prc = (1.15 * std_dev_in) #- std_dev_in
    #once outside is 15 percent greater than std of inside, break loop
    #while std_dev_out > prc:
    for i in range(79):
        a -= 8
        b += 8
        c -= 8
        d += 8
        
        #if a==0 or c==0 or b==2570 or d==4611: #the limiting pixels 
        #raise ValueError('No convergence of standard deviation.')
        
        #data_msd[a:b, c:d] = ma.masked #update mask
        std_dev_in.append(np.std(data_whole[a:b, c:d]))
        #std_dev_out.append(np.std(data_msd))
        #prc = 1.15 * std_dev_in
    
    #Now just plotting a line to show where we are    
    #y_1 = np.linspace(1640, 1710, 3)
    y_1 = [1640, 1720]
    x_1 = [143.97, 143.97]
    #for i in range(3):
    #    x_1.append(143.97)
    
    plt.figure(7)
    plt.xlim(632, 0)
    plt.title('Standard Deviation vs. Frame Width')
    plt.grid()
    plt.xlabel('Width of Frame (pixels)')
    plt.ylabel('Standard Deviation (counts)')
    plt.plot(x, std_dev_in) # label=f'sigma = {sig}'
    plt.plot(x_1, y_1, ls='dashed', linewidth= 3, label=f'Cut-Off Diameter: 144 pixels')
    #plt.plot(x, std_dev_out, label='Std_Dev Out')
    plt.legend()
    
    plt.show
    
    #print(len(x))
    #print(len(std_dev_in))
    #print(len(std_dev_out))

    return

print(edges())
#for i in range(6):
#    print(edges(i))

#%%

"""SPATIAL DENSITY PLOT CODE"""
data = pd.read_csv(r'Desktop/catalogue_8.txt', skiprows=1, sep="\t")
#data.columns["Number", "Central x", "Central y", "noise_local", "r_apt", 	"r_star", "flux_obj	", "noise_flux", "flux_obj_clean"]
#data.columns = ["Number", "Central x", "Central y", "Highest Count", "Local Background Noise", "Radius Aperture", "Radius Object", "Flux"]
x = data["centre_x"]
y = data["centre_y"]
#x = data["Central x"]
#y = data["Central y"]
#xy = np.vstack([x, y])
#z = gaussian_kde(xy)(xy)
#idx = z.argsort()
#x, y, z = x[idx], y[idx], z[idx]
#plt.hist2d(data["Central y"], data["Central x"], c=z)
#plt.show()
#fig, ax = plt.subplots()
#ax.scatter(x, y, c=z, s=50, edgecolor='')
#plt.show()

#h =plt.hist2d(x, y, normed = True)
#plt.colorbar(h[3]);


#https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.gaussian_kde.html
xmin = x.min()
print(xmin)
xmax = x.max()
print(xmax)
ymin = y.min()
ymax = y.max()

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([X.ravel(), Y.ravel()])
values = np.vstack([x, y])
kernel = gaussian_kde(values)
Z = np.reshape(kernel(positions).T, X.shape)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
im = ax.imshow(np.rot90(Z), cmap=plt.cm.hot,
          extent=[xmin, xmax, ymin, ymax])
ax.plot(x, y, 'k.', markersize=2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("2D Spatial Density Plot")
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
fig.colorbar(im, cax=cax)
plt.show()

#%%
