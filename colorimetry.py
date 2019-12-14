#! /usr/bin/env python
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

def XYZ2xy(color_coordinates):
    X = color_coordinates['X']
    Y = color_coordinates['Y']
    Z = color_coordinates['Z']
    
    xy_chromacity = {'x': X/(X + Y + Z), 'y': Y/(X + Y + Z)}
    
    return xy_chromacity              

def xy2uv(xy_chromacity):
    x = xy_chromacity['x']
    y = xy_chromacity['y']
    
    uv_chromacity = {'u': 4*x /(-2*x + 12*y + 3),
                     'v': 6*y /(-2*x + 12*y + 3)}
    
    return uv_chromacity
    

def XYZ(spd):
    xyz_cmf = pd.read_csv('ciexyz31_1.csv', index_col = 0)
    spd = spd[(spd.index >= 380) & (spd.index <= 780)]
    for i in ['X', 'Y', 'Z']:
        f = interpolate.interp1d(xyz_cmf.index, xyz_cmf[i])
        xyz_cmf[i] = f(spd.index)
    spectral_value = spd.columns[0]
    Xnm = spd[spectral_value] * xyz_cmf['X']
    Ynm = spd[spectral_value] * xyz_cmf['Y']
    Znm = spd[spectral_value] * xyz_cmf['Z']

    color_coordinates = {'X': np.trapz(Xnm.values, Xnm.index),
                         'Y': np.trapz(Ynm.values, Ynm.index),
                         'Z': np.trapz(Znm.values, Znm.index)}

    
    return color_coordinates

def XYZ2sRGB(xyz_color_coordinates):
    M = np.matrix([[3.2406, -1.5372, -0.4986],
                   [-0.9689, 1.8758,  0.0415],
                   [ 0.0557, -0.2040, 1.0570]])
    xyz_color_coordinates = np.matrix([[xyz_color_coordinates['X']],
                                       [xyz_color_coordinates['Y']],
                                       [xyz_color_coordinates['Z']]])
    
    rgb_color_coordinates = M * xyz_color_coordinates
    rgb_color_coordinates = np.array(rgb_color_coordinates).flatten()
    rgb_color_coordinates = [round(i, 3) for i in rgb_color_coordinates]
    rgb_color_coordinates = np.array(rgb_color_coordinates)
    
    if max(rgb_color_coordinates)>1:
        rgb_color_coordinates = rgb_color_coordinates/max(rgb_color_coordinates)
    Csrgb = {}
    for i, j in zip(rgb_color_coordinates, ['R', 'G', 'B']):
        if i <= 0.0031308:
            Csrgb[j] = i * 12.92
        else:
            Csrgb[j] = 1.055 * i**(1/2.4)-0.055
            
    return Csrgb

def colorviz(xyz_color_coordinates):
    plt.style.use('dark_background')
    M = np.matrix([[3.2406, -1.5372, -0.4986],
                   [-0.9689, 1.8758,  0.0415],
                   [ 0.0557, -0.2040, 1.0570]])
    xyz_color_coordinates = np.matrix([[xyz_color_coordinates['X']],
                                       [xyz_color_coordinates['Y']],
                                       [xyz_color_coordinates['Z']]])
    
    rgb_color_coordinates = M * xyz_color_coordinates
    rgb_color_coordinates = np.array(rgb_color_coordinates).flatten()
    rgb_color_coordinates = [round(i, 3) for i in rgb_color_coordinates]
    rgb_color_coordinates = np.array(rgb_color_coordinates)
    
    if max(rgb_color_coordinates)>1:
        rgb_color_coordinates = rgb_color_coordinates/max(rgb_color_coordinates)
    Csrgb = []
    for i in rgb_color_coordinates:
        if i <= 0.0031308:
            Csrgb.append(i * 12.92)
        else:
            Csrgb.append(1.055 * i**(1/2.4)-0.055)
    x = [-1, -1, 1, 1]
    y = [-1,  1, 1, -1]
    plt.figure(figsize=(10,10))
    plt.fill(x, y, color = (Csrgb[0], Csrgb[1], Csrgb[2]))    
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axis('off')
    
    
def plank(T, nm = np.arange(100, 6000, 1)):
    C1 = 3.741771e-16  # C1=2*pi*h*c^2, W/m^2
    C2 = 1.4388e-2     # ITS-90. C2=h*c/k, m*k
                   # c is the speed of light in vacuum,
                   # h is Planck’s constant,
                   # and k is the Boltzmann constant.

    M = C1*(nm*1e-9)**(-5)*(np.exp(C2/((nm*1e-9)*T))-1)**(-1) # W/(m^3)
    ref = pd.DataFrame({'W/m^3':M}, index = nm) # Спектр чёрного тела
    ref.index.name = 'nm'
    return ref

def mccamy(xy_chromacity):
    n = (xy_chromacity['x'] - 
         0.3320)/(xy_chromacity['y'] - 0.1858)
    T = - 449*n**3 + 3525*n**2 - 6823.3*n + 5530.33
    return T

def robertson(uv_chromacity):
    u = uv_chromacity['u']
    v = uv_chromacity['v']
    calc_table = pd.read_pickle('robertson_table')       
    calc_table['l'] = ((v - calc_table['v']) - calc_table['tg']*
                   (u - calc_table['u']))/np.sqrt(1 + calc_table['tg']**2)
    l = calc_table['l'].values
    r = l[:-1]/l[1:]
    i = np.argmin(r)
    Ti  = calc_table['mired'][i]
    Ti1 = calc_table['mired'][i + 1]
    Li  = calc_table['l'][i]
    Li1 = calc_table['l'][i + 1]
    
    mired = Ti + Li/(Li - Li1) * (Ti1 - Ti)
    T = 1e6/mired
    return T


if __name__ == '__main__':
    import sys
    fid = sys.argv[1]
    spd = pd.read_csv(fid, index_col = 0)
    color_coordinates = XYZ(spd)
    chromacity = XYZ2xy(color_coordinates)
    print("X = {0}, Y = {1}, Z = {2}".format(color_coordinates['X'],
                                             color_coordinates['Y'],
                                             color_coordinates['Z']))
    print("x = {0}, y = {1}".format(chromacity['x'], chromacity['y']))
