''' Generate random maps for the 2D planner.
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as mattrans

def generate_random_maps(
    width, 
    height=None, 
    dist_resl=0.05,
    num_circle=70,
    num_box=30,
    seed=1,
    fileName=None
    ):
    '''
    Randomly generate maps with the given width and height, resolution and
    obstacles.
    :param width: The width of the map in meters.
    :param breath: TODO.... The height of the map.
    :param dist_resl: distance(in meters) per pixels
    :param num_circle: Number of circular objects.
    :param num_box: Number of square objects.
    :param seed: The random seed value for random number generator.
    :param fileName: File name to save the map, if None the map is saved as 'map_temp.png'
    '''
    if height is None:
        height = width
    box_width, box_width = 1.5, 1.5
    cir_radius = 0.75

    np.random.seed(seed)

    xy_circle = np.c_[np.random.rand(num_circle)*(width-1)+0.5,  np.random.rand(num_circle)*(height-1)+0.5]
    xy_box = np.c_[np.random.rand(num_box)*(width-1)+0.5, np.random.rand(num_box)*(height-1)+0.5]
    xy_theta = np.random.rand(num_box)*np.pi*2


    sizeW = width*100/2.54 #In inches
    sizeH = height*100/2.54 #In inches
    dpi = 0.0254/dist_resl # dots per inches, convert the resolution to inches

    fig, ax = plt.subplots(figsize=(sizeW, sizeH), dpi=dpi)
    # NOTE: This is important to ensure that the loaded map, matches the set
    # distance resolution.
    plt.subplots_adjust(0, 0, 1, 1) # Removes all spacings on either sides of the map

    # Initialize the position of obstacles
    dimensions = [box_width, box_width]
    rectangle_corner = np.r_[(-dimensions[0]/2, -dimensions[1]/2)]
    
    plt_boundary = plt.Rectangle((0,0), width, height, color='k', fill=None, lw=2, alpha=1,)
    ax.add_patch(plt_boundary)
        
    for xy_i in  xy_circle:
        plt_cir = plt.Circle(xy_i, radius=cir_radius, color='k')
        ax.add_patch(plt_cir)

    for xy_i, theta_i in zip(xy_box, xy_theta):
        rotate = np.rad2deg(theta_i)
        T = mattrans.Affine2D().rotate_deg_around(*(xy_i[0],xy_i[1]), rotate) + ax.transData
        plt_box = plt.Rectangle(xy_i+rectangle_corner, dimensions[0], dimensions[1], color='k', transform=T)
        ax.add_patch(plt_box)        
    
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlim([0, width])
    ax.set_ylim([0, height])
    
    if fileName is None:
        fig.savefig('map_temp.png', pad_inches=0.0, bbox_inches='tight')
    else:
        fig.savefig(fileName, pad_inches=0.0, bbox_inches='tight')
    