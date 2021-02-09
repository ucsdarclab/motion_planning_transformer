''' Generate a map using matplotlib and save it.
'''
import numpy as np
import sys

import skimage.morphology as skim
from skimage import io
from skimage import color
import pickle

try:
    img = io.imread('map_1.png')
except FileNotFoundError:
    raise NameError("Did not find the file, please generate the map")

try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")

# All measurements are mentioned in meters
# Define global parameters
length = 24 # Size of the map
robot_radius = 0.1
dist_resl = 0.05

img_g = color.rgb2gray(color.rgba2rgb(img))
# Dilate image for collision checking
invert_img_g = np.abs(1-img_g)
invert_img_g_dilate = skim.dilation(invert_img_g, skim.disk(robot_radius/dist_resl))
img_g_dilate = np.abs(1-invert_img_g_dilate)



def geom2pix(pos, res=0.05, size=(480, 480)):
    """
    Convert geometrical position to pixel co-ordinates. The origin 
    is assumed to be at [image_size[0]-1, 0].
    :param pos: The (x,y) geometric co-ordinates.
    :param res: The distance represented by each pixel.
    :param size: The size of the map image
    :returns (int, int): The associated pixel co-ordinates.
    """
    return (np.int(size[0]-1-np.floor(pos[1]/res)), np.int(np.floor(pos[0]/res)))


class ValidityCheckerDistance(ob.StateValidityChecker):
    '''A class to check the validity of the state, by checking distance function
    '''
    
    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.STate object to be checked.
        :return bool: True if the state is valid.
        '''
        pix_dim = geom2pix(state)
        return img_g_dilate[pix_dim[0], pix_dim[1]]>0.5

# Define the space
space = ob.RealVectorStateSpace(2)

# Set the bounds 
bounds = ob.RealVectorBounds(2)
bounds.setLow(0)
bounds.setHigh(length)
space.setBounds(bounds)

# Define the SpaceInformation object.
si = ob.SpaceInformation(space)

# Validity checking
ValidityChecker_obj = ValidityCheckerDistance(si)
si.setStateValidityChecker(ValidityChecker_obj)

def get_path(start, goal):
    '''
    Get a RRT path from start and goal.
    :param start: og.State object.
    :param goal: og.State object.
    returns (np.array, np.array, success): A tuple of numpy arrays of a valid path,  
    interpolated path and whether the plan was successful or not.
    '''
    success = False
    # Create a simple setup
    ss = og.SimpleSetup(si)

    # Set the start and goal states:
    ss.setStartAndGoalStates(start, goal, 0.1)

    # # Use RRT*
    planner = og.RRTstar(si)

    ss.setPlanner(planner)

    # Attempt to solve within the given time
    time = 60
    solved = ss.solve(60.0)
    while not ss.haveExactSolutionPath():
        solved = ss.solve(30.0)
        time +=30
        if time>1200:
            break
    if ss.haveExactSolutionPath():
        success = True
        print("Found solution")
        path = [
            [ss.getSolutionPath().getState(i)[0], ss.getSolutionPath().getState(i)[1]]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
        # Define path
        ss.getSolutionPath().interpolate(100)
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i)[0], path_obj.getState(i)[1]] 
            for i in range(path_obj.getStateCount())
            ])
    else:
        path = [[start[0], start[1]], [goal[0], goal[1]]]
        path_interpolated = []

    return np.array(path), np.array(path_interpolated), success



def start_experiment_rrt(start, samples):
    '''
    Run the ccgp-mp experiment for random start and goal points.
    :param start: The start index of the samples
    :param samples: The number of samples to collect
    '''
    for i in range(start, start+samples):
        path_param = {}
        # Define the start and goal location
        start = ob.State(space)
        start.random()
        while not ValidityChecker_obj.isValid(start()):
            start.random()
        goal = ob.State(space)
        goal.random()
        while not ValidityChecker_obj.isValid(goal()):   
            goal.random()

        path, path_interpolated, success = get_path(start, goal)
        path_param['path'] = path
        path_param['path_interpolated'] = path_interpolated
        path_param['success'] = success

        pickle.dump(path_param, open('/root/global_planner_data/path_{}.p'.format(i), 'wb'))



if __name__ == "__main__":
    start, samples = int(sys.argv[1]), int(sys.argv[2])
    start_experiment_rrt(start, samples)