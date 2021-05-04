# This code is derived from the following repo:
# https://github.com/scipython/scipython-maths/tree/master/maze

import random
import matplotlib.pyplot as plt
from matplotlib import patches
class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}
        
    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False
        
    def __str__(self):
        ''' Display the values of the cell'''
        walls_pic = ''
        if self.walls['N']:
            walls_pic+= ' -- \n'
        else:
            walls_pic+= '   \n'
        if self.walls['W']:
            walls_pic+='|  '
        else:
            walls_pic+='   '
        if self.walls['E']:
            walls_pic+='|\n'
        else:
            walls_pic+=' \n'
        if self.walls['S']:
            walls_pic+= ' -- '
        return f'x:{self.x}, y:{self.y}, \n{walls_pic}'

class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, nx, ny, ix=0, iy=0):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).
        :param nx, ny: The number of cells on the x and y axis.
        :param ix, iy: Initial point of the  maze.
        :param pw: The width of the passage.
        :param wt: The thickness of the wall
        """

        self.nx, self.ny = nx, ny
        self.ix, self.iy = ix, iy
        self.maze_map = [[Cell(x, y) for y in range(ny)] for x in range(nx)]

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.maze_map[x][y]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = ['-' * self.nx * 2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)


    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)

            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1
            

def generate_random_maze(length, wt=0.375, pw=1.2, dist_resl=0.05,seed=1, fileName=None):
    '''
    Generate random square mazes
    :param length: The length of the whole map
    :param wt: The thickness of the wall (in meters)
    :param pw: The width of the passage (in meters)
    :param dist_resl: distance (in meters) per pixels
    :param seed: The seed value used for generating the map
    :param fileName: The location to save the map
    '''
    
    ch = cw = wt*2 + pw
    nx = ny = int((length-wt)/(pw+wt))

    random.seed(seed)
    ix, iy = random.randint(0, nx-1),  random.randint(0, ny-1)

    maze = Maze(nx, ny, ix, iy)
    maze.make_maze()

    # Save map
    size = length*100/2.54 #In inches
    dpi = 0.0254/dist_resl # dots per inches, convert the resolution to inches

    fig, ax = plt.subplots(figsize=(size, size), dpi=dpi)

    # NOTE: This is important to ensure that the loaded map, matches the set
    # distance resolution.
    plt.subplots_adjust(0, 0, 1, 1) # Removes all spacings on either sides of the map
    # Plot patches with north/south/west/east patch
    def plot_cell(ax, cell):
        '''
        Plot the walls for the given cell
        '''
        x, y = cell.x*(cw-wt), (ny-cell.y)*(ch-wt)+wt
        Walls = {
            'N':patches.Rectangle([x, y-wt], width=cw, height=wt, color='k', linewidth=0),
            'S':patches.Rectangle([x, y-ch], width=cw, height=wt, color='k', linewidth=0),
            'W':patches.Rectangle([x, y-ch], width=wt,height=ch, color='k', linewidth=0),
            'E':patches.Rectangle([x+cw-wt, y-ch], width=wt, height=ch, color='k', linewidth=0)
        }
        for wall,value in cell.walls.items():
            if value:
                ax.add_patch(Walls[wall])
    
    for cell_r in maze.maze_map:    
        for cell in cell_r:
            plot_cell(ax, cell)

    ax.set_xticks([])
    ax.set_yticks([])
    
    ax.set_xlim([0, length])
    ax.set_ylim([0, length])

    if fileName is None:
        fig.savefig('map_maze_temp.png', pad_inches=0.0, bbox_inches='tight')
    else:
        fig.savefig(fileName, pad_inches=0.0, bbox_inches='tight')

        