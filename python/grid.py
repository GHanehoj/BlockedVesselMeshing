"""
Grid class and subroutines to make life easier.
"""
class Grid:
    def __init__(self, values, dx, min):
        self.values = values
        self.dx = dx
        self.min = min
        self.dim = values.shape
        self.max = self.dim+self.min