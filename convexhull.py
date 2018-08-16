import matplotlib.pyplot as plt
import numpy as np
import time

class Line:
    def __init__(self, point1, point2):
        """A simple linear line"""
        self.point1 = point1
        self.point2 = point2
        # work out y = mx + c
        self.x1, self.y1 = point1
        self.x2, self.y2 = point2
        self.m = (self.y1-self.y2) / (self.x1-self.x2)
        self.c = self.y1 - self.m * self.x1
    
    def distance_from(self, x, y):
        """Work out the distance of the y co-ordinates for simplicity"""
        line_y = self.m*x + self.c
        return y - line_y
    
    def intersects(self, line):
        """Work out whether this line intersects another line"""
        if self.m == line.m or np.any(np.in1d(self.points, line.points)):
            return False # If the gradients are the same or the line edges are the same
        A = np.array([[-self.m, 1], [-line.m, 1]])
        b = np.array([self.c, line.c])
        x, y = np.linalg.solve(A, b) # Solve for x and y


        # If the lines intersect then the intersection point should be in range of the line's y co-ordinates
        top_point_y = max(self.points, key=lambda x: x[1])[1]
        bot_point_y = min(self.points, key=lambda x: x[1])[1]

        return bot_point_y < y < top_point_y
    
    def plot(self, **kwargs):
        """A method to plot the line for simplicity"""
        import matplotlib.pyplot as plt
        plt.plot([self.x1, self.x2], [self.y1, self.y2], **kwargs)

    @property
    def midpoint(self):
        """Returns the midpoint of the line"""
        xs = np.array([self.x1, self.x2])
        ys = np.array([self.y1, self.y2])
        return np.array([np.mean(xs), np.mean(ys)])

    @property
    def points(self):
        """Returns both of the points in a numpy array"""
        return np.array([self.point1, self.point2])


class ConvexPolygon:
    """A polygon object which may only be convex"""
    def __init__(self, *lines):
        self.lines = lines
        self.x = True
        self.vertices = np.unique([np.array(p) for l in self.lines for p in l.points], axis=0)
        
    def surrounds(self, point):
        """Check if the polygon surrounds a given point"""
        point = np.array(point)
        furthest_vertex = max(self.vertices, key=lambda x: np.linalg.norm(x-point))
        line = Line(furthest_vertex, point)
        return not any(line.intersects(x) for x in self.lines)

    def plot(self):
        """Plot the polygon for simplicity"""
        import matplotlib.pyplot as plt
        for l in self.lines:
            l.plot(color='r')
    
    @property
    def sides(self):
        """Returns the number of sides the polygon has"""
        return len(self.lines)

def convex_hull(number_of_points, plot=True):
    start_time = time.time()

    points = np.random.randn(number_of_points*2).reshape(number_of_points,2) # Generate 250 random co-ordinates
    
    if plot:
        plt.scatter(points.T[0], points.T[1], s=10) # Display all the points

    min_x = points[np.argmin(points.T[0])] # Find the co-ordinates with the smallest and largest x value
    max_x = points[np.argmax(points.T[0])] # These are guaranteed to be on the convex polygon so it's a good starting point
    line = Line(min_x, max_x) # Create a line which joins both the coords up
    max_y = max(points, key=lambda x: line.distance_from(*x)) # Find the co-ordinates which are furthest away from the line
    min_y = min(points, key=lambda x: line.distance_from(*x)) # These will also be on the polygon, now we have 4 lines for a 4 sided polygon

    # Collect all the lines and create a polygon object from them
    lines = [Line(min_x, max_y), Line(max_y, max_x), Line(max_x, min_y), Line(min_y, min_x)]
    poly = ConvexPolygon(*lines)

    def points_outside(polygon):
        """Used to filter through a numpy array to get only the points which lie outside the polygon"""
        return np.invert(np.apply_along_axis(polygon.surrounds, 1, points))

    def direct_point_function(polygon, line):
        """Used to filter through a numpy array to get only the points which are in direct view of a side of a polygon"""
        check_list = list(polygon.lines)
        check_list.remove(line)
        def is_direct_point(point):
            l = Line(point, line.midpoint)
            return not any(l.intersects(test_line) for test_line in check_list)
        return is_direct_point

    points = points[points_outside(poly)] # Get all points outside the polygon
    while points.size > 0: # Will remain true until there are no points outside the polygon
        new_polygon_sides = [] # will contain the sides for the newly generated polygon
        for line in poly.lines:
            # Get all points which are in direct view of the polygon's side
            is_direct_point = direct_point_function(poly, line)
            x = np.apply_along_axis(is_direct_point, 1, points)
            line_points = points[x]
            
            if not line_points.size: # This will be true if there are no points in direct view of this line
                new_polygon_sides.append(line)
                continue
            
            furthest_point = max(line_points, key=lambda x: abs(line.distance_from(*x))) # Furthest point from the line
            new_polygon_sides.append(Line(line.point1, furthest_point))
            new_polygon_sides.append(Line(line.point2, furthest_point))
        poly = ConvexPolygon(*new_polygon_sides)
        points = points[points_outside(poly)]

    time_taken = time.time() - start_time

    print("Generated a {} sided convex hull to contain {} points".format(poly.sides, number_of_points))
    print("Time taken: {}s".format(time_taken))
    
    if plot:
        poly.plot()
        plt.show()
    
    return time_taken