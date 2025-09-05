import cv2
import matplotlib.pyplot as plt
import numpy as np
import lms
import points
import ransac
import math

def largest_object(img):
    """
        Find the largest connected component in a binary image.

        Parameters
        ----------
        img (numpy.ndarray):
            Input binary image.

        Returns
        ----------
        numpy.ndarray:
            Binary image containing only the largest connected component.
    """
    num_labels, img_labels, stats, _ = cv2.connectedComponentsWithStats(img)
    areas = stats[1:, 4]
    index = np.argmax(areas, axis=0)
    img_largest_object = (img_labels == index + 1)
    img_largest_object = np.uint8(255 * img_largest_object)
    return img_largest_object


def centroid(img_bin):
    """
        Calculate the centroid coordinates of white (non-zero) pixels in a binary image.

        Iterates over each pixel in the binary image. For each white pixel (pixel value not zero), it accumulates the indices of rows and columns.
        If no white pixels are found, the function returns a default value. Otherwise, the centroid is computed as the average position of these white pixels.

        Parameters
        ----------
        img_bin (numpy.ndarray): A binary image represented as a 2D numpy array where white pixels are
                                 non-zero and black pixels are zero.

        Parameters
        ----------
        tuple or None: A tuple representing the centroid coordinates,
                       calculated as the integer division of the sum of pixel indices by the total number of white pixels.
                       Returns None if no white pixels are found.
    """

    rows, columns = img_bin.shape
    area = 0
    sum_i = 0
    sum_j = 0
    for i in range(rows):
        for j in range(columns):
            if img_bin[i, j]:
                area += 1
                sum_i += i
                sum_j += j
    if area == 0:
        return None
    return (sum_i // area, sum_j // area)


def radius_large(cont):
    """
        Calculate the radius of a circle fitted to a large contour.

        Parameters
        ----------
        cont (array_like):
            Array of contour points (x, y), used to fit a circle.

        Returns
        ----------
        int :
            The radius of the fitted circle, computed using `lms.circunf(cont)` that returns coefficients (A, B, C) for the circle equation: x^2 + y^2 + Ax + By + C = 0.

        Raises
        ----------
        ValueError: If the computed discriminant is negative, indicating an error in fitting or invalid data.
    """
    a, b, c = lms.circunf(cont)
    discriminant = a ** 2 + b ** 2 - 4 * c
    if discriminant < 0:
        raise ValueError("The discriminant is negative, suggesting an invalid or impossible circle fitting.")
    radius = int(np.sqrt(np.squeeze(discriminant)) / 2)
    return radius


def radius_small(cont):
    """
        Calculate the radius of a circle fitted to inner contour.

        Parameters
        ----------
        cont (array_like):
            Array of contour points (x, y), used to fit a circle.

        Returns
        ----------
        int :
            The radius of the fitted circle, computed using `ransac method` that returns coefficients (A, B, C)
            for the circle equation: x^2 + y^2 + Ax + By + C = 0.

        Raises
        ----------
        ValueError: If the computed discriminant is negative, indicating an error in fitting or invalid data.
    """
    inliers = ransac.ransac_circunf(cont, 50)
    a, b, c = lms.circunf(inliers)
    discriminant = a ** 2 + b ** 2 - 4 * c
    if discriminant < 0:
        raise ValueError("The discriminant is negative, suggesting an invalid or impossible circle fitting.")
    radius = int(np.sqrt(np.squeeze(discriminant)) / 2)
    return radius


def calculate_deviation_angle(line_start, line_end):
    """
    Calculate the deviation angle of a line from a vertical reference, in degrees.

    This function computes the angle between a line (defined by start and end points) and a vertical line passing through a given x-coordinate.
    The deviation angle is positive if the line deviates to the right of the vertical line and negative if it deviates to the left.

    Parameters
    ----------
    line_start : tuple of (int, int)
        The starting point (x1, y1) of the line.
    line_end : tuple of (int, int)
        The ending point (x2, y2) of the line.
    vertical_x : int
        The x-coordinate of the vertical line used as the reference for deviation.

    Returns
    -------
    float
        The deviation angle in degrees from the vertical line. Positive values indicate a deviation
        to the right, while negative values indicate a deviation to the left.
    """
    x1, y1 = line_start
    x2, y2 = line_end
    vertical_x = x1
    slope = (y2 - y1) / (x2 - x1)
    deviation_angle = math.degrees(math.atan(slope))
    deviation = vertical_x - x2
    if deviation > 0:
        deviation_angle -= 90.0
    else:
        deviation_angle += 90.0
    angle_text = f"angle: {round(deviation_angle, 2)}"
    create_text(img, angle_text, 20, 63)


def number_of_teeth(teeth_contours):
    """
       Count the number of teeth contours in the provided image and display the count as text.

       Takes the contours of teeth as input and uses 'findContours' function to identify the external contours,
       which represent the teeth. It then counts the number of contours and displays this count as text on the image.

       Parameters
       ----------
       teeth_contours : numpy.ndarray
           A binary image containing the contours of teeth. Each contour represents an individual tooth.

       Returns
       -------
       None
           The function does not return any value. Instead, it displays the number of teeth on the provided image.

       Raises
       ------
       ValueError
           If the provided teeth contours are not in the expected format or if the findContours function fails to find contours.
    """
    number_of_teeth, hierarchy = cv2.findContours(teeth_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if number_of_teeth is None:
        raise ValueError("Error while finding the teeth.")
    teeth_text = f"number of teeth: {len(number_of_teeth)}"
    create_text(img, teeth_text, 370, 30)


def depth(a, b, c):
    """
        Calculate and display the perpendicular depth from a geometric line to a specified point.

        Parameters
        ----------
        a : float
            The coefficient A in the line equation Ax + By + C = 0.
        b : float
            The coefficient B in the line equation Ax + By + C = 0.
        c : float
            The constant term C in the line equation Ax + By + C = 0.
        center : tuple of (int, int)
            x and y coordinates (x1, y1) of the point from which the depth to the line is measured.
        img : numpy.ndarray
            The image on which the depth text will be displayed.

        Returns
        -------
        float
            The calculated depth from the point to the line. This value is also displayed on the image.
    """
    norm_ab = math.sqrt(a ** 2 + b ** 2)
    depth = abs(a * center[1] + b * center[0] + c) / norm_ab
    depth_text = f"depth: {depth:.2f}"
    create_text(img, depth_text, 20, 95)


def create_text(img, text, x, y):
    """
        Draw text on an image at a specified location.

        Parameters
        ----------
        img : numpy.ndarray
            The image on which the text is to be drawn.
        text : str
            The string of text to be drawn on the image.
        x : int
            The x-coordinate on the image where the text will start.
        y : int
            The y-coordinate on the image where the text will start.

        Returns
        -------
        None
            The function does not return any value. It modifies the input image in-place.
    """
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)


# reading the image
img = cv2.imread("gear6.png", 0)
if img is not None:
    img_edges = cv2.Canny(img, 250, 500)
else:
    print("Error: Unable to load the image")

# largest contour
img_largest_object = largest_object(img_edges)
# inner contour
img_int_cont = img_edges - img_largest_object

# computing centroid of the large contour
center = centroid(img_largest_object)

# computing large radius
ext_cont = points.get_contour_points(img_largest_object)
radius_ext = radius_large(ext_cont)

# computing number of teeth
teeth_contours = cv2.circle(img_largest_object, (center[1], center[0]), radius_ext + 10, 0, -1)
number_of_teeth(teeth_contours)

# computing small radius
img_int_cont = largest_object(img_int_cont)
int_cont = points.get_contour_points(img_int_cont)
radius_int = radius_small(int_cont)
radius_text = f"radius: {radius_int}"
create_text(img, radius_text, 20, 30)

# line of the inner contour
cv2.circle(img_int_cont, (center[1], center[0]), radius_int, 0, 10)

# width
line_points = points.get_contour_points(img_int_cont)
P1 = line_points[0][0]
P2 = line_points[len(line_points) - 1][0]
width = round((np.sqrt((P2[0] - P1[0])**2 + (P2[1] - P1[1])**2)), 2)
width_text = f"width: {width}"
create_text(img, width_text, 20, 125)

# depth
[A, B, C] = lms.recta(line_points)
depth(A, B, C)

#line
line_center = centroid(img_int_cont)
cv2.line(img, (center[1], center[0]), (line_center[1], line_center[0]), 0, 1)

# angle
calculate_deviation_angle((center[1], center[0]), (line_center[1], line_center[0]))

# drawing external contour
cv2.circle(img, (center[1], center[0]), radius_ext, 0, 1)
# drawing internal contour
cv2.circle(img, (center[1], center[0]), radius_int, 0, 1)
# drawing vertical line from the center to the top
cv2.line(img, (center[1], center[0]), (center[1], 0), 0, 1)

# displaying the image
plt.imshow(img, 'gray')
plt.scatter(center[1], center[0], s=10, c='red')
plt.axis('off')
plt.show()
