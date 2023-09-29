import cv2
import numpy as np

# Load the image
image = cv2.imread("red.png")

# lower and upper bounds for cone color
lower_red = np.array([0, 0, 170])
upper_red = np.array([50, 50, 180])
# creates a mask using the bounds that shows just the cones
mask = cv2.inRange(image, lower_red, upper_red)
# Find contours in the mask
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the centroids of the red cones
cone_centroids = []
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cone_centroids.append((cX, cY))

# Separate the cones into right and left based on their X-coordinate
right_cones = []
left_cones = []

# Sorts left and right cones
for centroid in cone_centroids:
    if centroid[0] > image.shape[1] / 2:
        right_cones.append(centroid)
    else:
        left_cones.append(centroid)

# Sort the cones by x-coordinate to ensure they are in order
right_cones.sort(key=lambda x: x[1])
left_cones.sort(key=lambda x: x[1])


# Function that draws a line of best fit on the cones
def find_cones_line(cones):
    # Extract X and Y coordinates of cone centroids
    cone_x = [point[0] for point in cones]
    cone_y = [point[1] for point in cones]

    # Fit a linear regression line using the least-squares method
    A = np.vstack([cone_x, np.ones(len(cone_x))]).T
    m, b = np.linalg.lstsq(A, cone_y, rcond=None)[0]

    # Calculate the extended line coordinates
    x1 = min(cone_x) - 50
    x2 = max(cone_y) + 50
    y1 = int(m * x1 + b)
    y2 = int(m * x2 + b)
    # Ensure the line extends to the image boundaries
    if x1 > 0:
        x1 = 0
        y1 = int(b)
    if x2 < image.shape[1]:
        x2 = image.shape[1] - 1
        y2 = int(m * x2 + b)
    # Returns the start and endpoints of the line
    return (x1, y1), (x2, y2)


# Gets start and end points for the lines
start_pointL, end_pointL = find_cones_line(left_cones)
start_pointR, end_pointR = find_cones_line(right_cones)

# Draw the line on the left and right cones
cv2.line(image, start_pointL, end_pointL, (0, 0, 255), 4)
cv2.line(image, start_pointR, end_pointR, (0, 0, 255), 4)

# Save the result as "answer.png"
cv2.imwrite("answer.png", image)

# Displays the image
cv2.imshow("answer", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
