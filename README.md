# Perception-Coding-Challenge
# answer.png![answer](https://github.com/kbalaji4/Perception-Coding-Challenge/assets/135673889/d1d29e1a-f3cd-47bb-ba48-651056d842e9)

# Methodolgy
So the task was to draw boundry lines on the image using the path represented by the cones. So to that I first had to filter out all but the cones in the image, and since the cones were a pretty bright shade of red, I created a mask that filitered out everything but that shade of red. So I was left with a rough outine of the cones and with that I was able to find the contours of the cones. And from there I found the centroid of all the cones as I needed points to draw a line. I also seperated the cones into two lists, one for the left side and another for the right so it would draw two lines. Then I found the line of best fit for the left and right cones respectivly, which I then extended, so they would intersect like the sample-answer. And finally I drew the line onto the image to create my answer.
# What I Tried
Since I am new to using opencv I tried a couple of different things to try and create a mask for the red cones. One thing I tried was to convert the image to the hsv colorspace and then try to create a mask that way. But when I converted the image into the hsv colorspace it was a bit more hard than I thought to find the right rgb value to isolate just cones, as the chiars and door kept on getting included as well. I tried to blur the image to get rid of some of the noise, but some of the cones got blurred out as well, so I decided just to create a mask oon the original image and that worked out pretty well. The other thing that took a couple of tries to get right was drawing the lines on the cones. Finding a line of best fit was not that hard, but extending the line was a bit tricky. With the first couple of tries extending the lines would cause it to shift off the cones entirely. Using the built in least squares method I was able to use the slope and intercept it gave me the create a pretty good line that stayed on the cones.
# Libraries Used
I used the cv2 and NumPy libraries for my code
