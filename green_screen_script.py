import cv2 
import matplotlib.pyplot as plt

#Loading the screen 
img = cv2.imread('GreenScreen.jpg')

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Define the lower and upper bounds of the green color
lower_green = (35, 25, 25)
upper_green = (90, 255, 255)

# Threshold the image to extract the green color
mask = cv2.inRange(hsv_image, lower_green, upper_green)
green_image = cv2.bitwise_and(img, img, mask=mask)

# Find contours in the mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
cv2.drawContours(img, contours, -1, (265, 165, 0), 3)

# Display the original image with contour and green images

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.show()
plt.subplot(1, 2, 2)
plt.imshow(green_image)
plt.show()

