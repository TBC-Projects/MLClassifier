import cv2

# Load an image
image = cv2.imread('taco_bell_sauce.png')

# Display the image
cv2.imshow('Image Window', image)

# Wait for a key press and close the window
cv2.waitKey(0)
cv2.destroyAllWindows()