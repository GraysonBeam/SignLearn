import os

import cv2

# Open the default camera
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

label = input("Enter the label for this frame: ")

if not os.path.exists(f"SignLearn/data/{label}"):
    os.makedirs(f"SignLearn/data/{label}")

count = len(os.listdir(f"SignLearn/data/{label}")) + 1

while True:
    ret, frame = cam.read()

    # Display the captured frame
    cv2.imshow("Camera", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord("q"):
        break

    if cv2.waitKey(1) == ord("s"):
        # Save the current frame as an image file
        cv2.imwrite(f"SignLearn/data/{label}/{label}_{count}.jpg", frame)
        print(f"Frame saved as {label}_{count}.jpg")
        count += 1

# Release the capture and writer objects
cam.release()
cv2.destroyAllWindows()
