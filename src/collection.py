import os

import cv2


def start_collection():
    # Open the default camera
    cam = cv2.VideoCapture(0)

    label = input("Enter the label for this frame: ")

    if not os.path.exists(f"data/{label}/raw_data"):
        os.makedirs(f"data/{label}/raw_data")

    count = len(os.listdir(f"data/{label}/raw_data")) + 1

    while True:
        ret, frame = cam.read()

        # Display the captured frame
        cv2.imshow("Camera", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord("q"):
            break

        if cv2.waitKey(1) == ord("s"):
            # Save the current frame as an image file
            cv2.imwrite(f"data/{label}/raw_data/{label}_{count}.jpg", frame)
            print(f"Frame saved as {label}_{count}.jpg")
            count += 1

    # Release the capture and writer objects
    cam.release()
    cv2.destroyAllWindows()
