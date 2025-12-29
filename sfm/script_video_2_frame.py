import cv2
import os

# Open a video file with opencv
right_or_left = "left" # change that to "left"/"rigth" depending on which video you want 
path_video = f"./video/{right_or_left}.mp4"
cam = cv2.VideoCapture(path_video)

os.makedirs(f"./video/{right_or_left}_views/")
path_img = f"./video/{right_or_left}_views/" + "frame-{fr:03.0f}.png"

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

idx = 0
steps = 3 # change this to increase/decrease the nb of exported frames. If lower, more frames. If higher, lesser.
while True:
    ret, frame = cam.read()
    if not ret: break

    # Export every "step"-th frame in the video
    if (idx % steps) == 0:
        cv2.imwrite(path_img.format(fr=idx), frame)

    idx += 1

# Release the capture and writer objects
cam.release()