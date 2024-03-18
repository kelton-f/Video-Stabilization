#########################################
#
# Written by: Kelton French
# Date: 11/28/2022
#
#########################################

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Subroutines
def stabilizeVideo(cap, out1, out2):
  frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
  height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
  width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

  # Will use this to store previous shifts. Starting with identity matrix
  prevShift = [np.identity(3)]

  ret, prev = cap.read()
  prevGray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)

  transform = []
  for i in range(frameCount-1):

    # Read in the current frame and make sure it is read correctly
    success, curr = cap.read()
    currGray = cv.cvtColor(curr, cv.COLOR_BGR2GRAY)
    if not success:
      break

    # Find features and calculate the optical flow
    prevPts = cv.goodFeaturesToTrack(prevGray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    currPts, status, err = cv.calcOpticalFlowPyrLK(prevGray, currGray, prevPts, None)

    # Using the calculated points, find the homography for frames
    h, status = cv.findHomography(prevPts, currPts, cv.RANSAC,5.0)

    # Append the dot product of prevShift and current frame homography
    prevShift.append(np.dot(h, prevShift[-1]))

    # Update prev image
    prevGray = currGray

    # Get the average of the previous shifts then calculate the dot product of the average and inverse of last shift
    applyShift = np.dot(np.average(np.array(prevShift), 0), np.linalg.inv(prevShift[-1]))
    output = cv.warpPerspective(curr, applyShift, (width, height))

    # Output transformed image
    frameOut = cv.hconcat([curr, output])
    out1.write(frameOut)
    out2.write(output)
    if i % 10 == 0:
      percent = str(round((i/frameCount) * 100)) + "%"
      print("Percent done: %s" % percent)
  print("Done!")

cap = cv.VideoCapture("/content/pianoVideo.mp4")

frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
out1 = cv.VideoWriter('sideBySide.avi', cv.VideoWriter_fourcc(*'MJPG'), cap.get(cv.CAP_PROP_FPS), (2*width, height))
out2 = cv.VideoWriter('stabilized.avi', cv.VideoWriter_fourcc(*'MJPG'), cap.get(cv.CAP_PROP_FPS), (width, height))

stabilizeVideo(cap, out1, out2)

# Clean up
cap.release()
out1.release()
out2.release()
cv.destroyAllWindows()
