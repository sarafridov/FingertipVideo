"""
This code batch processes all videos in a directory, and
estimates an ellipse representing the contact surface area
of the fingertip against the smartphone camera in each video.

The provided directory should contain videos of the form
video_name.MOV. The following output files are produced for
each video_name.MOV file:
video_name.ellipse.MOV: A video showing the contour points and 
    estimated ellipse boundary in each frame of the video.
video_name.area.csv: Estimated ellipse area in each frame,
    normalized by the total area of the frame.
video_name.color.csv: Average color of the pixels inside the
    estimated ellipse, in BGR order, for each frame.

Run this code using the command:
python extractSignals.py directory
"""

import cv2
import numpy as np
import sys
import glob
import skvideo.io

# Take the name of the folder as input
videonames = glob.glob(sys.argv[1] + "/*.MOV")

for videoname in videonames:
    print(videoname)

    ellipsevideoname = videoname[0:-3] + "ellipse.MOV"
    area_csvname = videoname[0:-3] + "area.csv"
    color_csvname = videoname[0:-3] + "color.csv"

    # Read through the video
    frames_elapsed = 0
    video = skvideo.io.vreader(videoname)

    ellipse_areas = []
    ellipse_colors = []

    # Prepare output video file
    ellipsewriter = skvideo.io.FFmpegWriter(ellipsevideoname, 
        inputdict={'-r': '30'}, 
        outputdict={'-vcodec': 'libx264', '-b': '30000000', '-r': '30'})

    for frame in video:

        height = frame.shape[0]
        width = frame.shape[1]
        frame_area = height*width

        frames_elapsed += 1
        # Use the red channel instead of brightness for ellipse estimation. 
        # This is more robust to flash and glare.
        gray = frame[:,:,0] 

        # Threshold (adaptive to creeping flash)
        ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        threshframeB = frame[:,:,2]*thresh.astype(float)/255.0
        threshframeG = frame[:,:,1]*thresh.astype(float)/255.0
        threshframeR = frame[:,:,0]*thresh.astype(float)/255.0

        # Record the average color of the ellipse in B,G,R order
        ellipse_colors.append([np.sum(threshframeB), np.sum(threshframeG), 
            np.sum(threshframeR)]/np.sum(thresh))

        # Find contour
        img,contours,hierarchy = cv2.findContours(thresh, 1, 2)
        cv2.drawContours(img, contours, -1, (255,0,0), 3)
        # Flatten the contours list
        cnt = [item for sublist in contours for item in sublist] 

        pts = np.asarray(cnt)
        pts = np.squeeze(pts)
        # Remove contour points that are along the image border. 
        # These are added by opencv to make the contour closed.
        pts = pts[pts[:,0] > 0]
        pts = pts[pts[:,1] > 0]
        pts = pts[pts[:,0] < width-1]
        pts = pts[pts[:,1] < height-1]

        # If fewer than 5 contour points, skip this video 
        if pts.shape[0] < 5:
            print("Error processing video: ", videoname)
            print("Unable to find enough points to fit an ellipse in frame: ", 
                frames_elapsed)
            print("Moving on to the next video.")
            break

        # Fit an ellipse to each frame independently, using opencv
        ellipse = cv2.fitEllipse(np.asarray(pts)) 
        cv2.ellipse(img,ellipse,(255,0,0),10) 

        # Record the area of the ellipse
        area = np.pi * ellipse[1][0] * ellipse[1][1] / 4.0
        ellipse_areas.append(area / frame_area)
            
        # Save the ellipse frame to the ellipse video
        ellipsewriter.writeFrame(img)
    
    if frames_elapsed > 1:
        ellipsewriter.close()

    np.savetxt(area_csvname, ellipse_areas)   
    np.savetxt(color_csvname, ellipse_colors)

