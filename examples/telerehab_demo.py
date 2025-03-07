import cv2
from math import atan2, degrees
import sys
sys.path.append("..")
from BlazeposeDepthaiEdge_telerehab import BlazeposeDepthai
from BlazeposeRenderer import BlazeposeRenderer
from mediapipe_utils import KEYPOINT_DICT
import argparse

####################
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from PIL import Image
####################


exclude_joints = [
    'left_eye_inner', 
    'left_eye', 
    'left_eye_outer',     
    'right_eye_inner', 
    'right_eye', 
    'right_eye_outer', 
    'left_ear', 
    'right_ear',
    'mouth_left', 
    'mouth_right',
    'left_pinky',
    'right_pinky',
    'left_heel',
    'right_heel'
    ]


######################################        
        
def draw_skeleton(x_pos, y_pos, z_pos):
    
    bones = np.array([
        [1, 2],
        
        [1, 3],
        [3, 5],
        [5, 7],
        [5, 9],
        
        [2, 4],
        [4, 6],
        [6, 8],
        [6, 10],
        
        [1, 11],
        [2, 12],
        [11, 12],
        
        [11, 13],
        [13, 15],
        [15, 17],
        
        [12, 14],
        [14, 16],
        [16, 18]])
    
    
    ax = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x_pos, y_pos, z_pos, c='k')
    ax.scatter3D(0, 0, 0, c='y')
    
    left_shoulder = [x_pos[1], y_pos[1], z_pos[1]]    
    right_shoulder = [x_pos[2], y_pos[2], z_pos[2]]    
    shoulder_vector = np.subtract(left_shoulder, right_shoulder) 
    mid_shoulders = left_shoulder - np.divide(shoulder_vector, 2)
    ax.scatter3D(mid_shoulders[0], mid_shoulders[1], mid_shoulders[2], c='r')
    
    ax.set_box_aspect((np.ptp(x_pos), np.ptp(y_pos), np.ptp(z_pos)))
    ax.set_xlabel('x (mm)', fontsize=15, linespacing=2)
    ax.set_ylabel('y (mm)', fontsize=15, linespacing=2)
    ax.set_zlabel('\n\nz (mm)', fontsize=15, linespacing=2)

    # Hide grid and axes ticks
    # ax.grid(False)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    
    ax.view_init(20, 250)
    
    for i in range(0, len(bones)):
        a, b = bones[i]
        ax.plot3D([x_pos[a], x_pos[b]], [y_pos[a], y_pos[b]], [z_pos[a], z_pos[b]], 'k', linewidth=3)
        
    plt.show()
    
    
#######################################
    

##############################################################

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite'], default='lite',
                        help="Landmark model to use (default=%(default)s")

parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")  

parser.add_argument("-o","--output",
                    help="Path to output video file")

args = parser.parse_args()            

pose = BlazeposeDepthai(input_src=args.input, lm_model=args.model)
renderer = BlazeposeRenderer(pose, output=args.output)

###################
joint_names = list(KEYPOINT_DICT.keys())
exclude_joints_idx = [joint_names.index(i) for i in exclude_joints]

joint_coords_df = pd.DataFrame(columns=joint_names)

num_samples = 0

###################

while (True):
    
    # Run blazepose on next frame
    frame, body = pose.next_frame()
    if frame is None: 
        break
    
    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    
    # Pose detection
    if body: 
        
        if pose.all_present(body):
            cv2.putText(frame, "IN FRAME", (5, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0,190,255), 3)
        else:
            cv2.putText(frame, "NOT IN FRAME", (5, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 3)
    
    if (pose.count > 0) and (pose.count % pose.frames == 0):
        num_samples = num_samples + 1
    
    if (pose.count > pose.frames):
        cv2.putText(frame, f"RGB COUNT = {num_samples}", (5, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
        
    if pose.exercise and pose.rating:
        cv2.putText(frame, f"EXERCISE = {pose.exercise}", (5, 400), cv2.FONT_HERSHEY_PLAIN, 10, (20, 20, 255), 15)
        cv2.putText(frame, f"RATING = {pose.rating}", (5, 600), cv2.FONT_HERSHEY_PLAIN, 10, (20, 20, 255), 15)
       
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q'):
        break
    


renderer.exit()
pose.exit()

