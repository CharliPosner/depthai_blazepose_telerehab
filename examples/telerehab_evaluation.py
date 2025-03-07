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

import os
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


exercise_names = ['ARM LIFT', 'LATERAL TRUNK TILT', 'TRUNK ROTATION', 'PELVIS ROTATION', 'SQUATTING'] 
state = ['WAIT', 'RECORD', 'STOP']

state_frame_idx = 0
state_frame_total = 930
frame_idx = 0
current_state = 'WAIT'


def update_state(state_frame_idx, exercise_idx, body):
    if (current_state == 'WAIT'):
        if (state_frame_idx >= 90):
            if body:
                if (pose.all_present(body)):
                    state_frame_idx = 0   
                    next_state = 'RECORD'
                else:
                    state_frame_idx = 0
                    next_state = 'WAIT'
            else:
                state_frame_idx = 0
                next_state = 'WAIT'
                
        elif (exercise_idx > 5):
            state_frame_idx = 0   
            next_state = 'STOP'
            
        else: 
            state_frame_idx = state_frame_idx + 1
            next_state = 'WAIT'
            
    elif (current_state == 'RECORD'):
        if body:
            if (state_frame_idx < state_frame_total):
                if (pose.all_present(body)):
                    state_frame_idx = state_frame_idx + 1
                    next_state = 'RECORD'
                else:
                    state_frame_idx = 0  
                    next_state = 'WAIT'
                    
            else: 
                state_frame_idx = 0   
                
                if (exercise_idx == 4):
                    next_state = 'STOP'
                else:    
                    exercise_idx = exercise_idx + 1 
                    next_state = 'WAIT'
        else:
            state_frame_idx = 0  
            next_state = 'WAIT'
            
    elif (current_state == 'STOP'):
        state_frame_idx = state_frame_idx + 1
        next_state = 'STOP'   
    
    else:
        state_frame_idx = 0   
        next_state = 'STOP'
        print("ERROR! UNDEFINED")
            
    return next_state, state_frame_idx, exercise_idx
            
            

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
parser.add_argument("-m", "--model", type=str, choices=['full', 'lite', '831'], default='full',
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
exercise_idx = 0

frame_count = 0

frame_count_list = []
state_frame_idx_list = []
rgb_sample_idx_list = []
raw_rating_output_list = []
raw_exercise_output_list = []
rating_list = []
exercise_list = []
state_list = []
###################

while (True):
    # Run blazepose on next frame  
    frame, body = pose.next_frame()
    current_state, state_frame_idx, exercise_idx = update_state(state_frame_idx, exercise_idx, body)
    
    # cv2.putText(frame, f"{state_frame_idx}", (5, 1000), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 255), 15)
      
    if frame is None: 
        break

    # Draw 2d skeleton
    frame = renderer.draw(frame, body)
    
    if (current_state == 'WAIT'):
        ## tell DepthAI class to not collect coordinate info
        pose.recording = False
        
        cv2.putText(frame, "WAIT", (5, 200), cv2.FONT_HERSHEY_PLAIN, 10, (255, 255, 0), 15)
        
        # Pose detection
        if body: 
            if pose.all_present(body):
                cv2.putText(frame, "IN FRAME", (5, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 190, 255), 15)
            else:
                cv2.putText(frame, "NOT IN FRAME", (5, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 15)
                
    elif (current_state == 'RECORD'):
        ## tell DepthAI class to collect coordinate info
        pose.recording = True
        
        ex = exercise_idx + 1
        
        # cv2.putText(frame, "RECORD", (5, 200), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 0), 15)
        
        exercise = exercise_names[exercise_idx]
        cv2.putText(frame, f"{exercise}", (100, 200), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 15)
        if pose.rating is not None:
            cv2.putText(frame, f"Exercise: {pose.exercise}", (1000, 350), cv2.FONT_HERSHEY_PLAIN, 8, (0, 255, 0), 15)
            cv2.putText(frame, f"Rating: {pose.rating}", (1000, 450), cv2.FONT_HERSHEY_PLAIN, 8, (255, 0, 0), 15)
            # cv2.putText(frame, f"E:{pose.exercise}, R:{pose.rating}", (5, 600), cv2.FONT_HERSHEY_PLAIN, 10, (20, 255, 255), 15)
            
        if (pose.count > 0) and (pose.count % pose.frames == 0):
            
            ###########################
            if (exercise_idx == 3):
                with open("ex4_x_coords.csv", "ab") as f:
                    np.savetxt(f, pose.coord_buffer[:, :, 0], delimiter=",")
                with open("ex4_y_coords.csv", "ab") as f:
                    np.savetxt(f, pose.coord_buffer[:, :, 1], delimiter=",")
                with open("ex4_z_coords.csv", "ab") as f:
                    np.savetxt(f, pose.coord_buffer[:, :, 2], delimiter=",")

            ############################
            
            if pose.rgb_image is not None:
                img_filename = f"sample{pose.rgb_sample_idx}_ex{ex}_RGB.png"
                img_filepath = os.path.join(os.getcwd(), img_filename)
                cv2.imwrite(img_filepath, pose.rgb_image)
                
                frame_filename = f"sample{pose.rgb_sample_idx}_ex{ex}_FRAME.png"
                frame_filepath = os.path.join(os.getcwd(), frame_filename)
                cv2.imwrite(frame_filepath, frame)
                                
                state_list.append(ex)
                frame_count_list.append(frame_count)
                state_frame_idx_list.append(state_frame_idx)
                rgb_sample_idx_list.append(pose.rgb_sample_idx)    
                            
                rating_list.append(pose.rating)
                exercise_list.append(pose.exercise)
                raw_rating_output_list.append(pose.raw_rating_output)
                raw_exercise_output_list.append(pose.raw_exercise_output)
                pose.rgb_sample_idx = pose.rgb_sample_idx + 1
                
    # if (pose.count > pose.frames):
        # cv2.putText(frame, f"RGB COUNT = {pose.rgb_sample_idx}", (5, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 3)
    
    frame_count = frame_count + 1
    
    key = renderer.waitKey(delay=1)
    if key == 27 or key == ord('q') or (current_state == 'STOP'):
        break

if pose.rgb_image is not None:
    if (pose.rgb_sample_idx % 10 == 0):
        ex = exercise_idx
    else:
        ex = exercise_idx + 1
        
    img_filename = f"sample{pose.rgb_sample_idx}_ex{ex}.png"
    img_filepath = os.path.join(os.getcwd(), img_filename)
    cv2.imwrite(img_filepath, pose.rgb_image) 
    
    state_list.append(ex)
    frame_count_list.append(frame_count)
    state_frame_idx_list.append(state_frame_idx)
    rgb_sample_idx_list.append(pose.rgb_sample_idx)    
            
    rating_list.append(pose.rating)
    exercise_list.append(pose.exercise)
    raw_rating_output_list.append(pose.raw_rating_output)
    raw_exercise_output_list.append(pose.raw_exercise_output)
    
lists_for_file = [frame_count_list, state_frame_idx_list, rgb_sample_idx_list, raw_rating_output_list, rating_list, raw_exercise_output_list, exercise_list, state_list]
 
with open("output.csv", "w") as csv:
    csv.write("frame_count;state_frame_idx;rgb_sample_idx;raw_rating_output;rating_prediction;raw_exercise_output;exercise_prediction;state;\n")
    print(min(map(len, lists_for_file)), max(map(len, lists_for_file)))
    
    for i in range(max(map(len, lists_for_file))):
        try: 
            list_to_write =[item[i] for item in lists_for_file]
            for item in list_to_write:
                csv.write(str(item))
                csv.write(';')
            csv.write("\n")
        except:
            pass
        
            

renderer.exit()
pose.exit()

