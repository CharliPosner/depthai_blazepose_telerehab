import numpy as np
import cv2
from numpy.core.fromnumeric import trace
import mediapipe_utils as mpu
from pathlib import Path
from FPS import FPS, now
import depthai as dai
import marshal
import sys

###########################################################################

import struct
import time
import inspect
import textwrap 

from PIL import Image

###########################################################################

from string import Template
from math import sin, cos

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_DETECTION_MODEL = str(SCRIPT_DIR / "models/pose_detection_sh4.blob")
LANDMARK_MODEL_FULL = str(SCRIPT_DIR / "models/pose_landmark_full_sh4.blob")
# LANDMARK_MODEL_HEAVY = str(SCRIPT_DIR / "models/pose_landmark_heavy_sh4.blob")
LANDMARK_MODEL_LITE = str(SCRIPT_DIR / "models/pose_landmark_lite_sh4.blob")
DETECTION_POSTPROCESSING_MODEL = str(SCRIPT_DIR / "custom_models/DetectionBestCandidate_sh1.blob")
DIVIDE_BY_255_MODEL = str(SCRIPT_DIR / "custom_models/DivideBy255_sh1.blob")

###########################################################################

TEMPLATE_MANAGER_SCRIPT = str(SCRIPT_DIR / "template_manager_script_telerehab.py")
REHAB_MODEL = str(SCRIPT_DIR / "models/telerehab_sh8.blob")

###########################################################################


def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    return cv2.resize(arr, shape).transpose(2,0,1).flatten()

class BlazeposeDepthai:
    """
    Blazepose body pose detector
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host,
                    Note that as we are in Edge mode, input sources coming from the host like a image or a video is not supported 
    - pd_model: Blazepose detection model blob file (if None, takes the default value POSE_DETECTION_MODEL),
    - pd_score: confidence score to determine whether a detection is reliable (a float between 0 and 1). <<<<<<<<<<<<<<<<<<<<<<<<<, INCREASE
    - pp_model: detection postprocessing model blob file  (if None, takes the default value DETECTION_POSTPROCESSING_MODEL),,
    - lm_model: Blazepose landmark model blob file
                    - None or "full": the default blob file LANDMARK_MODEL_FULL,
                    - "lite": the default blob file LANDMARK_MODEL_LITE,
                    - "831": the full model from previous version of mediapipe (0.8.3.1) LANDMARK_MODEL_FULL_0831,
                    - a path of a blob file. 
    - lm_score_thresh : confidence score to determine whether landmarks prediction is reliable (a float between 0 and 1). <<<<<<<<<<<<<<<<<<<< USEFUL
    - xyz: boolean, when True get the (x, y, z) coords of the reference point (center of the hips) (if the device supports depth measures).
    - crop : boolean which indicates if square cropping is done or not
    - smoothing: boolean which indicates if smoothing filtering is applied
    - filter_window_size and filter_velocity_scale:
            The filter keeps track (on a window of specified size) of
            value changes over time, which as result gives velocity of how value
            changes over time. With higher velocity it weights new values higher.
            - higher filter_window_size adds to lag and to stability
            - lower filter_velocity_scale adds to lag and to stability

    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                                The width is calculated accordingly to height and depends on value of 'crop'
    - stats : boolean, when True, display some statistics when exiting.  
    - trace: boolean, when True print some debug messages 
    - force_detection:     boolean, force person detection on every frame (never use landmarks from previous frame to determine ROI)    
    """
    def __init__(self, input_src="rgb",
                pd_model=None, 
                pd_score_thresh=0.9, 
                pp_model=None,
                lm_model=None,
                lm_score_thresh=0.9,
                xyz=True,
                crop=False,
                smoothing= True,
                filter_window_size=5,
                filter_velocity_scale=10,
                stats=False,               
                internal_fps=None,
                internal_frame_height=1080,
                trace=True,
                force_detection=False,
                
                frames=90):##### add R, G, B, flag instantiations here if needed
        
        self.pd_model = pd_model if pd_model else POSE_DETECTION_MODEL
        self.pp_model = pp_model if pd_model else DETECTION_POSTPROCESSING_MODEL
        self.divide_by_255_model = DIVIDE_BY_255_MODEL
        print(f"Pose detection blob file : {self.pd_model}")
        self.rect_transf_scale = 1.25
        if lm_model is None or lm_model == "lite":
            self.lm_model = LANDMARK_MODEL_LITE
        elif lm_model == "full":
            self.lm_model = LANDMARK_MODEL_FULL
        # elif lm_model == "heavy":
        #     self.lm_model = LANDMARK_MODEL_HEAVY
        else:
            self.lm_model = lm_model
        print(f"Landmarks using blob file : {self.lm_model}")

        self.pd_score_thresh = pd_score_thresh
        self.lm_score_thresh = lm_score_thresh
        self.smoothing = smoothing
        self.crop = crop
        self.internal_fps = internal_fps
        self.stats = stats
        self.presence_threshold = 0.9
        self.visibility_threshold = 0.9

        self.trace = trace
        self.force_detection = force_detection

        self.device = dai.Device()
        self.xyz = False
        
        #################################################################################################################
        
        self.R = None
        self.G = None
        self.B = None
        
        self.count = 0
        self.flag = 0
        self.frames = frames
        
        self.rehab_model = REHAB_MODEL
        self.coord_buffer = np.empty((frames, len(mpu.KEYPOINT_DICT), 3))
        
        self.raw_exercise_output = None
        self.raw_rating_output = None
        
        self.exercise = None
        self.rating = None
        
        self.recording = False
        self.rgb_image = None
        
        self.rgb_sample_idx = 0
        
        #################################################################################################################
        
        
        if input_src == None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb" # OAK* internal color camera
            self.laconic = input_src == "rgb_laconic" # Camera frames are not sent to the host      
            if xyz:
                # Check if the device supports stereo
                cameras = self.device.getConnectedCameras()
                if dai.CameraBoardSocket.LEFT in cameras and dai.CameraBoardSocket.RIGHT in cameras:
                    self.xyz = True
                else:
                    print("Warning: depth unavailable on this device, 'xyz' argument is ignored")

            if internal_fps is None:            
                if "full" in str(self.lm_model):
                    self.internal_fps = 18 if self.xyz else 20
                elif "heavy" in str(lm_model):
                    self.internal_fps = 7 if self.xyz else 8
                else: # "lite"
                    self.internal_fps = 22 if self.xyz else 26
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps
            
            if self.crop:
                self.frame_size, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height)
                self.img_h = self.img_w = self.frame_size
                self.pad_w = self.pad_h = 0
                self.crop_w = (int(round(1920 * self.scale_nd[0] / self.scale_nd[1])) - self.img_w) // 2

            else:
                width, self.scale_nd = mpu.find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
                self.img_h = int(round(1080 * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(1920 * self.scale_nd[0] / self.scale_nd[1]))
                self.pad_h = (self.img_w - self.img_h) // 2
                self.pad_w = 0
                self.frame_size = self.img_w
                self.crop_w = 0

            print(f"Internal camera image size: {self.img_w} x {self.img_h} - pad_h: {self.pad_h}")

        else:
            print("Invalid input source:", input_src)
            sys.exit()

        self.nb_kps = 33

        if self.smoothing:
            self.filter_landmarks = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.05,
                beta=80,
                derivate_cutoff=1
            )
            # landmarks_aux corresponds to the 2 landmarks used to compute the ROI in next frame
            self.filter_landmarks_aux = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.01,
                beta=10,
                derivate_cutoff=1
            )
            self.filter_landmarks_world = mpu.LandmarksSmoothingFilter(
                frequency=self.video_fps,
                min_cutoff=0.1,
                beta=40,
                derivate_cutoff=1,
                disable_value_scaling=True
            )
            if self.xyz:
                self.filter_xyz = mpu.LowPassFilter(alpha=0.25)

        # Define and start pipeline
        usb_speed = self.device.getUsbSpeed()
        self.device.startPipeline(self.create_pipeline())
        print(f"Pipeline started - USB speed: {str(usb_speed).split('.')[-1]}")

        # Define data queues 
        if not self.laconic:
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False) ### video output
        self.q_manager_out = self.device.getOutputQueue(name="manager_out", maxSize=1, blocking=False) ### pose output
        
        
        ################################################################################################################################
        
        self.q_rgb_in = self.device.getInputQueue(name="rgb_in", maxSize=1, blocking=False) ## imageManip rgb input 
        self.q_rgb_out = self.device.getOutputQueue("test_imagemanip", maxSize=1, blocking=False)
        
        ################################################################################################################################
        
        # For debugging
        #self.q_pre_pd_manip_out = self.device.getOutputQueue(name="pre_pd_manip_out", maxSize=1, blocking=False)
        #self.q_pre_lm_manip_out = self.device.getOutputQueue(name="pre_lm_manip_out", maxSize=1, blocking=False)

        self.fps = FPS()

        self.nb_pd_inferences = 0
        self.nb_lm_inferences = 0
        self.nb_lm_inferences_after_landmarks_ROI = 0
        self.nb_frames_no_body = 0
        

    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)
        self.pd_input_length = 224
        self.lm_input_length = 256

        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera) 
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        cam.setFps(self.internal_fps)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)

        if self.crop:
            cam.setVideoSize(self.frame_size, self.frame_size)
            cam.setPreviewSize(self.frame_size, self.frame_size)
        else: 
            cam.setVideoSize(self.img_w, self.img_h)
            cam.setPreviewSize(self.img_w, self.img_h)

        if not self.laconic:
            cam_out = pipeline.create(dai.node.XLinkOut)
            cam_out.setStreamName("cam_out")
            cam_out.input.setQueueSize(1)
            cam_out.input.setBlocking(False)
            cam.video.link(cam_out.input)


        # Define manager script node
        manager_script = pipeline.create(dai.node.Script)
        manager_script.setScript(self.build_manager_script())

        if self.xyz:
            print("Creating MonoCameras, Stereo and SpatialLocationCalculator nodes...")
            # For now, RGB needs fixed focus to properly align with depth.
            # The value used during calibration should be used here
            calib_data = self.device.readCalibration()
            calib_lens_pos = calib_data.getLensPosition(dai.CameraBoardSocket.RGB)
            print(f"RGB calibration lens position: {calib_lens_pos}")
            cam.initialControl.setManualFocus(calib_lens_pos)

            mono_resolution = dai.MonoCameraProperties.SensorResolution.THE_400_P
            left = pipeline.createMonoCamera()
            left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            left.setResolution(mono_resolution)
            left.setFps(self.internal_fps)

            right = pipeline.createMonoCamera()
            right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            right.setResolution(mono_resolution)
            right.setFps(self.internal_fps)

            stereo = pipeline.createStereoDepth()
            stereo.setConfidenceThreshold(230)
            # LR-check is required for depth alignment
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setSubpixel(False)  # subpixel True -> latency
            # MEDIAN_OFF necessary in depthai 2.7.2. 
            # Otherwise : [critical] Fatal error. Please report to developers. Log: 'StereoSipp' '533'
            # stereo.setMedianFilter(dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF)

            spatial_location_calculator = pipeline.createSpatialLocationCalculator()
            spatial_location_calculator.setWaitForConfigInput(True)
            spatial_location_calculator.inputDepth.setBlocking(False)
            spatial_location_calculator.inputDepth.setQueueSize(1)

            left.out.link(stereo.left)
            right.out.link(stereo.right)    

            stereo.depth.link(spatial_location_calculator.inputDepth)

            manager_script.outputs['spatial_location_config'].link(spatial_location_calculator.inputConfig)
            spatial_location_calculator.out.link(manager_script.inputs['spatial_data'])

        # Define pose detection pre processing (resize preview to (self.pd_input_length, self.pd_input_length))
        print("Creating Pose Detection pre processing image manip...")
        pre_pd_manip = pipeline.create(dai.node.ImageManip)
        pre_pd_manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
        pre_pd_manip.setWaitForConfigInput(True)
        pre_pd_manip.inputImage.setQueueSize(1)
        pre_pd_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_pd_manip.inputImage)
        manager_script.outputs['pre_pd_manip_cfg'].link(pre_pd_manip.inputConfig)

        # For debugging
        # pre_pd_manip_out = pipeline.createXLinkOut()
        # pre_pd_manip_out.setStreamName("pre_pd_manip_out")
        # pre_pd_manip.out.link(pre_pd_manip_out.input)

        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(self.pd_model)
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)
        pre_pd_manip.out.link(pd_nn.input)
       
        # Define pose detection post processing "model"
        print("Creating Pose Detection post processing Neural Network...")
        post_pd_nn = pipeline.create(dai.node.NeuralNetwork)
        post_pd_nn.setBlobPath(self.pp_model)
        pd_nn.out.link(post_pd_nn.input)
        post_pd_nn.out.link(manager_script.inputs['from_post_pd_nn'])

        # Define link to send result to host 
        manager_out = pipeline.create(dai.node.XLinkOut)
        manager_out.setStreamName("manager_out")
        manager_out.input.setQueueSize(1)
        manager_out.input.setBlocking(False)
        # manager_out.video.link(cam_out.input)
        manager_script.outputs['host'].link(manager_out.input)
        
        ###################################################################
        ###################################################################
        
        
        #################### HOST -> DEVICE DATA QUEUE ####################
       
        rgb_in = pipeline.create(dai.node.XLinkIn)
        rgb_in.setMaxDataSize(150528)
        rgb_in.setStreamName("rgb_in")
        
        
        ################## RGB (3, 90, 19) TO IMAGEMANIP ##################
  
        pre_rehab_manip = pipeline.create(dai.node.ImageManip)
        pre_rehab_manip.initialConfig.setKeepAspectRatio(False)
        pre_rehab_manip.initialConfig.setResize(224, 224)
        pre_rehab_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
        pre_rehab_manip.inputImage.setBlocking(False)
        pre_rehab_manip.inputImage.setQueueSize(1)
        
        rgb_in.out.link(pre_rehab_manip.inputImage)
        
        
        ###################### VIEW IMAGEMANIP IMAGE ######################
        
        out = pipeline.create(dai.node.XLinkOut)
        out.setStreamName('test_imagemanip')
        out.input.setBlocking(False)
        out.input.setQueueSize(1)
        pre_rehab_manip.out.link(out.input)
        
        
        ###################### REHAB NEURAL NETWORK #######################
        
        print("Creating REHAB MODEL Neural Network...")
        rehab_nn = pipeline.create(dai.node.NeuralNetwork)
        rehab_nn.setBlobPath(self.rehab_model)
        pre_rehab_manip.out.link(rehab_nn.input)
        
        rehab_nn.out.link(manager_script.inputs['from_rehab_nn'])
         
        ###################################################################
        ###################################################################

        # Define landmark pre processing image manip
        print("Creating Landmark pre processing image manip...") 
        pre_lm_manip = pipeline.create(dai.node.ImageManip)
        pre_lm_manip.setMaxOutputFrameSize(self.lm_input_length*self.lm_input_length*3)
        pre_lm_manip.setWaitForConfigInput(True)
        pre_lm_manip.inputImage.setQueueSize(1)
        pre_lm_manip.inputImage.setBlocking(False)
        cam.preview.link(pre_lm_manip.inputImage)

        # For debugging
        # pre_lm_manip_out = pipeline.createXLinkOut()
        # pre_lm_manip_out.setStreamName("pre_lm_manip_out")
        # pre_lm_manip.out.link(pre_lm_manip_out.input)
    
        manager_script.outputs['pre_lm_manip_cfg'].link(pre_lm_manip.inputConfig)

        # Define normalization model between ImageManip and landmark model
        # This is a temporary step. Could be removed when support of setFrameType(RGBF16F16F16p) in ImageManip node
        print("Creating DiveideBy255 Neural Network...") 
        divide_nn = pipeline.create(dai.node.NeuralNetwork)
        divide_nn.setBlobPath(self.divide_by_255_model)
        pre_lm_manip.out.link(divide_nn.input) 

        # Define landmark model
        print("Creating Landmark Neural Network...") 
        lm_nn = pipeline.create(dai.node.NeuralNetwork)
        lm_nn.setBlobPath(self.lm_model)
        # lm_nn.setNumInferenceThreads(1)
  
        divide_nn.out.link(lm_nn.input)       
        lm_nn.out.link(manager_script.inputs['from_lm_nn'])

        print("Pipeline created.")

        return pipeline        



    def build_manager_script(self):
        '''
        The code of the scripting node 'manager_script' depends on :
            - the NN model (full, lite, 831),
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_manager_script.py which is a python template
        '''
        # Read the template
        with open(TEMPLATE_MANAGER_SCRIPT, 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _TRACE = "node.warn" if self.trace else "#",
                    _pd_score_thresh = self.pd_score_thresh,
                    _lm_score_thresh = self.lm_score_thresh,
                    _force_detection = self.force_detection,
                    _pad_h = self.pad_h,
                    _img_h = self.img_h,
                    _img_w = self.img_w,
                    _frame_size = self.frame_size,
                    _crop_w = self.crop_w,
                    _rect_transf_scale = self.rect_transf_scale,
                    _IF_XYZ = "" if self.xyz else '"""',
                    _buffer_size = 2910 if self.xyz else 2863,
                    _visibility_threshold = self.visibility_threshold
                    
        )
        
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debugging
        if self.trace:
            with open("tmp_code.py", "w") as file:
                file.write(code)

        return code

    def is_present(self, body, lm_id):
        return body.presence[lm_id] > self.presence_threshold

    def lm_postprocess(self, body, lms, lms_world):
        # lms : landmarks sent by Manager script node to host (list of 39*5 elements for full body or 31*5 for upper body)
        lm_raw = np.array(lms).reshape(-1,5)
        # Each keypoint have 5 information:
        # - X,Y coordinates are local to the body of
        # interest and range from [0.0, 255.0].
        # - Z coordinate is measured in "image pixels" like
        # the X and Y coordinates and represents the
        # distance relative to the plane of the subject's
        # hips, which is the origin of the Z axis. Negative
        # values are between the hips and the camera;
        # positive values are behind the hips. Z coordinate
        # scale is similar with X, Y scales but has different
        # nature as obtained not via human annotation, by
        # fitting synthetic data (GHUM model) to the 2D
        # annotation.
        # - Visibility, after user-applied sigmoid denotes the
        # probability that a keypoint is located within the
        # frame and not occluded by another bigger body
        # part or another object.
        # - Presence, after user-applied sigmoid denotes the
        # probability that a keypoint is located within the
        # frame.

        # Normalize x,y,z. Scaling in z = scaling in x = 1/self.lm_input_length
        lm_raw[:,:3] /= self.lm_input_length
        # Apply sigmoid on visibility and presence (if used later)
        body.visibility = 1 / (1 + np.exp(-lm_raw[:,3]))
        body.presence = 1 / (1 + np.exp(-lm_raw[:,4]))

        # body.norm_landmarks contains the normalized ([0:1]) 3D coordinates of landmarks in the square rotated body bounding box
        body.norm_landmarks = lm_raw[:,:3]
        # Now calculate body.landmarks = the landmarks in the image coordinate system (in pixel) (body.landmarks)
        src = np.array([(0, 0), (1, 0), (1, 1)], dtype=np.float32)
        dst = np.array([ (x, y) for x,y in body.rect_points[1:]], dtype=np.float32) # body.rect_points[0] is left bottom point and points going clockwise!
        mat = cv2.getAffineTransform(src, dst)
        lm_xy = np.expand_dims(body.norm_landmarks[:self.nb_kps,:2], axis=0)
        lm_xy = np.squeeze(cv2.transform(lm_xy, mat))  

        # A segment of length 1 in the coordinates system of body bounding box takes body.rect_w_a pixels in the
        # original image. Then we arbitrarily divide by 4 for a more realistic appearance.
        lm_z = body.norm_landmarks[:self.nb_kps,2:3] * body.rect_w_a / 4
        lm_xyz = np.hstack((lm_xy, lm_z))

        # World landmarks are predicted in meters rather than in pixels of the image
        # and have origin in the middle of the hips rather than in the corner of the
        # pose image (cropped with given rectangle). Thus only rotation (but not scale
        # and translation) is applied to the landmarks to transform them back to
        # original  coordinates.
        body.landmarks_world = np.array(lms_world).reshape(-1,3) ## list [1, 99] -> np.array [33, 3]
        sin_rot = sin(body.rotation)
        cos_rot = cos(body.rotation)
        rot_m = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])
        body.landmarks_world[:,:2] = np.dot(body.landmarks_world[:,:2], rot_m) ### changes origin?

        if self.smoothing:
            timestamp = now()
            object_scale = body.rect_w_a
            lm_xyz[:self.nb_kps] = self.filter_landmarks.apply(lm_xyz[:self.nb_kps], timestamp, object_scale)
            lm_xyz[self.nb_kps:] = self.filter_landmarks_aux.apply(lm_xyz[self.nb_kps:], timestamp, object_scale)
            body.landmarks_world = self.filter_landmarks_world.apply(body.landmarks_world, timestamp)

        body.landmarks = lm_xyz.astype(int)
        # If we added padding to make the image square, we need to remove this padding from landmark coordinates and from rect_points
        if self.pad_h > 0:
            body.landmarks[:,1] -= self.pad_h
            for i in range(len(body.rect_points)):
                body.rect_points[i][1] -= self.pad_h
        # if self.pad_w > 0:
        #     body.landmarks[:,0] -= self.pad_w
        #     for i in range(len(body.rect_points)):
        #         body.rect_points[i][0] -= self.pad_w  

#####################################################################################################        
    
    # ensure that the whole body is present in the frame
    def all_present(self, body):
        for i in range(len(mpu.KEYPOINT_DICT)):
            if body.presence[i] < self.presence_threshold:
                return False
        return True
    
    #  normalise landmarks_world joint_coords for 1 frame:
        # set the origin to the mid-hips
        # scale each skeleton according to vector between mid-hips and mid-shoulders
    def normalise_skeleton(self, joint_coords):      
        # Indexes of some keypoints 
        left_shoulder = 11
        right_shoulder = 12
        left_hip = 23
        right_hip = 24
        
        # get coordinates of mid hip and set to origin
        left_hip_coords = joint_coords[left_hip, :]
        right_hip_coords = joint_coords[right_hip, :]
        hip_vector = np.subtract(left_hip_coords, right_hip_coords)
        mid_hip_coords = left_hip_coords - np.divide(hip_vector, 2)

        new_origin_coords = np.subtract(joint_coords, mid_hip_coords)

        left_shoulder_coords = new_origin_coords[left_shoulder, :]    
        right_shoulder_coords = new_origin_coords[right_shoulder, :]
        shoulder_vector = np.subtract(left_shoulder_coords, right_shoulder_coords) 
        mid_shoulder_coords = left_shoulder_coords - np.divide(shoulder_vector, 2)
        
        ref_length = np.linalg.norm(np.absolute(mid_shoulder_coords))
        norm_coords = np.divide(new_origin_coords, ref_length)
        
        return norm_coords # shape: (33, 3)
    
    
    ## coord_buffer shape = (90, 33, 3)
    def coords_to_rgb(self, coord_buffer):
        exclude_joints_idx =  sorted([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 18, 29, 30], reverse=True)
        
        ## multiply y and z by -1
        coord_buffer[:, :, 1:] = coord_buffer[:, :, 1:] * -1
        
        ## delete from coord buffer by exclude idx
        R = np.delete(coord_buffer[:, :, 0], exclude_joints_idx, axis=1)
        G = np.delete(coord_buffer[:, :, 1], exclude_joints_idx, axis=1)
        B = np.delete(coord_buffer[:, :, 2], exclude_joints_idx, axis=1)
        
        R = ((R - R.min())/(R.max()-R.min()) * 255).astype('uint8')
        G = ((G - G.min())/(G.max()-G.min()) * 255).astype('uint8')
        B = ((B - B.min())/(B.max()-B.min()) * 255).astype('uint8')

        # combine each channel into one ndarray   
        RGB = np.dstack((R,G,B))
        
        # image = Image.fromarray(RGB)
        # image = image.resize((224, 224))

        # image_file = 'D:\\Misc\\OAK-D\\depthai_blazepose\\examples\\semaphore_alphabet\\test_pipelineRGB.png'
        # image.save(image_file)

        return RGB
       
        
    ## transform RGB ndarray of shape (90, 19, 3) into an imgFrame  
    def create_img_frame(self, RGB):

        ## Takes R, G, B lists, each of dimensions (90, 19), 
        ## and generates the DepthAI ImgFrame object with size (224, 224, 3).
        ## Returns the ImgFrame object with FP16 data type.

        # Define the dimensions of the image frame we want to create
        
        width = RGB.shape[1]
        height = RGB.shape[0]
        
        img_frame = dai.ImgFrame()
        img_frame.setWidth(width)
        img_frame.setHeight(height)
        img_frame.setType(dai.RawImgFrame.Type.RGB888p) ## p = PLANAR (CHW), i = INTERLEAVED (HWC)
        img_frame.setData(RGB.transpose(2, 0, 1))

        # Return the generated ImgFrame object
        return img_frame    
    
    
#####################################################################################################
              
                
    def next_frame(self):

        self.fps.update()
            
        if self.laconic:
            video_frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        else:
            in_video = self.q_video.get()
            video_frame = in_video.getCvFrame()       
        
        # For debugging
        # pre_pd_manip = self.q_pre_pd_manip_out.tryGet()
        # if pre_pd_manip:
        #     pre_pd_manip = pre_pd_manip.getCvFrame()
        #     cv2.imshow("pre_pd_manip", pre_pd_manip)
        # pre_lm_manip = self.q_pre_lm_manip_out.tryGet()
        # if pre_lm_manip:
        #     pre_lm_manip = pre_lm_manip.getCvFrame()
        #     cv2.imshow("pre_lm_manip", pre_lm_manip)
                                
        
        #############################################################################
        
        frame_idx = self.count % self.frames
        joint_coords = np.empty((33, 3))
        
        if (self.count > 0) and (frame_idx == 0):
            RGB = self.coords_to_rgb(self.coord_buffer)
            img_frame = self.create_img_frame(RGB)
            self.rehab_imagemanip_input = self.q_rgb_in.send(img_frame)
            
        img = self.q_rgb_out.tryGet()

        try:
            manip_img = img.getCvFrame()
            # cv2.imshow("manip_img", manip_img)
            self.rgb_image = manip_img
            
        except:
            pass
        
        
        #############################################################################
        
        
        # Get result from device
        res = marshal.loads(self.q_manager_out.get().getData())
        
        if res["exercise_output"] and res["rating_output"]:
            
            exercise_output = np.array(res["exercise_output"])
            rating_output = np.array(res["rating_output"])
            
            self.raw_exercise_output = exercise_output
            self.raw_rating_output = rating_output
            
            self.exercise = np.argmax(exercise_output) + 1
            self.rating = np.argmax(rating_output) + 1
            
        if res["type"] != 0 and res["lm_score"] > self.lm_score_thresh:
            body = mpu.Body()
            body.rect_x_center_a = res["rect_center_x"] * self.frame_size
            body.rect_y_center_a = res["rect_center_y"] * self.frame_size
            body.rect_w_a = body.rect_h_a = res["rect_size"] * self.frame_size
            body.rotation = res["rotation"] 
            body.rect_points = mpu.rotated_rect_to_points(body.rect_x_center_a, body.rect_y_center_a, body.rect_w_a, body.rect_h_a, body.rotation)
            body.lm_score = res["lm_score"]
            self.lm_postprocess(body, res['lms'], res['lms_world'])
            if self.xyz:
                if res['xyz_ref'] == 0:
                    body.xyz_ref = None
                else:
                    if res['xyz_ref'] == 1:
                        body.xyz_ref = "mid_hips"
                    else: # res['xyz_ref'] == 2:
                        body.xyz_ref = "mid_shoulders"
                    body.xyz = np.array(res["xyz"])
                    if self.smoothing:
                        body.xyz = self.filter_xyz.apply(body.xyz)
                    body.xyz_zone = np.array(res["xyz_zone"])
                    body.xyz_ref_coords_pixel = np.mean(body.xyz_zone.reshape((2,2)), axis=0)


#####################################################################################################               
                
                
                if (self.recording):
                    for landmark in mpu.KEYPOINT_DICT.values():
                        joint_coords[landmark, :] = np.array(body.landmarks_world[landmark, :])
                     
                    norm_coords = self.normalise_skeleton(joint_coords)
                    self.coord_buffer[frame_idx, :, :] = norm_coords
                    self.count = self.count + 1 
                    
                else:
                    self.count = 0
                    self.coord_buffer = np.empty((self.frames, len(mpu.KEYPOINT_DICT), 3))
                      
                      
#####################################################################################################

        else:
            body = None
            if self.smoothing: 
                self.filter_landmarks.reset()
                self.filter_landmarks_aux.reset()
                self.filter_landmarks_world.reset()
                if self.xyz: self.filter_xyz.reset()

        # Statistics
        if self.stats:
            if res["type"] == 0:
                self.nb_pd_inferences += 1
                self.nb_frames_no_body += 1
            else:  
                self.nb_lm_inferences += 1
                if res["type"] == 1:
                    self.nb_pd_inferences += 1
                else: # res["type"] == 2
                    self.nb_lm_inferences_after_landmarks_ROI += 1
                if res["lm_score"] < self.lm_score_thresh: self.nb_frames_no_body += 1

        return video_frame, body
    

    def exit(self):
        self.device.close()
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.get_global():.1f} f/s (# frames = {self.fps.nbf})")
            print(f"# frames without body       : {self.nb_frames_no_body}")
            print(f"# pose detection inferences : {self.nb_pd_inferences}")
            print(f"# landmark inferences       : {self.nb_lm_inferences} - # after pose detection: {self.nb_lm_inferences - self.nb_lm_inferences_after_landmarks_ROI} - # after landmarks ROI prediction: {self.nb_lm_inferences_after_landmarks_ROI}")
        
        
           

