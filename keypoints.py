import os
os.environ["GLOG_minloglevel"] = "2" 
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

def extract_pose_keypoints_timeseries(video_path, max_frames=None):
    
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    cap = cv2.VideoCapture(video_path)
    
    keypoints_list = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if max_frames and frame_count > max_frames:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            frame_keypoints = []
            for lm in results.pose_landmarks.landmark:
                frame_keypoints.extend([lm.x, lm.y, lm.z])
            keypoints_list.append(frame_keypoints)
        else:
            frame_keypoints = [0] * (33 * 3)
            keypoints_list.append(frame_keypoints)

    cap.release()
    pose.close()

    return keypoints_list

