# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:14:42 2023

@author: Jiaqi Ye
"""

import cv2
from ultralytics import YOLO



class YOLOV8_detector():
    def __init__(self, capture_index, model_name = 'best.pt'):
        self.model = YOLO(model_name)
        self.cap = cv2.VideoCapture(capture_index)
    
    def __call__(self):
        while True:
            success, img = self.cap.read()
            
            if success:
                results  = self.model(img, stream=True, conf = 0.5)
                results = list(results)
                
                for result in results:
                    keypoints = result.keypoints  # Keypoints object for pose outputs
                    print(keypoints)
                    
                # Visualize the results on the frame
                annotated_frame = results[0].plot()    
                cv2.imshow("Image", annotated_frame)
                
            else:
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            
#%%    

# cap = cv2.VideoCapture(0)
# model = YOLO('best.pt')

# while True:
#     success, img = cap.read()
    
#     if success:
#         results  = model(img, stream=True)
#         results = list(results)
        
#         for result in results:
#             keypoints = result.keypoints  # Keypoints object for pose outputs
#             print(keypoints)
            
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()    
#         cv2.imshow("Image", annotated_frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
