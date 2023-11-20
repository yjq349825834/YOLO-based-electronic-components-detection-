# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:53:44 2022

@author: ye
"""
import torch
import numpy as np
import cv2
from time import time
import sys

sys.path.insert(0, 'C:/Users/Jiaqi Ye/Desktop/DMFOHKSPQJVBN/Interesting projects to recap/Side Projects/electronic-components/Model/YOLOV8')


from yolov8_electronics import YOLOV8_detector



dir_model = 'C:/Users/Jiaqi Ye/Desktop/DMFOHKSPQJVBN/Interesting projects to recap/Side Projects/electronic-components/Model/yolov5-master/'


class YOLOV5_detector():
    """
    Class implements Yolo5 model to make inferences on a youtube video using Opencv2.
    """

    def __init__(self, capture_index, model_name):
        """
        Initializes the class with youtube url and output file.
        :param url: Has to be as youtube URL,on which prediction is made.
        :param out_file: A valid output file name.
        """
        # self.port = serialPort
        # self.baud = serialBaud
        # self.serialConnection = serial.Serial(serialPort, serialBaud, timeout=4)
        self.capture_index = capture_index
        self.model = self.load_model(model_name)
        self.classes = self.model.names
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

    
    def get_video_capture(self):
        """
        Creates a new video streaming object to extract video frame by frame to make prediction on.
        :return: opencv2 video capture object, with lowest quality frame available for video.
        """
      
        return cv2.VideoCapture(self.capture_index)

    def load_model(self, model_name):
        """
        Loads Yolo5 model from pytorch hub.
        :return: Trained Pytorch model.
        """
        if model_name:
            model = torch.hub.load(dir_model, 'custom', source='local', path=model_name, force_reload=True)
        else:
            model = torch.hub.load(dir_model, 'yolov5s',source='local', pretrained=True)
        return model

    def score_frame(self, frame):
        """
        Takes a single frame as input, and scores the frame using yolo5 model.
        :param frame: input frame in numpy/list/tuple format.
        :return: Labels and Coordinates of objects detected by model in the frame.
        """
        self.model.to(self.device)
        frame = [frame]
        results = self.model(frame)
        # print(results.xyxy[0][:, :-1])
        labels, cord = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        # print(labels,cord)
        return labels, cord

    def class_to_label(self, x):
        """
        For a given label value, return corresponding string label.
        :param x: numeric label
        :return: corresponding string label
        """
        return self.classes[int(x)]
    
    def class_to_color(self,y):
        
        colors = [(170, 170, 255),(0, 72, 255),(255,255,86),(0,255,255),(0,255,0),(255,0,0)]
        
        return colors[int(y)]
        

    def plot_boxes_serialWrite(self, results, frame):
        """
        Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
        :param results: contains labels and coordinates predicted by model on the given frame.
        :param frame: Frame which has been scored.
        :return: Frame with bounding boxes and labels ploted on it.
        """
        labels, cord = results
        # print('labels:',labels)
        # print('cord:',cord)
        n = len(labels)
        # print(n)
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i in range(n):
            row = cord[i]
            # print(row[4])
            if row[4] >= 0.5:
                # print(row[4].item())
                x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
                cv2.rectangle(frame, (x1, y1), (x2, y2), self.class_to_color(labels[i]), 2)
                cv2.putText(frame, self.class_to_label(labels[i])[-7:], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.class_to_color(labels[i]), 2)
                cv2.putText(frame, str(round(row[4].item(),2)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                # if self.class_to_label(labels[i]) == 'left':
                #     # print(self.class_to_label(labels[i]))
                #     self.s.serialConnection.write(b'L')
                # if self.class_to_label(labels[i]) == 'right':
                #     # print(self.class_to_label(labels[i]))
                #     self.s.serialConnection.write(b'R')
        return frame
    
    # def to_arduino(self,results):
    #     labels, cord = results 
    #     n = len(labels)
    #     conf = []
    #     for i in range(n):
    #         row = cord[i]
    #         conf.append(row[4])   
    #         if row[4] >= 0.3:
    #             label = labels[i]
    #     return label
    
    def serialRead(self):
        line = self.s.serialConnection.readline().decode() 
        return line

    def __call__(self):
        """
        This function is called when class is executed, it runs the loop to read the video frame by frame,
        and write the output into a new file.
        :return: void
        """
        cap = self.get_video_capture()
        assert cap.isOpened()
        # We need to set resolutions.
        # so, convert them from float to integer.
        # frame_width = int(cap.get(3))
        # frame_height = int(cap.get(4))
           
        # size = (frame_width, frame_height)
           
        # Below VideoWriter object will create
        # a frame of above defined The output 
        # is stored in 'filename.avi' file.
        # result = cv2.VideoWriter('iphone_spring_detection2.avi', 
        #                          cv2.VideoWriter_fourcc(*'MJPG'),
        #                          30, (1920,1080))
        
        count = 0
        
        while True:
          
            ret, frame = cap.read()
            # frame = cv2.flip(frame, 1) 
            # flip the image is important for the model to indentify left&right collectly!!!
            assert ret
            count+=1
            
            if count % 2 == 0:
                
                frame = cv2.resize(frame, (640,640))
                
                start_time = time()
                results = self.score_frame(frame)
                
                # command = self.serialRead()
                # if command == 'on':
                    # print('triggered')
                frame = self.plot_boxes_serialWrite(results, frame)
                    # test = self.to_arduino(results)
                    # print(test)
                frame = cv2.resize(frame, (1920,1080))
                
                end_time = time()
                fps = 1/np.round(end_time - start_time, 2)
                #print(f"Frames Per Second : {fps}")
                 
                cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
                
                cv2.imshow('YOLOv5 Detection', frame)
                # write the flipped frame
                # result.write(frame)
 
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
      
        cap.release()
        
        
        
if __name__ == '__main__':        
    # Create a new object and execute.
    capture_index = 0
    
    # detector = YOLOV5_detector(capture_index, model_name='best.pt')
    
    detector = YOLOV8_detector(capture_index, model_name='C:/Users/Jiaqi Ye/Desktop/DMFOHKSPQJVBN/Interesting projects to recap/Side Projects/electronic-components/Model/YOLOV8/best.pt')
    # s = serialComm()
    detector()