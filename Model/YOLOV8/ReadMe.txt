Ultralytics YOLOV8 is the latest version of the YOLO. It support promient
object detection and image segmentation. 

ref:https://docs.ultralytics.com/models/yolov8/#supported-modes

1. YOLOv8 can be installed in two ways :from the source and via pip. 
   This is because it is the first iteration of YOLO to have an official package.

e.g. From pip (recommended)

pip install ultralytics


2. After intalling the ultralytics package, you can use the pretrained 
   yolo models (pretrained on COCO dataset: https://paperswithcode.com/dataset/coco)
   directly with Python interface or CLI (Command Line Interface) command:

e.g. Python:
   
     from ultralytics import YOLO

     # Create a new YOLO model from scratch
     model = YOLO('yolov8n.yaml')

     # Load a pretrained YOLO model (recommended for training)
     model = YOLO('yolov8n.pt')

     # Train the model using the 'coco128.yaml' dataset for 3 epochs
     results = model.train(data='coco128.yaml', epochs=3)

     # Evaluate the model's performance on the validation set
     results = model.val()

     # Perform object detection on an image using the model
     results = model('https://ultralytics.com/images/bus.jpg')

     # Export the model to ONNX format
     success = model.export(format='onnx')

     CLI:
     
     # Train a detection model for 10 epochs with an initial learning_rate of 0.01
     yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01

     # Predict a YouTube video using a pretrained segmentation model at image size 320:
     yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

Ref： https://docs.ultralytics.com/usage/python/


3. Similar to previous YOLOs, YOLOV8 aslo supports custom training using
   Python and CLI commands.
   
   Still, it is recommanded to use cloud computing paltforms so that you can
   take the advantage of advanced GPU accelearation resources. 
   
   Ref: https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb
        https://blog.roboflow.com/how-to-train-yolov8-on-a-custom-dataset/
   
   This example uses iteractive Python Notebook (.ipynb) in Google Colab.

   Main steps for custom training:
   
   -Install YOLOv8
   -Prepare and load a custom dataset using Roboflow
   -Train a pre-trained yolo model on the custom dataset
   -Validate and deploy the trianed model (best.pt)
   
   Tips:
   * When intalling the YOLOV8 (ultralytics), try to match the version of
     YOLOv8 you installed on the device you are going to deploy. 
     
     e.g. If you are going to deploy the custom model on to your PC, and the 
     ultralytics you installed on your PC is version  8.0.201, then when you
     train the model online, you can contrain the version of the ultralytics
     package to 8.0.201 by using: 

     !pip install ultralytics==8.0.201

   * YOLO model itself is created in PyTroch, so to run the YOLO model locally 
     you need intall pytorch (CPU or GPU) and other dependencies in requirements.txt
   

   * the configuration file (data.yaml) loaded from roboflow lasks the path 
     of the dataset, this needs to be added manully for now, as:
   e.g.
       path: /content/datasets  # root path to the custom dataset
     (This may be fixed in the future)
    
   * After training, the 'best.pt' containing the trained weights can be used 
     locally.
   e.g. Following step-2 to use it on a local pc
   
   or be deployed to an online API.
   e.g. to Robowflow API, using Roboflow(api).project(project).deply(model_type, model_path)
   
   or be deployed on to other edge devices in supported model formats 
  (e.g. ONNX for raspberry pi in Arm Cortex 64 bit CPU, TF Lite for openMV, 
        coreML for Apple ios App, TF json for web API )
   Ref: https://docs.ultralytics.com/integrations/#deployment-integrations
   
   