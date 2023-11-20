1. Online YOLO respoitory is always updated. After training a customed YOLO model online (ie.,the weights file best.pt), 
always remember to download the lastest "yolo-master" folder from the respoitory, to have all compatible necessarties 
for your trained model.

2. Sometimes, the exiting python environement may not support the requirements of your new model (with some packages 
   out of date)
   
   The best practice is to create a new virtual env, using  conda create --name name_of_your_new_env

   And then install the complier with lastest python, using conda install -c anaconda spyder

   Next, follow the 'requirements.txt' file in the 'yolo-master' folder to install all necessary packages
   
   pip install -r requirements.txt

3. The requirements.txt file only help you install the cpu version torch, if you want to run your model on GPU. 
   
   You need to uninstall the installed torch and its dependencies, using pip uninstall torch torchvision torchaudio

4. Then Check your system's CUDA verison (using nvidia-smi). And then follow the instructions here (pip works better than conda 
   in this case): 

   https://pytorch.org/get-started/locally/
 
   to install compatible CUDA version torch for your environment. 

   *Usually the higher version CUDA (eg: 12.2) supprt lower version torch (torch 2.1.0+cu11.8), so you may need to update your GPU driver and CUDA via:
    
   https://www.nvidia.com/Download/index.aspx?lang=en-us
