## End-to-End Supervised Lung Lobe Segmentation
In this project we present a fully automatic and supervised approach to the problem of the segmentation of the pulmonary lobes from a CT scan.
A 3D fully convolutional neural network was used based on the V-Net wich we called Fully Regularized V-Net (FRV-Net).
This work was performed in the Biomedical Imaging group at C-BER centre of INESC TEC, Portugal and it resulted in the paper "End-to-End Supervised Lung
Lobe Segmentation" accepted to the IJCNN2018 conference.
Here are the code and scripts to train our FRV-Net (as you select wich regularization techniques do you want) and to run the segmentations.


## Running a single segmentation with a pre-trained model.
To run a single segmentation with a pre-trained model a example file called "run_single_segmentation.py" is available. 
It teaches you how to open a CT scan, to open the model and to predict and save the segmentation.


## Train a model
If you want to train your model, a file called "train.py" is available.
It allows you to set the specific regularization techniques and parameters of the desired net.
	
	-path : Model path		  		(path)
	-train: Train data		  		(path)
	-val  : Validation data		 		(path)
	-lr   : Set the learning rate     		(float)
	-load : load a pre-trained model  		(boolean)
	-aux  : Multi-task learning	  		(float - weight in the loss function)
	-ds   : Number of Deep Supervisers		(int   - nº of layers)
	-bn   : Set Batch normalization  		(boolean)
	-dr   : Set Dropout              		(boolean)
	-fs   : Number of initial of conv channels	(int)

The train and validation datasets has to contain two folders A and B. where the folder A contains the CT scans and the B the correspondent ground-truth.
In the script file "train_session.sh", the examples used for our results are presented.

## Software
Our project was developed using Python (2.7) and Keras (2.0.4) framework that are required to use it.


## Expected usage

From looking at the paper ( https://ieeexplore.ieee.org/document/8489677 ), the training data is:
* windowed between [-1000, 400];
* of size 512, 512, 256;
* max voxel resolution of 1mm;
* the volume is axially cropped to around the lung volume;
* the input image gets resampled to 256, 256, 128;
* patch size is 128, 128, 64;

Note:
* this implies that the runtime of an inference is approximately the same for all scans, as they all get resampled to the same size;
* the windowing is completed within the inference, no need to window the scan before sending for inference;
* to get the best results for a scan, the trick is to axially crop the scan to the lung region before sending to end2end for the final lobe seg.
* POTENTIAL APPROACH: use end2end to do a first pass, remove all connected components that lie on an edge voxel (artefacts cluster there), then find the 5 largest connected components for the remaining labels, axially crop to include those components only. Repeat process. QA using stats based on final segmentations e.g. lobe volumes, relative lobe centre-of-mass distribution, etc.
