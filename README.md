# Semantic Segmentation of Driving Videos on Learning based Image Compression
### Official implementation of the paper: Kakaiya, R., Sathish, R., Sheet, D. and Sethuraman, R.  2023. Exploiting Richness of Learned Compressed Representation of Images for Semantic Segmentation. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME) 2023 Brisbane, Australia.

Autonomous vehicles and Advanced Driving Assistance Systems (ADAS) have the potential to radically change the way we travel. Many of such vehicles currently rely on segmentation and object detection algorithms to detect and track objects around its surrounding. The data collected from the vehicles are often sent to cloud servers to facilitate continual/life-long learning of these algorithms. Considering the bandwidth constraints, the data is compressed before sending it to servers, where it is typically decompressed for training and analysis. In this work, we propose the use of a learning-based compression Codec to reduce the overhead in latency incurred for the decompression operation in the standard pipeline. We demonstrate that the learned compressed representation can also be used to perform tasks like semantic segmentation in addition to decompression to obtain the images. We experimentally validate the proposed pipeline on the Cityscapes dataset, where we achieve a compression factor up to 66× while preserving the information required to perform segmentation with a dice coefficient of 0.84 as compared to 0.88 achieved using decompressed images while reducing the overall compute by 11%.


<p align="center">
<img src="media/pipeline.png" alt="seg" style="width:1000px; height:400px" title="Compressor Architecture"/>


<p>
<p align="center">
			    	Overview of proposed method
		<p>

​


>**Paper** : Kakaiya, R., Sathish, R., Sheet, D. and Sethuraman, R.  2023. Exploiting Richness of Learned Compressed Representation of Images for Semantic Segmentation. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME) 2023 Brisbane, Australia. 
</br> 

> Access the paper via the [Arxiv link]( https://arxiv.org/abs/2307.01524)
​




<!---
BibTeX reference to cite, if you use it:

​

```bibtex
@INPROCEEDINGS{9175649,

  author={Sathish, Rakshith and Sathish, Rachana and Sethuraman, Ramanathan and Sheet, Debdoot},

  booktitle={2020 42nd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC)}, 

  title={Lung Segmentation and Nodule Detection in Computed Tomography Scan using a Convolutional Neural Network Trained Adversarially using Turing Test Loss}, 

  year={2020},

  volume={},

  number={},

  pages={1331-1334},

  doi={10.1109/EMBC44109.2020.9175649}}

```
-->
## Dataset used

The Cityscapes dataset was used for our training and evaluation. It has 5,000 images of size 1,024×2,048 with polygon annotations for 34 classes. We use the validation set provided in the dataset as our held-out test set and the training set is divided into training and testing datasets in the ratio of 80:20.

 >Dataset download page: [Click the link.](https://www.cityscapes-dataset.com/)
<!---
>**Note**: Systematically sampled slice numbers/images to be used are given in the repository inside the data preparation folder.
-->

​

## Code and Directory Organization



	Semantic-Segmentation-of-Driving-Videos-on-Learning-based-Image-Compression/
	  media/
	  src/
		compression/
        		utils/
				dataloader.py
    			dataloader_test.py
   				model.py

	      		infer_and_save.py
	      		train.py
  		
  
      	data_preperation/
      			generation_of_patches.py
		
  		sample_images/
    			for_compression/
       				folder_for_inference
	   			for_segmentation/
      				folder_for_inference/
	  					gt/
       						test
	     					lat_5_d2/
	  						test
	   

     	segmentation/
      		libs/
				core/
    				loss.py
	 				operators.py
   				dataloader_latent/
       					latent_city.py
      			models/
	 				DualGCNNet.py
     				GLADNet.py
	 				PSPNet.py
	 			utils/
					image_utils.py
     				logger.py
	  				tools.py
       					
      		utils/
				dataloader.py
   				model.py

	        	infer_and_save.py
	  		infer.sh
     		train_dual_seg_r_50.sh
			train.py
   		requirements_all.txt
   			
	LICENSE
	README.md




## System Specifications

The code and models were tested on a system with the following hardware and software specifications.

- Ubuntu* 16.04

- Python* 3.6

- NVidia* GPU for training

- 16GB RAM for inference

## Installing Dependencies
The provided codes have required dependencies that need to be installed before running them. We have included all the requirements in the requirements_all.txt file which need to be installed. We have utilised the Apex toolkit to use multiple GPU's for parallel training of heavy models. Follow the said procedure to get your system ready.
1. Create a new conda environment.
2. Clone the entie repository in your system.
3. Go to the src directory using
> cd /Semantic-Segmentation-of-Driving-Videos-on-Learning-based-Image-Compression/src
> 
> pip install -r requirements_all.txt
4. Go to the segmentation directory and Install APEX by following the following steps
> git clone https://github.com/ptrblck/apex.git
>
>cd apex
>
>git checkout apex_no_distributed
>
>pip install -v --no-cache-dir ./



## Data Preparation

Follow the below steps to prepare and organize the data for training.

> Details about the arguments being passed and their purpose is explained within the code.
​
> Make sure the dataset has been adequately downloaded and extracted before proceeding.

1. The compression model ( $net_C( . ) - net_D( . )$ ) for all baselines and the proposed method are trained with patches of 256×256, and segmentation models ( $net_{seg}( . ), net_{seg, D'}( . )$ ) were trained using non-overlapping patches of size 840×840, respectively, which were extracted from the training set without any overlapping.
2.  An important point that is to be noted is that the data for training and inference should should be kept folderwise inside a master folder as we have kept in sample_images/for_compression/folder_for_imference and sample_images/for_segmentation/folder_for_inference.
3. The training and inference of the Segmentation model takes ground truths of segmentation and latent vector of the original images as its input. Create patches of 840×840 of the ground truths and original images using the generation_of_patches.py program. The latent vector of those image patches can be obtained using the infer_and_save.py program in the compression directory where the model needs to be the compression model with d=2. 
## Training
> Details about the arguments being passed and their purpose is explained within the code. 

​The compression-decompression and the segmentation training routine are explained adequately in the paper. In the compression block, weights are updated for both  $net_C( . ) - net_D( . )$ with respect to gradients calculated using the reconstruction error between the Input Image and the decompressed Image. We pass the arguements from the command line.For the segmentation model, we use the dual graph convolutional neural network (DGCN) architecture proposed to perform segmentation. The segmentation network $net_{seg}(·)$ consists of a backbone network that provides a feature map X and dual graph convolutional layers, which effectively and efficiently models contextual information for semantic segmentation. We use ResNet-50 architecture as our backbone network.
The compression model $net_C( . ) - net_D( . )$ was trained for 100 epochs with Adam as optimizer using a step learning rate scheduler with an initial learning
rate of $1 × 10^{-2}$, step size of 10 and multiplication factor γ of 0.75. The segmentation decoder ($net_{seg,D′}$) was trained for 40,000 iterations using SGD as the optimizer with an initial learning rate of $1×10^{-3}$. Mean square error and cross-entropy loss were chosen as loss functions for compression and segmentation, respectively.

<p align="center">
<img src="media/compression_pipeline.png" alt="cmpr" style="width:800px; height:400px" title="Compressor Architecture"/>

</p>
<p align="center">
			    	Compressor-Decompressor Architecutre
		<p>
<p align="center">
<img src="media/segmentation_pipeline.png" alt="seg" style="width:800px; height:400px" title="Segmentation Architecture"/>


<p>
<p align="center">
			    	Segmentation Architecture
		<p>


​

## Evaluation


> Details about the arguments being passed and their purpose is explained within the code.

​
The quality of compression in terms of SSIM and pSNR at varying network depth or the number of digest units (d) and bit length (n) is shown in the paper. It can be observed that for all values of d in the range 1 to 3, we do not observe significant degradation in the quality of the decompressed image. For values of n less than 6, we can observe a noticeable drop in performance. Further, we can observe that with a learnable compression codec, we can compress the images up to 200× without a significant drop in performance for a bit length of 8.
In the case of the segmentation model, dice coefficient values for the baselines and $net_{seg,D}(·)$, which is trained using compressed representations, are reported in Table I. The results indicate that $net_{seg,D}(·)$ performs similarly to BL 3 and BL 4 in terms of dice coefficient. This suggests that the compressed representations produced by $net_C(·)$ contain significant semantic information that can be leveraged for other image analysis tasks, even though $net_C(·)$ was not explicitly trained for this purpose. Further, it can be observed that increasing the value of d, which results in a deeper network and higher compression factor, results in poorer reconstruction from the compressed representation owing to loss of information.

## Executing the Code
For the compression module, execute the train.py file for training of the model, and infer_and_save.py for inference and saving the latent vectors. The parameters needs to be passed as arguements for these codes which are well commented.
An example to show how to rub the inference code is 
>  python < location of infer_and_save.py> -m < location of weights> --inferdata < location of the data for inference> --gpu --with_aac -l -d --out_latent < location to save latent vectors> --out_decom < location to save decompressed outputs>


For the execution of the segmentation module, run the train_dual_seg_r_50.sh file for training and the infer.sh file for the inference. They contain the argguements that need to be passed for the train.py and infer_and_save.py files which are actually the programs for training and inference of the model.
An example to show how to run the training and inference code is 
> sh < location of train_dual_seg_r_50.sh>

> sh < location of infer.sh>

## Pre-trained Models
Pretrained models for inference are available [here](http://kliv.iitkgp.ac.in/projects/miriad/model_weights/dl4c/model_weights.zip). 

## Sample Images for inference
The [sample_images](src/sample_images) folder under src contains images for you to test  out.
​
## Sample Output
A comparison with the ground truth, baselines, and model outputs is shown below.
![ravisave](https://github.com/DL4Compression/Exploiting-Richness-of-Learned-Compressed-Representation/assets/118466922/41ed4537-c3c1-4317-8301-185ffb59d677)


## Acknowledgement

**Principal Investigators**

<a href="https://www.linkedin.com/in/debdoot/">Dr Debdoot Sheet</a></a></br>
Department of Electrical Engineering,</br>
Indian Institute of Technology Kharagpur</br>
email: debdoot@ee.iitkgp.ac.in


<a href="https://www.linkedin.com/in/ramanathan-sethuraman-27a12aba/">Dr Ramanathan Sethuraman</a>,</br>
Intel Technology India Pvt. Ltd.</br>
email: ramanathan.sethuraman@intel.com




**Contributors**
<br>
The codes/model were contributed by:
<br>

<a href="https://github.com/ravikakaiya"> Ravi Kakaiya</a>,</br>
Department of Electrical Engineering,</br>
Indian Institute of Technology Kharagpur</br>
email: ravijk8299@kgpian.iitkgp.ac.in</br>
Github username: ravikakaiya


<a href="https://github.com/Rakshith2597"> Rakshith Sathish</a>,</br>
Advanced Technology Development Center,</br>
Indian Institute of Technology Kharagpur</br>
email: rakshith.sathish@kgpian.iitkgp.ac.in</br>
Github username: Rakshith2597


<a href="https://github.com/PragyadiptaAdhya"> Pragyadipta Adhya</a>,</br>
Department of Electrical Engineering,</br>
Indian Institute of Technology Kharagpur</br>
email: pragyadipta.adhya@kgpian.iitkgp.ac.in</br>
Github username: PragyadiptaAdhya 
​

## References

<div id="densenet">
<a href="#abs">[1]</a> Kakaiya, R., Sathish, R., Sheet, D. and Sethuraman, R.  2023. Exploiting Richness of Learned Compressed Representation of Images for Semantic Segmentation. In Proceedings of the IEEE International Conference on Multimedia and Expo (ICME) 2023 Brisbane, Australia. </a> 

</div>
