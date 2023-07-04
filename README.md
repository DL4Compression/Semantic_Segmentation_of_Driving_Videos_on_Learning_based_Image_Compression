# Exploiting-Richness-of-Learned-Compressed-Representation
Official implementation of the paper "Exploiting Richness of Learned Compressed Representation of Images for Semantic Segmentation"

Autonomous vehicles and Advanced Driving Assistance Systems (ADAS) have the potential to radically change the way we travel. Many of such vehicles currently rely on segmentation and object detection algorithms to detect and track objects around its surrounding. The data collected from the vehicles are often sent to cloud servers to facilitate continual/life-long learning of these algorithms. Considering the bandwidth constraints, the data is compressed before sending it to servers, where it is typically decompressed for training and analysis. In this work, we propose the use of a learning-based compression Codec to reduce the overhead in latency incurred for the decompression operation in the standard pipeline. We demonstrate that the learned compressed representation can also be used to perform tasks like semantic segmentation in addition to decompression to obtain the images. We experimentally validate the proposed pipeline on the Cityscapes dataset, where we achieve a compression factor up to 66× while preserving the information required to perform segmentation with a dice coefficient of 0.84 as compared to 0.88 achieved using decompressed images while reducing the overall compute by 11%.

​

​

>**Paper** : Ravi Kakaiya, Rakshith Sathish, Debdoot Sheet, Ramananthan Sethuraman **"Exploiting Richness of Learned Compressed
Representation of Images for Semantic Segmentation"** . </br> 

> _Access the paper via_ 
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

 >Dataset download page: [https://www.cityscapes-dataset.com/] 
<!---
>**Note**: Systematically sampled slice numbers/images to be used are given in the repository inside the data preparation folder.
-->

​

## Code and Directory Organization

​

richness_of_learned_representation/

	src/

      	utils/

        data_prep/

        	generation_of_patches.py

        compression/
		dataloader/
			compression_train_dataset.py
   		model/
     			model_without_batchnorm.py
		training_and_inference/
			compression_training.py
        		infer_save_lat_decom.py

        segmentation/
		dataloader/
			datasets.py 
   		model/
  			SEG_DECODER.py
     		training_and_inference/
    			training.py
      			eval_lat.py
		

     

## System Specifications

​

The code and models were tested on a system with the following hardware and software specifications.

- Ubuntu* 16.04

- Python* 3.6

- NVidia* GPU for training

- 16GB RAM for inference

​

# Using the code

​


## Dataset used and Data Preparation

Follow the below steps to prepare and organize the data for training.

> Details about the arguments being passed and their purpose is explained within the code.
​

>  Make sure the dataset has been adequately downloaded and extracted before proceeding.

1. The compression model ( $net_C( . ) - net_D( . )$ ) for all baselines and the proposed method are trained with patches of 256×256, and segmentation models ( $net_{seg}( . ), net_{seg, D'}( . )$ ) were trained using non-overlapping patches of size 840×840, respectively, which were extracted from the training set without any overlapping.

​
<!---
1. ` python prepare_data.py  --genslices --masktype <type> --datasetpath <path> --save_path <path>  `

	This step extracts induvidual CT slices from the CT volumes provided in the dataset. Each of these slices are saved seperately as npy files with filename in the format `[series_uid]_slice[sliceno].npy`.

	Perform the above step for masktype nodule and lung seperately before proceeding to the next step.

​

2. `python prepare_data.py --createfolds --datapath <path> --savepath <path>

--datasetpath <path> `

	The above step first classifies the slices into two categories, positive and negative based on the presence of nodules in them. On completion, the dataset consists of CT volumes from 880 subjects, provided that ten subsets is divided into 10-folds for cross-validation. In each fold of the experiment, eight subsets from the dataset are separated for training and one each for validation and testing. A balanced dataset consisting of an equal number (approx.) of positive and negative slices is identified for each fold. Filenames of these slices of each fold are stored in separate JSON files.

​

3. `python prepare_data.py --genpatch --jsonpath <path> --foldno <int> --category <type> --data_path <path> --lungsegpath <path> --savepath <path> --patchtype <type> `

	The above step generates patches which are used to train the classifier network.

​

4. `python prepare_data.py --visualize --seriesuid <str> --sliceno <int> --datapath <path> --savepath <path>`

	To visualize a particular slice use the above line. 

​-->

## Training

​

> Details about the arguments being passed and their purpose is explained within the code. <!---To see the details run `python train_network.py -h` -->

​The compression-decompression and the segmentation training routine are explained adequately in the paper. In the compression block, weights are updated for both  $net_C( . ) - net_D( . )$ with respect to gradients calculated using the reconstruction error between the Input Image and the decompressed Image. For the segmentation model, we use the dual graph convolutional neural network (DGCN) architecture proposed to perform segmentation. The segmentation network $net_{seg}(·)$ consists of a backbone network that provides a feature map X and dual graph convolutional layers, which effectively and efficiently models contextual information for semantic segmentation. We use ResNet-50 architecture as our backbone network.
The compression model $net_C( . ) - net_D( . )$ was trained for 100 epochs with Adam as optimizer using a step learning rate scheduler with an initial learning
rate of $1 × 10^{-2}$, step size of 10 and multiplication factor γ of 0.75. The segmentation decoder ($net_{seg,D′}$) was trained for 40,000 iterations using SGD as the optimizer with an initial learning rate of $1×10^{-3}$. Mean square error and cross-entropy loss were chosen as loss functions for compression and segmentation, respectively.


<!---

To train the lung segmentation network without the discriminator execute the following line of code.

`python train_network.py --lungseg --foldno <int> --savepath <path> --jsonpath <path> --datapath <path> --lungsegpath <path> --network <str> --epochs <int:optional> `

​

To train the lung segmentation network with discrimator and turing test loss, execute 

`python train_network.py --lungsegadv --foldno <int> --savepath <path> --jsonpath <path> --datapath <path> --lungsegpath <path> --network <str> --epochs <int:optional> `

​

To train the patch classifier network execute

`python patch_classifier --savepath <path> --imgpath <path> --epochs <int:optional> `

​-->

​

## Evaluation

​

> Details about the arguments being passed and their purpose is explained within the code.

​
The quality of compression in terms of SSIM and pSNR at varying network depth or the number of digest units (d) and bit length (n) is shown in Fig. 5 and Fig. 6, respectively. It can be observed that for all values of d in the range 1 to 3, we do not observe significant degradation in the quality of the decompressed image. However, as shown in Fig. 5 and Fig. 6 for values of n less than 6, we can observe a noticeable drop in performance. Further, we can observe that with a learnable compression codec, we can compress the images up to 200× without a significant drop in performance for a bit length of 8.
In the case of the segmentation model, dice coefficient values for the baselines and $net_{seg,D}(·)$, which is trained using compressed representations, are reported in Table I. The results indicate that $net_{seg,D}(·)$ performs similarly to BL 3 and BL 4 in terms of dice coefficient. This suggests that the compressed representations produced by $net_C(·)$ contain significant semantic information that can be leveraged for other image analysis tasks, even though $net_C(·)$ was not explicitly trained for this purpose. Further, it can be observed that increasing the value of d, which results in a deeper network and higher compression factor, results in poorer reconstruction from the compressed representation owing to loss of information


<!---
To evaluate the segmentation models execute

 `python inference.py --lunseg --foldno <int> --savepath <path> --jsonpath <path> --network <str>`

​

To evaluate the classifier network execute

`python inference.py --patchclass --savepath <path> --imgpath <path>`
-->
​

## Pre-trained Models

​

Pretrained models for inference are available in the code section. 

​
​

## Acknowledgement

**Principal Investigators**

​

<a href="https://www.linkedin.com/in/debdoot/">Dr Debdoot Sheet</a></a></br>

Department of Electrical Engineering,</br>

Indian Institute of Technology Kharagpur</br>

email: debdoot@ee.iitkgp.ac.in

​

<a href="https://www.linkedin.com/in/ramanathan-sethuraman-27a12aba/">Dr Ramanathan Sethuraman</a>,</br>

Intel Technology India Pvt. Ltd.</br>

email: ramanathan.sethuraman@intel.com

​

**Contributor**

​

The codes/model were contributed by

​
<a href="https://github.com/ravikakaiya"> Ravi Kakaiya</a>,</br>

Department of Electrical Engineering,</br>

Indian Institute of Technology Kharagpur</br>

email: ravijk8299@kgpian.iitkgp.ac.in</br>

Github username: ravikakaiya

​


<a href="https://github.com/Rakshith2597"> Rakshith Sathish</a>,</br>

Advanced Technology Development Center,</br>

Indian Institute of Technology Kharagpur</br>

email: rakshith.sathish@kgpian.iitkgp.ac.in</br>

Github username: Rakshith2597

​

## References

​

<div id="densenet">

<a href="#abs">[1]</a> Ravi Kakaiya, Rakshith Sathish, Debdoot Sheet, Ramananthan Sethuraman "Exploiting Richness of Learned Compressed Representation of Images for Semantic Segmentation" .  </a> 

</div>
