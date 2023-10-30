# hack2022-02-segmentation
# Challenge 2: Towards artefact-robust cell segmentation models

### *Create a single cell segmentation algorithm that is robust to artefacts*

## Challenge Description: 
This challenge proposes the training of cell segmentation models that are robust against visual artefacts (specifically debris, fluorescent antibody aggregates, illumination artefacts) in multiplex images of normal and diseased human tissue. We envision that model training be based on manual annotations of cells and artefact instances in multiplex images of tissue. The set of density-based cluster algorithms implemented by CyLinter (https://github.com/labsyspharm/cylinter) will be used to evaluate model performance.

![](https://github.com/IAWG-CSBC-PSON/hack2022-02-segmentation/blob/main/robustsegmentationmodels.jpg)

## Background:
Segmentation models can produce ambiguous and unexpected results when low-quality images are used for training. Existing methods based on machine learning techniques are trained on multiple object classes to delineate foreground signals (i.e. cells) from background signals (i.e. areas lacking biological sample). Although existing models such as UNet and MaskRCNN have extensive training data on manually curated foreground objects (i.e. nuclei/cells), they lack quality control (QC) annotations for artefacts. This results in the aberrant classification of  antibody aggregates, out-of-focus cells, illumination artefacts, and other spurious imaging aberrations as cells. State-of-the-art methods for identifying cells corrupted by these microscopy artefacts involve user-guided curation of artefacts that stem from multiple sources throughout the experimental process. While this approach is effective in cases of limited samples (between 12-24) probed with relatively few immunomarkers (20-30), it fails with larger datasets, as the number of channels required for curation scales rapidly as the product of tissues by immunomarkers.


## Ground truth annotations and data provided: 
Training/validation set: <br>
- fluorescence images of nuclei stained with Hoechst from different tissues with corresponding manual annotations of nuclei contours and background
- Manual annotations of artefacts (lint, fluorescent blobs, illumination tiles) and corresponding fluorescence images stained with Hoechst <br>

Test set: <br>
* A small set fluorescence images of nuclei stained with Hoechst with visible artefacts in the same field and corresponding manual annotations


## All data are available from [Synapse](https://www.synapse.org/#!Synapse:syn26848606)  

### To jump ahead and save some time, some of the data has been prepared in advance. Download this zip file [https://www.synapse.org/#!Synapse:syn27087655](https://www.synapse.org/#!Synapse:syn27087655)
#### 1 Tissues trainingset
-Ground truth annotations have suffix 'Ant' and are in .png format. Size is 256 x 256 px. Classes correspond to pixel values: 1 - background, 2 - nuclei outlines, 3 - nuclei centers, 4 - difficult pixels <br>
-Images have suffix 'Img' and are in .tif format. They are 2 channels (DAPI and lamin). Size is 256 x 256 px. <br>
-Weight annotations for incurring higher penalty in loss function 'wt' and are in .png format. Size is 256 x 256 px. <br>

#### 2 Artefacts trainingset 
-The ground truth is 33-Ant.tif. This is a 15,000 x 15,000 x 25 voxels 3D image!
-The image has is 33-Img.tif. This is 15,000 x 15,000 x 25 x 2 voxels 3D image **It has TWO channels just like the tissues trainingset - DAPI and lamin** <br>
The classes have values: 1 - blobs, 2 - lint, 3 - tile artefacts, 4 - false positives (do not let model classify these as artefacts). All other pixels can be considered as background.

#### 3 Artefacts testset
-Ground truth annotations have suffix 'Ant' and are in .png format. Size is 256 x 256 px. Classes correspond to pixel values: 1 - artefact (background), 2 - nuclei outlines, 3 - nuclei centers,  <br>
-Images have suffix 'Img' and are in .tif format. They are 2 channels (DAPI and lamin). Size is 256 x 256 px. <br>


### To start from scratch... 
#### The ground truth annotations are in .png file extensions. The class index is in the filename: 
-class1 - nuclei contours (outlines) <br>
-class2 - nuclei centers on the border. For other nuclei, use an imfill operation. <br>
-class3 - NO CLASS 3. SKIP. <br>
-class4 - background pixels <br>
-class5 - pixels that correspond to areas that models generally find challenging. Nuclei are more dense here. You can use these to add a higher penalty to your loss function to get higher accuracy. <br>
<br>
#### Images are in .tif file extensions
-DAPI - corresponding to channel 1 of DAPI-stained nuclei. DAPI stains DNA in cells.<br>
-lamin - corresonding to channel 2 of the same nuclei. Lamin stains the outer membrane in nuclei producing a ring. This is useful in challenging areas for segmentation.<br>
-stack - OPTIONAL. This is the DAPI and lamin channels concatenated into one multipage file.<br>
<br>
# Suggested procedure<br>
1. Set the artefacts testset folder aside. Use as test set later.<br>
2. Split the tissue trainingset into training, validation, test sets of your choosing. You may want to slice the image into smaller tiles. <br>
3. From the artefacts trainingset, extract regions containing artefacts from the image based on the ground truth annotation.

# Tips <br>
1. use the wt files to increase penalty between nuclei in loss function. This will increase accuracy.

## Compute requirements:
* GPU: ie 1080GTX, 3070 RTX, etc 
* RAM: 64GB

## Recommended software:
* Fiji or Napari
* If programming in Python, install scikit-image, opencv, Tensorflow, PyTorch
* If programming in MATLAB, install image processing toolbox
