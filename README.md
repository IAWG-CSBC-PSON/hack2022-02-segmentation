# hack2022-02-segmentation
# Challenge 2: Towards artefact-robust cell segmentation models

### *Create a single cell segmentation algorithm that is robust to artefacts*

## Challenge Description: 
This challenge proposes the training of cell segmentation models that are robust against visual artefacts (specifically debris, fluorescent antibody aggregates, illumination artefacts) in multiplex images of normal and diseased human tissue. We envision that model training be based on manual annotations of cells and artefact instances in multiplex images of tissue. The set of density-based cluster algorithms implemented by CyLinter (https://github.com/labsyspharm/cylinter) will be used to evaluate model performance.

## Background:
Segmentation models can produce ambiguous and unexpected results when low-quality images are used for training. Existing methods based on machine learning techniques are trained on multiple object classes to delineate foreground signals (i.e. cells) from background signals (i.e. areas lacking biological sample). Although existing models such as UNet and MaskRCNN have extensive training data on manually curated foreground objects (i.e. nuclei/cells), they lack quality control (QC) annotations for artefacts. This results in the aberrant classification of  antibody aggregates, out-of-focus cells, illumination artefacts, and other spurious imaging aberrations as cells. State-of-the-art methods for identifying cells corrupted by these microscopy artefacts involve user-guided curation of artefacts that stem from multiple sources throughout the experimental process. While this approach is effective in cases of limited samples (between 12-24) probed with relatively few immunomarkers (20-30), it fails with larger datasets, as the number of channels required for curation scales rapidly as the product of tissues by immunomarkers.


## Ground truth annotations and data provided: 
Training/validation set: <br>
- fluorescence images of nuclei stained with Hoechst from different tissues with corresponding manual annotations of nuclei contours and background
- Manual annotations of artefacts (lint, fluorescent blobs, illumination tiles) and corresponding fluorescence images stained with Hoechst <br>

Test set: <br>
* A small set fluorescence images of nuclei stained with Hoechst with visible artefacts in the same field and corresponding manual annotations


## Compute requirements:
* GPU: ie 1080GTX, 3070 RTX, etc 
* RAM: 64GB

## Recommended software:
* Fiji or Napari
* If programming in Python, install scikit-image, opencv, Tensorflow, PyTorch
* If programming in MATLAB, install image processing toolbox
