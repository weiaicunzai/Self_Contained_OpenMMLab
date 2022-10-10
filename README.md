# Self Contained OpenMM Lab
[OpenMM lab](https://github.com/open-mmlab) is a great open source codebase for fast reproducing SOTA results. 
However, for some cases, it is not flexible enough as plain Pytorch and requires a lot of modifications.
For example: (1) When the model is none end-to-end trainable. (2) The dataset folder structure are fixed, do not support other dataset formats, like LMBD. (3) Save the certain feature maps of the models for visualization.
In order to allivate these limitations, I remove the the dependency of openmmlab modules (e.g. image transform module), to make these module self-contained.
These self-contained modules can be easily integrated into your research project by simply copying and pasting them.
This project is suitable for research projects.

***Note: Since mmcv is a foundational library for computer vision research and supports many openmmlab code bases, and installing mmcv through pip is fairly easy. So I keep dependecies of mmcv in this project to keep the code simple and clean.***


## Structure

### mmsegmentation

#### data augmentation
mmsegmentation.mmseg.datasets.pipelines.transforms.py --> mmsegmentation.transform.py
mmsegmentation.mmseg.datasets.pipelines.formatting.py --> mmsegmentation.transform.py
mmsegmentation.mmseg.datasets.pipelines.compose.py --> mmsegmentation.transform.py