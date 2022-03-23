## Histology Usecase Demo with Intel OpenVINO™ 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/junwenwu/histology/master?labpath=histology_demo.ipynb)

## Introduction

This demo uses the [colorectal histology images dataset](https://www.tensorflow.org/datasets/catalog/colorectal_histology) to train a simple convolutional neural network in TensorFlow and demonstrates how to use OpenVINO™ integration with Tensorflow and OpenVINO™ Inference Engine to do inference on different Hardware architecture.

All images are RGB, 0.495 µm per pixel, digitized with an Aperio ScanScope (Aperio/Leica biosystems), magnification 20x. Histological samples are fully anonymized images of formalin-fixed paraffin-embedded human colorectal adenocarcinomas (primary tumors) from our pathology archive (Institute of Pathology, University Medical Center Mannheim, Heidelberg University, Mannheim, Germany).

The model is trained using the histology model introduced here: [Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zollner F: Multi-class texture analysis in colorectal cancer histology (2016), Scientific Reports (in press)] (https://zenodo.org/record/53169#.X_T3iC1h10v)



### Key concepts
This sample application includes an example for the following:
- Application:
  - Load and visualize Tensorflow dataset
- Intel® DevCloud for the Edge: submitting jobs to perform on different edge compute nodes (rather than on the development node hosting this Jupyter* notebook)
  - Training jobs that train a convolutional neural network with Tensorflow V2
  - Running inference jobs with OpenVINO™ integration with Tensorflow
  - Running Inference jobs with OpenVINO™ Inference Engine
  - Monitoring job status
  - Viewing results and assessing performance for hardware on different compute nodes
- [Intel® Distribution of OpenVINO™ toolkit](https://software.intel.com/openvino-toolkit):
  - Create the necessary Intermediate Representation (IR) files for the inference model using [Model Optimizer](http://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
  - Run an inference application on multiple hardware devices using the [Inference Engine](http://docs.openvinotoolkit.org/latest/_docs_IE_DG_Deep_Learning_Inference_Engine_DevGuide.html)
