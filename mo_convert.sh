#!/bin/bash
# ==============================================================================
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================

model=$1

mkdir temp_build
cd temp_build

python3 -m venv env
source env/bin/activate

pip install --upgrade pip
pip install tensorflow==2.4.4 --force-reinstall
pip install networkx defusedxml requests
pip install pillow
if [ "$model" = "saved_model" ]; then
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --saved_model_dir "../results/xeon/cascade_lake/checkpoints" \
          --input_shape [1,150,150,3] \
          --output_dir "../models/ov/FP32" \
          --data_type FP32 
else
python3 /opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py --input_model "../results/xeon/cascade_lake/frozen_histology.pb" \
          --input_shape [1,150,150,3] \
          --output_dir "../models/ov/FP32" \
          --data_type FP32 
fi
deactivate
rm -rf ../temp_build
