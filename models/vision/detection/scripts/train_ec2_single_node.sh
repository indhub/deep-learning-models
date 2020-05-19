# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env bash
NUM_GPU=${1:-1}
TRAIN_CFG=$2

echo ""
echo "NUM_GPU: ${NUM_GPU}"
echo "TRAIN_CFG: ${TRAIN_CFG}"
echo ""

cd /deep-learning-models/models/vision/detection
export PYTHONPATH=${PYTHONPATH}:${PWD}

/shared/bin/herringrun -n 8 \
-x PYTHONPATH \
python tools/train.py ${TRAIN_CFG} \
--validate \
--autoscale-lr \
--amp

