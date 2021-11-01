# coding=utf-8
# Copyright 2020 The Compressive Visual Representations Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash

# Architecture settings
RESNET_DEPTH=50
WIDTH_MULTIPLIER=1

CONFIG="csimclr_300k"
TRAIN_BATCH_SIZE=4096
EVAL_BATCH_SIZE=256

if [[ $CONFIG == "simclr_300k" ]]
then
  ADDITIONAL_FLAGS="--pretrain_loss=contrastive --train_epochs=300 --weight_decay=1e-6 --learning_rate=0.3"
elif [[ $CONFIG == "simclr_1000k" ]]
then
  ADDITIONAL_FLAGS="--pretrain_loss=contrastive --train_epochs=1000 --weight_decay=1.5e-6 --learning_rate=0.2"
elif [[ $CONFIG == "csimclr_300k" ]]
then
  ADDITIONAL_FLAGS="--pretrain_loss=contrastive_ceb --ceb_beta=1.0 --kappa_e=1024.0 --kappa_b=10.0 --train_epochs=300 --weight_decay=1e-6 --learning_rate=0.3"
elif [[ $CONFIG == "csimclr_1000k" ]]
then
  ADDITIONAL_FLAGS="--pretrain_loss=contrastive_ceb --ceb_beta=1.0 --kappa_e=1024.0 --kappa_b=10.0 --train_epochs=1000 --weight_decay=1.5e-6 --learning_rate=0.2"
else
  echo "Unknown config"
  exit 1
fi

# Training job.
python run.py \
--dataset="imagenet2012" --train_mode="pretrain" \
--resnet_depth=${RESNET_DEPTH} --width_multiplier=${WIDTH_MULTIPLIER} \
--train_batch_size=${TRAIN_BATCH_SIZE} \
${ADDITIONAL_FLAGS}


# Evaluation job. This can run continuously during training.
python run.py \
--dataset="imagenet2012" --mode="eval" \
--resnet_depth=${RESNET_DEPTH} --width_multiplier=${WIDTH_MULTIPLIER} \
--eval_batch_size=${EVAL_BATCH_SIZE} \
${ADDITIONAL_FLAGS}
