#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

CUDA_VISIBLE_DEVICES=6 python main.py experiment/dataset=neural_3d \
    experiment/training=neural_3d_tensorf \
    experiment.training.val_every=1 \
    experiment.training.test_every=100 \
    experiment.training.ckpt_every=10 \
    experiment.training.render_every=30 \
    experiment.training.num_epochs=30 \
    experiment/model=neural_3d_z_plane \
    experiment.params.print_loss=True \
    experiment.dataset.collection=cut_roasted_beef/videos \
    experiment.dataset.num_frames=200 \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.dataset.start_frame=0 \
    experiment.params.name=neural_3d_0_start_cut_roasted_beef

