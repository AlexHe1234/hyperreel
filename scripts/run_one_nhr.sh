CUDA_VISIBLE_DEVICES=5 python main.py experiment/dataset=nhr \
    experiment/training=nhr_tensorf \
    experiment.training.val_every=1 \
    experiment.training.test_every=100 \
    experiment.training.ckpt_every=1 \
    experiment.training.num_epochs=30 \
    experiment/model=nhr \
    experiment.params.print_loss=False \
    experiment.dataset.collection=sport_1_easymocap \
    experiment.dataset.num_frames=2 \
    experiment.dataset.keyframe_step=1 \
    +experiment/regularizers/tensorf=tv_4000 \
    experiment.params.name=nhr_sport_1
