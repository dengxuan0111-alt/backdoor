python scripts/train_poisoned_visualglm_classifier.py \
  --config configs/visualglm_defense.yaml \
  --train-dir /home/dengxuan/DATA/imagenet1k/train \
  --val-dir /home/dengxuan/DATA/imagenet1k/val \
  --device cuda:4 \
  --epochs 1 \
  --batch-size 16 \
  --num-workers 8 \
  --lr-head 1e-4 \
  --num-classes 1000 \
  --image-size 224 \
  --normalize-for-model \
  --attack-type badnet \
  --poison-ratio 0.1 \
  --target-id 239 \
  --patch-ratio 0.06 \
  --patch-position bottom_right \
  --init-classifier-head-path checkpoints/classifier_head.pt \
  --save-dir checkpoints \
  --save-prefix imagenet_badnet_t239

  #目前都是训练一个epoch
