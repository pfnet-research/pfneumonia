PYTHONPATH=. mpiexec -n 8 python examples/segmentation/predict.py \
--model \
  cross-validation/cv1/final_model.npz \
  cross-validation/cv3/final_model.npz \
  cross-validation/cv5/final_model.npz \
  cross-validation/cv7/final_model.npz \
  cross-validation/cv9/final_model.npz \
--config \
  examples/configs/seg_resnet.yaml \
  examples/configs/seg_resnet.yaml \
  examples/configs/seg_resnet.yaml \
  examples/configs/seg_resnet.yaml \
  examples/configs/seg_resnet.yaml \
--test --x-flip 2 -t 0.1 -o $1
