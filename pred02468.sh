PYTHONPATH=. mpiexec -n 8 python examples/segmentation/predict.py \
--model \
  cross-validation/cv0/final_model.npz \
  cross-validation/cv2/final_model.npz \
  cross-validation/cv4/final_model.npz \
  cross-validation/cv6/final_model.npz \
  cross-validation/cv8/final_model.npz \
--config \
  examples/configs/seg_resnet.yaml \
  examples/configs/seg_resnet.yaml \
  examples/configs/seg_resnet.yaml \
  examples/configs/seg_resnet.yaml \
  examples/configs/seg_resnet.yaml \
--test --x-flip 2 -t 0.1 -o $1
