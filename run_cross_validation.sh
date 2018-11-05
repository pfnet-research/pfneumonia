mkdir cross-validation
mpiexec -n 8 python examples/segmentation/train.py --val-set 0 --snapshot 0 -o cross-validation/cv0
mpiexec -n 8 python examples/segmentation/train.py --val-set 1 --snapshot 0 -o cross-validation/cv1
mpiexec -n 8 python examples/segmentation/train.py --val-set 2 --snapshot 0 -o cross-validation/cv2
mpiexec -n 8 python examples/segmentation/train.py --val-set 3 --snapshot 0 -o cross-validation/cv3
mpiexec -n 8 python examples/segmentation/train.py --val-set 4 --snapshot 0 -o cross-validation/cv4
mpiexec -n 8 python examples/segmentation/train.py --val-set 5 --snapshot 0 -o cross-validation/cv5
mpiexec -n 8 python examples/segmentation/train.py --val-set 6 --snapshot 0 -o cross-validation/cv6
mpiexec -n 8 python examples/segmentation/train.py --val-set 7 --snapshot 0 -o cross-validation/cv7
mpiexec -n 8 python examples/segmentation/train.py --val-set 8 --snapshot 0 -o cross-validation/cv8
mpiexec -n 8 python examples/segmentation/train.py --val-set 9 --snapshot 0 -o cross-validation/cv9
