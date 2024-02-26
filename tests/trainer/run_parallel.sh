export OMP_NUM_THREADS=8
torchrun --nproc_per_node=1 test_trainer.py
torchrun --nproc_per_node=2 test_trainer.py
torchrun --nproc_per_node=4 test_trainer.py
torchrun --nproc_per_node=8 test_trainer.py
