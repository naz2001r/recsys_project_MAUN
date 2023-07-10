import sys
from train_base import TrainStepABC


if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py train_file test_file\n'
    )
    sys.exit(1)

trainer = TrainStepABC('HybridNN_Recommender')
trainer.train()
