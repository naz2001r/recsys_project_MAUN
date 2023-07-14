import sys
sys.path.append('./src/models/')
from train_base import TrainStepABC


if len(sys.argv) != 2:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 train.py split_directory\n'
    )
    sys.exit(1)

trainer = TrainStepABC('baseline')
trainer.train()
