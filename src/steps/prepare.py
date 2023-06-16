import sys
import os
import yaml
import pandas as pd

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write(
        '\tpython3 prepare.py input_file output-file\n'
    )
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))['prepare']

data_path = os.path.join('data', 'prepared')
os.makedirs(data_path, exist_ok=True)

input_file = sys.argv[1]
print(input_file)
df = pd.read_csv(input_file, encoding_errors='ignore', sep=';')

# Add preprocessing

output_file = sys.argv[2]
df.to_csv(output_file, index=False)
