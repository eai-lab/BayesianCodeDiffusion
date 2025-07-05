# import packages
import pandas as pd
import numpy as np

# configuration
test_idx = 0
target = 'cuda' # 'llvm' or 'cuda'
model = 'squeezenet_v1.1'
batch_size = 1
if model == 'squeezenet_v1.1' or model == 'mxnet':
    layout = "NCHW"
else:
    layout = "NHWC"

# set file paths
a_path = f'log_ansor/{model}/{test_idx}/ansor-{model}-{layout}-B{batch_size}-{target}-64.tsv'
b_sketch = f'log_codediffusion-sketch/{model}/{test_idx}/our-{model}-{layout}-B{batch_size}-{target}-64-sketch.tsv'
b_op = f'log_codediffusion-operator/{model}/{test_idx}/our-{model}-{layout}-B{batch_size}-{target}-64-operator.tsv'

# read the log files
a = pd.read_csv(a_path, sep='\t', header=None)
b_sketch = pd.read_csv(b_sketch, sep='\t', header=None)
b_op = pd.read_csv(b_op, sep='\t', header=None)

# remove na values
a[3].fillna(np.inf, inplace=True)
a_min_idx = a[3].idxmin()
a_min_elapsed_time = a.iloc[a_min_idx][1]
a_min_latency = a.iloc[a_min_idx][3]

b_sketch[3].fillna(np.inf, inplace=True)
b_op[3].fillna(np.inf, inplace=True)

# calc values for comparison
b_sketch_elapsed_time = b_sketch[b_sketch[3] <= a_min_latency][1].iloc[0]
b_op_elapsed_time = b_op[b_op[3] <= a_min_latency][1].iloc[0]
b_elapsed_time = min(b_sketch_elapsed_time, b_op_elapsed_time)

a_first_iter_latency = a[a[3] != np.inf][3].iloc[0]
a_last_iter_latency = a[a[3] != np.inf][3].iloc[-1]
b_sketch_first_iter_latency = b_sketch[b_sketch[3] != np.inf][3].iloc[0]
b_sketch_last_iter_latency = b_sketch[b_sketch[3] != np.inf][3].iloc[-1]
b_op_first_iter_latency = b_op[b_op[3] != np.inf][3].iloc[0]
b_op_last_iter_latency = b_op[b_op[3] != np.inf][3].iloc[-1]
b_first_iter_latency = min(b_sketch_first_iter_latency, b_op_first_iter_latency)
b_last_iter_latency = min(b_sketch_last_iter_latency, b_op_last_iter_latency)

# print the results
print(f'tested on {target} with {model}')
print(f'compilation time {(a_min_elapsed_time / b_elapsed_time):.2f} times faster')
print(f'first iter latency: {(a_first_iter_latency / b_first_iter_latency):.2f} times faster')
print(f'last iter latency: {(a_last_iter_latency / b_last_iter_latency):.2f} times faster')