import numpy as np

import tvm
from tvm import relay, auto_scheduler
import tvm.relay.testing
from tvm.contrib import graph_executor
import argparse
import os

import transformers
from transformers import *
import torch
import pickle
from transformers import logging
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import copy


def task_desc_to_group_name(
    task_desc
):
    task_layers = task_desc.split('_')
    if task_layers[-1].isdigit():
        task_layers.pop()
    return '_'.join(task_layers)

def args_parser():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--target', help='target hardware')
    parser.add_argument('--model', help='model name')
    parser.add_argument('--num_measures_per_round', type=int, default=64, help='number of measure trials')
    parser.add_argument('--num_trials', type=int, default=200, help='number of measure trials per task')
    parser.add_argument('--test_idx', type=int, default=0, help='test idx')
    parser.add_argument('--log_dir', default='log')
    args = parser.parse_args()
    return args

# Define the neural network and compilation target
args = args_parser()
if args.target != 'llvm':
    device = 'cuda'
else:
    device = 'llvm'

log_dir = args.log_dir
test_idx = args.test_idx
dir_name=f"{log_dir}/{args.model}/{test_idx}"

network = args.model
batch_size = 1
if network == 'squeezenet_v1.1' or network == 'mxnet':
    layout = "NCHW"
else:
    layout = "NHWC"
target = tvm.target.Target(device) # cpu: llvm
dtype = "float32"
# rm_ratio = args.rm
num_measures_per_round = args.num_measures_per_round
num_trials = args.num_trials
# num_target_tune = args.num_target_tune
# target_task_idx = args.target_task_idx

def get_network(name, batch_size, layout="NHWC", dtype="float32"):
    """Get the symbol definition and random weight of a network"""

    from mxnet.gluon.model_zoo.vision import get_model
    from gluoncv2.model_provider import get_model as glcv2_get_model

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = relay.testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "mobilenetv2":
        block = get_model("mobilenetv2_1.0", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
        
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = relay.testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
        
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = relay.testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
        
    elif name == "efficientnet":

        block = net = glcv2_get_model("EfficientNet_B0", pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": (batch_size, 3, 224, 224)}, dtype=dtype
        )
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
        
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet18_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
        
    elif "densenet" in name:
        mod, params = relay.testing.densenet.get_workload(batch_size=batch_size, dtype=dtype)
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)
        
    elif name == "bert":
        from transformers import BertTokenizer
        from transformers import BertConfig
        from transformers import BertModel

        enc = BertTokenizer.from_pretrained("bert-base-uncased")
        # Tokenizing input text
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = enc.tokenize(text)

        # Masking one of the input tokens
        masked_index = 8
        tokenized_text[masked_index] = "[MASK]"
        indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

        # Creating a dummy input
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        dummy_input = [tokens_tensor, segments_tensors]

        # Initializing the model with the torchscript flag
        # Flag set to True even though it is not necessary as this model does not have an LM Head.
        config = BertConfig(
            vocab_size_or_config_json_file=32000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            torchscript=True,
        )
        # Instantiating the model
        model = BertModel(config)
        # The model needs to be in evaluation mode
        model.eval()
        # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
        model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)
        # Creating the trace
        traced_model = torch.jit.trace(model, [tokens_tensor, segments_tensors])

        shape_list = [
            (i.debugName().split(".")[0], i.type().sizes())
            for i in list(traced_model.graph.inputs())[1:]
        ]

        mod, params = tvm.relay.frontend.pytorch.from_pytorch(
            traced_model, shape_list, default_dtype="float32"
        )
        input_shape = tokens_tensor.numpy()
        output_shape = segments_tensors.numpy()
        
    elif "vgg" in name:
        n_layer = int(name.split("-")[1])
        mod, params = relay.testing.vgg.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            dtype=dtype,
        )
        if layout == "NHWC":
            desired_layouts = {"nn.conv2d": ["NHWC", "default"]}
            seq = tvm.transform.Sequential(
                [
                    relay.transform.RemoveUnusedFunctions(),
                    relay.transform.ConvertLayout(desired_layouts),
                ]
            )
            with tvm.transform.PassContext(opt_level=3):
                mod = seq(mod)

    return mod, params, input_shape, output_shape

# Extract tasks from the network
print("Extract tasks...")
mod, params, input_shape, output_shape = get_network(network, batch_size, layout, dtype=dtype)
tasks, task_weights = auto_scheduler.extract_tasks(mod["main"], params, target)

if not os.path.exists(f"./{dir_name}/"):
    os.makedirs(f"./{dir_name}")

log_file = f"./{dir_name}/ansor-{network}-{layout}-B{batch_size}-{target.kind.name}-{num_measures_per_round}.json"
log_file_name = f"./{dir_name}/ansor-{network}-{layout}-B{batch_size}-{target.kind.name}-{num_measures_per_round}.tsv"

for idx, task in enumerate(tasks):
    print("========== Task %d  (workload key: %s) ==========" % (idx, task.workload_key))
    task.id = idx
    
def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, task_weights, log_file_name=log_file_name)
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials = len(tasks)*num_trials, # * 800, # 2000, #800*6, # len(tasks) * 800 #200,  # change this to 20000 to achieve the best performance
        early_stopping = None,
        num_measures_per_round = num_measures_per_round,
        verbose=1,
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    tuner.tune(tune_option)
    
print("Begin tuning... all tasks")
start_time = time.time()
run_tuning()
end_time = time.time()
execution_time = end_time - start_time
print(f"all tasks Execution time: {execution_time} seconds")
