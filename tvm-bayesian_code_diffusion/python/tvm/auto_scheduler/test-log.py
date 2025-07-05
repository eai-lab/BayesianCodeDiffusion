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

# from .search_policy import SearchPolicy, SketchPolicy, PreloadMeasuredStates
# from .cost_model import RandomModel, XGBModel
# from .utils import array_mean
# from .measure import ProgramMeasurer
# from .measure_record import RecordReader
# from . import _ffi_api

log_path = "/workspace/TaskSimilarity/240409-init-population-target/experiments/__extracted_log/0/ansor-resnet-50-NHWC-B1-cuda-32-log.pkl"

with open(log_path, 'rb') as f:
    log = pickle.load(f)
    
import code; code.interact(local=locals())