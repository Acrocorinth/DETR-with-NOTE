# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build

#返回build函数
def build_model(args):
    return build(args)
