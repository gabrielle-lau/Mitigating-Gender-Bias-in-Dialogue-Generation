#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Download and build the data if it does not exist.


import os
from parlai.core.build_data import DownloadableFile
import parlai.core.build_data as build_data
import parlai.utils.logging as logging



def build(opt):
    version = 'v1.0'
    dpath = os.path.join(opt['datapath'], 'genderation_data', 'empathetic_dialogues')

    # don't download data, use local data file instead
    build_data.mark_done(dpath, version)
    return

