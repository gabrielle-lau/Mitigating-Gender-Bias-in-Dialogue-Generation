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
    dpath = os.path.join(opt['datapath'], 'reddit_pushshift')

    # don't download data, use local data file instead
    build_data.mark_done(dpath, version)
    return

    # if not build_data.built(dpath, version):
    #     logging.info('building data: ' + dpath)
    #     if build_data.built(dpath):
    #         # An older version exists, so remove these outdated files.
    #         build_data.remove_dir(dpath)
    #     build_data.make_dir(dpath)

    #     # Download the data.
    #     for downloadable_file in RESOURCES:
    #         downloadable_file.download_file(dpath)

    #     # Mark the data as built.
    #     build_data.mark_done(dpath, version)
