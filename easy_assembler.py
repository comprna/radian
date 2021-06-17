# Copyright 2017 The Chiron Authors. All Rights Reserved.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Created on Tue May  2 15:39:29 2017
#
# Source: https://github.com/scutbioinformatic/causalcall/blob/master/easy_assembler.py
# Commit: f5ab3db0f6e373679df8baf2ed42031b43e4f368

from __future__ import absolute_import
from __future__ import print_function
import difflib
import numpy as np


#########################Simple assembly method################################
def simple_assembly(bpreads):
    # print(bpreads)
    concensus = np.zeros([4, 1000])
    pos = 0
    length = 0
    census_len = 1000
    for indx, bpread in enumerate(bpreads):
        if indx == 0:
            add_count(concensus, 0, bpread)
            continue
        d = difflib.SequenceMatcher(None, bpreads[indx - 1], bpread)
        match_block = max(d.get_matching_blocks(), key=lambda x: x[2])
        disp = match_block[0] - match_block[1]
        if disp + pos + len(bpreads[indx]) > census_len:
            concensus = np.lib.pad(concensus, ((0, 0), (0, 1000)),
                                   mode='constant', constant_values=0)
            census_len += 1000
        add_count(concensus, pos + disp, bpreads[indx])
        pos += disp
        length = max(length, pos + len(bpreads[indx]))
    return concensus[:, :length]


def add_count(concensus, start_indx, segment):
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        concensus[base_dict[base]][start_indx + i] += 1


#########################Simple assembly method with quality score################################
def simple_assembly_qs(bpreads, qs_list):
    concensus = np.zeros([4, 1000])
    concensus_qs = np.zeros([4, 1000])
    pos = 0
    length = 0
    census_len = 1000
    assert len(bpreads) == len(qs_list)
    for indx, bpread in enumerate(bpreads):
        if indx == 0:
            add_count_qs(concensus, concensus_qs, 0, bpread, qs_list[indx])
            continue
        d = difflib.SequenceMatcher(None, bpreads[indx - 1], bpread)
        match_block = max(d.get_matching_blocks(), key=lambda x: x[2])
        disp = match_block[0] - match_block[1]
        if disp + pos + len(bpread) > census_len:
            concensus = np.lib.pad(concensus, ((0, 0), (0, 1000)),
                                   mode='constant', constant_values=0)
            concensus_qs = np.lib.pad(concensus_qs, ((0, 0), (0, 1000)),
                                      mode='constant', constant_values=0)
            census_len += 1000
        add_count_qs(concensus, concensus_qs, pos + disp, bpread, qs_list[indx])
        pos += disp
        length = max(length, pos + len(bpread))
    return concensus[:, :length], concensus_qs[:, :length]


def add_count_qs(concensus, concensus_qs, start_indx, segment, qs):
    base_dict = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'a': 0, 'c': 1, 'g': 2, 't': 3}
    if start_indx < 0:
        segment = segment[-start_indx:]
        start_indx = 0
    for i, base in enumerate(segment):
        concensus[base_dict[base]][start_indx + i] += 1
        concensus_qs[base_dict[base]][start_indx + i] += qs[0]

# Source: https://github.com/scutbioinformatic/causalcall/blob/f5ab3db0f6e373679df8baf2ed42031b43e4f368/basecall.py#L39

def index2base(read):
    """Transfer the number into dna base.
    The transfer will go through each element of the input int vector.
    """
    base = ['A', 'C', 'G', 'T']
    bpread = [base[x] for x in read]
    bpread = ''.join(x for x in bpread)
    return bpread