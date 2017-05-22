#! /usr/bin/env python

# Simple tools for copying fish data around

import os
import re
import shutil

def copy_file(filename, dest_dir):
    if not os.path.isfile(filename):
        print('\twarning: %s does not exist' % (filename))
        return False
    shutil.copy2(filename, dest_dir)
    return True

def copy_data_files(fns, src_dir, dest_dir):
    copied = 0
    limit  = len(fns)

    for fn in fns:
        fName = os.path.join(src_dir, fn)

        print('[%d/%d] %s' % (copied, limit, fName))

        cp_err = copy_file(fName, dest_dir)

        if cp_err:
            copied += 1

def list_yaml_files(src_dir):
    fns = []
    yamlRE = re.compile(r'.+_.+_[0123456789]+\.yaml')

    for fn in os.listdir(src_dir):
        if yamlRE.match(fn):
            fns.append(fn)

    return fns

def copy_data_files_from_yamls(yaml_fns, src_dir, dest_dir):
    copied = 0
    limit  = len(yaml_fns)

    for fn in yaml_fns:
        words = fn[:-5].split('_')
        base = words[0]
        idx = int(words[2])
        yamlFName    = os.path.join(src_dir, fn)
        featureFName = os.path.join(src_dir, '%s_rotBallSamp_%d.doubles' % (base, idx))
        labelFName   = os.path.join(src_dir, '%s_rotEdgeSamp_%d.doubles' % (base, idx))

        print('[%d/%d] %s' % (copied, limit, yamlFName))
        print(' + %s' % featureFName)
        print(' + %s' % labelFName)

        cp_yaml_err = copy_file(yamlFName, dest_dir)
        cp_ball_err = copy_file(featureFName, dest_dir)
        cp_edge_err = copy_file(labelFName, dest_dir)

        if cp_edge_err and cp_ball_err and cp_edge_err:
            copied += 1
