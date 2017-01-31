#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import sys
import os
import numpy as np
import yaml

sys.path.extend(['/home/welling/Fiasco/fiasco_final/bin/LINUXX86_64',
                 '/home/welling/shtools/SHTOOLS-3.2'])

from yamlblocks import loadFieldEntry
from writegeom import plotSphere
from shtransform import SHTransformer

srcDirs = ['/pylon2/pscstaff/welling/fish_cubes_new_group1',
           '/pylon2/pscstaff/welling/fish_cubes_new_group2'
           ]



radPixels = 20
baseName = 'block'
maxL = 48


def main():
    edgeLen = 2 * radPixels + 1
    sumArr = None
    rotSumArr = None
    count = 0
    baseDir = os.getcwd()
    for dirP in srcDirs:
        os.chdir(dirP)
        for fname in os.listdir('.'):
            if fname.endswith('.yaml'):
                with open(fname, 'rU') as f:
                    yamlDict = yaml.load(f)
                edgeArr = loadFieldEntry(yamlDict, baseName, 'edgeSamp')
                rotEdgeArr = loadFieldEntry(yamlDict, baseName, 'rotEdgeSamp')
                if rotSumArr is not None:
                    rotSumArr += rotEdgeArr
                    sumArr += edgeArr
                else:
                    rotSumArr = rotEdgeArr
                    sumArr = edgeArr
                count += 1
                print fname
    rotSumArr /= count
    sumArr /= count
    print 'loaded %d samples' % count
    np.save('mean_rotEdge.npy', rotSumArr)
    os.chdir(baseDir)
    transformer = SHTransformer(edgeLen, maxL)
    transformer.initLChain()
    thetaglq, phiglq, _, _ = transformer.prepL(maxL)
    plotSphere(thetaglq, phiglq, {'rotEdges': rotSumArr, 'edges': sumArr.transpose()})

if __name__ == '__main__':
    main()
