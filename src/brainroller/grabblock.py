#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import sys
import random

sys.path.extend(['/home/welling/Fiasco/fiasco_final/bin/LINUXX86_64',
                 '/home/welling/shtools/SHTOOLS-3.2'])
# sys.path.extend(['/home/welling/Fiasco/Fiasco_final/src/fmri',
#                  '/home/welling/Fiasco/Fiasco_final/bin/LINUXX86_64',
#                  '/home/welling/git/SHTOOLS'])

from .traceneighbors import UsefulVtx
# from transforms import eulerRzRyRzToTrans, transToEulerRzRyRz, makeAligningRotation
# from writegeom import writeBOV, plotSphere, writeVtkPolylines
# from sampler import ArraySampler
from .yamlblocks import BlockGenerator

radPixels = 20
baseName = 'block'
# radPixels = 100
# baseName = 'bigblock'
maxL = 48
fishCoreFile = '/pylon1/pscstaff/awetzel/ZF-test-files/60nm-cores/V4750_2150_04000-08999.vol'
fishCoreXSize = 1024
fishCoreYSize = 1024
fishCoreZSize = 4900
fishCoreXOffset = 4750. - (fishCoreXSize/2)
fishCoreYOffset = 2150. - (fishCoreYSize/2)
fishCoreZOffset = 4000
traceFile = '/pylon2/pscstaff/welling/useful_trace_neighborhoods.pkl'
skipFile = '/pylon2/pscstaff/welling/skips.txt'


def main():
    edgeLen = 2*radPixels + 1
    rMax = float(radPixels)
#    transformer = SHTransformer(edgeLen, maxL)

    with open(traceFile, 'r') as pklF:
        with open(skipFile, 'r') as skipF:
            usefulVtxDict = UsefulVtx.load(pklF, 10000, skipF)
#             usefulVtxDict = UsefulVtx.load(pklF, 10000, None)
    print('Loaded %d useful vertices' % len(usefulVtxDict))

    blockGen = BlockGenerator(rMax, edgeLen, maxL, 
                              usefulVtxDict, fishCoreFile,
                              fishCoreXSize, fishCoreYSize, fishCoreZSize,
                              baseName=baseName)

    #sampleVtx = usefulVtxDict[6985]
    #sampleVtx = usefulVtxDict.values()[17]
    random.seed(1234)
    indexList = list(usefulVtxDict.keys())[:]
    indexList.sort()
    for idx, sampleId in enumerate(random.sample(indexList, 5000)):
        if (idx >= 4968):
            try:
                print('starting sample %s' % sampleId)
                blockGen.writeBlock(sampleId, 
                                    {'xOffset': fishCoreXOffset,
                                     'yOffset': fishCoreYOffset,
                                     'zOffset': fishCoreZOffset})
            except Exception as e:
                print('Sample id %s failed: %s' % (sampleId, e))
    print('completed main loop')

if __name__ == '__main__':
    main()
