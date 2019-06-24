#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import sys
import random
import pickle

sys.path.extend(['/home/welling/Fiasco/fiasco_final/bin/LINUXX86_64',
                 '/home/welling/shtools/SHTOOLS-3.2'])
# sys.path.extend(['/home/welling/Fiasco/Fiasco_final/src/fmri',
#                  '/home/welling/Fiasco/Fiasco_final/bin/LINUXX86_64',
#                  '/home/welling/git/SHTOOLS'])

from .traceneighbors import Vtx, UsefulVtx, loadSkipTable
# from transforms import eulerRzRyRzToTrans, transToEulerRzRyRz, makeAligningRotation
# from writegeom import writeBOV, plotSphere, writeVtkPolylines
# from sampler import ArraySampler
from .yamlblocks import BlockGenerator

radPixels = 20
baseName = 'empty'
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
emptyLocFile = '/home/welling/brainroller/emptySamps.pkl'


class FakeVtx(UsefulVtx):
    def __init__(self, vid, x, y, z):
        super(UsefulVtx, self).__init__(vid, 0, None, x, y, z)
        self.edges = []
    
    @property
    def z(self):
        return self.absCoords[2]
        
    @property
    def realCoords(self):
        return (self.x, self.y, self.z)

    def getLoc(self):
        """
        For the benefit of the Spilltree module, which expects points to have
        a getLoc method.
        """
        return [self.x, self.y, self.z]

def main():
    edgeLen = 2*radPixels + 1
    rMax = float(radPixels)
#    transformer = SHTransformer(edgeLen, maxL)

    with open(emptyLocFile, 'r') as f:
        emptyLocs = pickle.load(f)
    print('loaded %d empty locations' % len(emptyLocs))
    fakeVtxDict = {}
    for idx, (x, y, z) in enumerate(emptyLocs):
        fakeVtxDict[idx] = FakeVtx(idx, x, y, z)
#     with open(traceFile, 'r') as pklF:
#         with open(skipFile, 'r') as skipF:
#             usefulVtxDict = UsefulVtx.load(pklF, 10000, skipF)
# #             usefulVtxDict = UsefulVtx.load(pklF, 10000, None)
#     print 'Loaded %d useful vertices' % len(usefulVtxDict)

    blockGen = BlockGenerator(rMax, edgeLen, maxL, 
                              fakeVtxDict, fishCoreFile,
                              fishCoreXSize, fishCoreYSize, fishCoreZSize,
                              baseName=baseName)

    #sampleVtx = usefulVtxDict[6985]
    #sampleVtx = usefulVtxDict.values()[17]
    random.seed(1234)
    indexList = list(fakeVtxDict.keys())[:]
    indexList.sort()
    for sampleId in indexList:
        if sampleId > 3468:
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
