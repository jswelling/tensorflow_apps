#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import sys
import random
import os.path
import pickle
import math

import spilltree3D

sys.path.extend(['/home/welling/Fiasco/fiasco_final/bin/LINUXX86_64',
                 '/home/welling/shtools/SHTOOLS-3.2'])
# sys.path.extend(['/home/welling/Fiasco/Fiasco_final/src/fmri',
#                  '/home/welling/Fiasco/Fiasco_final/bin/LINUXX86_64',
#                  '/home/welling/git/SHTOOLS'])

from .traceneighbors import UsefulVtx, Vtx, loadSkipTable
# from transforms import eulerRzRyRzToTrans, transToEulerRzRyRz, makeAligningRotation
# from writegeom import writeBOV, plotSphere, writeVtkPolylines
# from sampler import ArraySampler

#from yamlblocks import BlockGenerator

radPixels = 20
cutoffRad = 10.0
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

#baseDir = '/pylon2/pscstaff/welling'
baseDir = '/home/welling/brainroller'
usefulVtxFile = os.path.join(baseDir, 'useful_trace_neighborhoods.pkl')
skipFile = os.path.join(baseDir, 'skips.txt')
traceFile = os.path.join(baseDir, 'traces.pkl')

class PlainPt(object):
    def __init__(self, x, y, z):
        self.coords = [x, y, z]
        
    def getLoc(self):
        return self.coords
    

class BoundedRandomSampler(object):
    def __init__(self, rMax):
        self.rMax = rMax
        xMin = rMax
        xMax = fishCoreXSize - rMax
        yMin = rMax
        yMax = fishCoreYSize - rMax
        zMin = rMax
        zMax = fishCoreYSize - rMax
        self.xOffset = xMin + fishCoreXOffset
        self.yOffset = yMin + fishCoreYOffset
        self.zOffset = zMin + fishCoreZOffset
        self.xScale = xMax - xMin
        self.yScale = yMax - yMin
        self.zScale = zMax - zMin

    def getPt(self):
        return PlainPt(random.random() * self.xScale + self.xOffset,
                       random.random() * self.yScale + self.yOffset,
                       random.random() * self.zScale + self.zOffset)
        
    def outerClip(self, pt):
        x, y, z = pt.getLoc()
        if z < fishCoreZOffset - self.rMax:
            return False
        elif x < fishCoreXOffset - self.rMax:
            return False
        elif y < fishCoreYOffset - self.rMax:
            return False
        elif z > fishCoreZOffset + fishCoreZSize + self.rMax:
            return False
        elif x > fishCoreXOffset + fishCoreXSize + self.rMax:
            return False
        elif y > fishCoreYOffset + fishCoreYSize + self.rMax:
            return False
        else:
            return True
        

def main():
    edgeLen = 2*radPixels + 1
    rMax = float(radPixels)
#    transformer = SHTransformer(edgeLen, maxL)

#     with open(usefulVtxFile, 'r') as pklF:
#         with open(skipFile, 'r') as skipF:
#             usefulVtxDict = UsefulVtx.load(pklF, 30000, skipF)

    with open(traceFile, 'r') as f:
        vtxDict, objDict = pickle.load(f)
    with open(skipFile, 'rU') as skipF:
        skipTbl = loadSkipTable(skipF, 30000)
    for v in list(vtxDict.values()):
        v.setSkipTable(skipTbl)
    print('%d vertices in %d objects' % (len(vtxDict), len(objDict)))
#     print 'Loaded %d useful vertices' % len(usefulVtxDict)

    ptSampler = BoundedRandomSampler(rMax)
    testPts = []
    for v in list(vtxDict.values()):
        if ptSampler.outerClip(v):
            x, y, z = v.getLoc()
            testPts.append(PlainPt(x, y, z))
    print('%d useful trace points' % len(testPts))
    spilltree = spilltree3D.SpTree(testPts)
    print('spilltree created')
    
    random.seed(1234)
    samplePts = []
    ct = 0
    tryCt = 0
    cutSqr = cutoffRad * cutoffRad
    while True:
        pt = ptSampler.getPt()
        _, sepsqr = spilltree.findApproxNearest(pt)
#         print 'samplept: %s' % pt.getLoc()
#         print 'nearPt: %s at %s' % (nearPt.id, nearPt.getLoc())
#         print 'sepsqr: %s' % sepsqr
        if sepsqr > cutSqr:
            samplePts.append(tuple(pt.getLoc()))
            ct += 1
        tryCt += 1
        if tryCt % 1000 == 1:
            print('%d samples in %d tries' % (ct, tryCt))
        if ct >= 5000:
            break
        
    with open('emptySamps.pkl', 'w') as f:
        pickle.dump(samplePts, f)

#     blockGen = BlockGenerator(rMax, edgeLen, maxL, 
#                               usefulVtxDict, fishCoreFile,
#                               fishCoreXSize, fishCoreYSize, fishCoreZSize,
#                               baseName=baseName)
# 
#     #sampleVtx = usefulVtxDict[6985]
#     #sampleVtx = usefulVtxDict.values()[17]
#     random.seed(1234)
#     indexList = usefulVtxDict.keys()[:]
#     indexList.sort()
#     for idx, sampleId in enumerate(random.sample(indexList, 5000)):
#         if (idx >= 4968):
#             try:
#                 print 'starting sample %s' % sampleId
#                 blockGen.writeBlock(sampleId, 
#                                     {'xOffset': fishCoreXOffset,
#                                      'yOffset': fishCoreYOffset,
#                                      'zOffset': fishCoreZOffset})
#             except Exception, e:
#                 print 'Sample id %s failed: %s' % (sampleId, e)
    print('completed main loop')

if __name__ == '__main__':
    main()
