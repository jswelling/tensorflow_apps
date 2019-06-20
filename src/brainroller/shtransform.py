#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import math
import numpy as np

import pyshtools as psh
import fiasco_numpy as fiasco

def latToTheta(lat):
    """
    Takes latitude values as provided by GLQGridCoord and converts to correctly oriented radians.
    """
    return (90.0 - lat) * (np.pi/180.0)


def lonToPhi(lon):
    """
    Takes longitude values as provided by GLQGridCoord and converts to correctly oriented radians.
    """
    return (np.pi/180.0) * lon


class SHTransformer(object):
    def __init__(self, edgeLen, maxL):
        self.edgeLen = edgeLen
        self.maxL = maxL
        self.rMax = 0.5*float(edgeLen + 1)
        self.intrp = fiasco.createInterpolator3DByType(fiasco.INTRP_LINEAR,
                                                       edgeLen, edgeLen, edgeLen, 1)
        self.lDict = {}
        self.lChain = None
        
    def prepL(self, maxL):
        if maxL not in self.lDict:
            latglq, longlq = psh.GLQGridCoord(maxL)
            thetaglq = latToTheta(latglq)
            phiglq = lonToPhi(longlq)
            nodes, weights = psh.SHGLQ(maxL)
            self.lDict[maxL] = (thetaglq, phiglq, nodes, weights)
        return self.lDict[maxL]
    
    def expand(self, samps):
        _, _, nodes, weights = self.prepL(samps.shape[0] - 1)
        return psh.SHExpandGLQ(samps, weights, nodes)
                               
    def calcSphereOfSamples(self, dataCube, radius, sig=(1.0, 1.0, 1.0),
                            maxL=None):
        if maxL is None:
            maxL = self.maxL
        thetaglq, phiglq, nodes, weights = self.prepL(maxL)  # @UnusedVariable
        assert dataCube.shape == (self.edgeLen, self.edgeLen, self.edgeLen), \
            "Data cube is not %d by %d by %d" % (self.edgeLen, self.edgeLen, self.edgeLen)
        assert radius <= self.rMax, ("Radius %s is larger than %s"
                                     % radius, self.rMax)
        self.intrp.prep(dataCube)
        tout = np.zeros(1)
        samps = np.empty((len(thetaglq), len(phiglq)), dtype=np.float64)
        xCtr = yCtr = zCtr = 0.5 * float(self.edgeLen - 1)
        sx, sy, sz = sig
        for i, theta in enumerate(thetaglq):
            for j, phi in enumerate(phiglq):
                x = sx * (radius * np.sin(theta) * np.cos(phi)) + xCtr
                y = sy * (radius * np.sin(theta) * np.sin(phi)) + yCtr
                z = sz * (radius * np.cos(theta)) + zCtr
                self.intrp.calc(tout, (z, y, x), 1, 0)
                samps[i, j] = tout[0]
        return samps

    def calcHarmonics(self, dataCube, radius, sig=(1.0, 1.0, 1.0),
                      maxL=None):
        if maxL is None:
            maxL = self.maxL
        thetaglq, phiglq, nodes, weights = self.prepL(maxL)  # @UnusedVariable
        samps = self.calcSphereOfSamples(dataCube, radius, sig=sig,
                                         maxL=maxL)
        return psh.SHExpandGLQ(samps, weights, nodes)

    def rotateHarmonics(self, harmonics, thetaZ0, thetaY, thetaZ1, maxL=None):
        if maxL is None:
            maxL = self.maxL
        rotMatrix = psh.djpi2(maxL)
        rotHarm = psh.SHRotateRealCoef(harmonics,
                                       np.array([-thetaZ1, -thetaY, -thetaZ0]),
                                       rotMatrix)
        return rotHarm

    def reconstructSamples(self, harmonics, maxL=None):
        if maxL is None:
            maxL = self.maxL
        thetaglq, phiglq, nodes, weights = self.prepL(maxL)  # @UnusedVariable
        assert harmonics.shape[1] == nodes.shape[0], 'maxL does not match'
        return psh.MakeGridGLQ(harmonics, nodes)
    
    def initLChain(self):
        """
        If maxL corresponds to r=self.rMax, what L corresponds to
        each of the other relevant r's?
        """
        if self.lChain is None:
            self.lChain = {}  # maps edge -> (r, L) where 0 <= edge <= edgeLen+1
            self.lChain[0] = (0, 0)
            self.lChain[(self.edgeLen+1)] = (self.rMax, self.maxL)
            for edge in range(1, self.edgeLen+1):
                r = 0.5 * edge
                l = int(math.ceil((edge * self.maxL)/ float(self.edgeLen + 1)))
                self.lChain[edge] = (r, l)
#             dumpList = [(e, r, l) for e, (r, l) in self.lChain.items()]
#             dumpList.sort()
#             for e, r, l in dumpList:
#                 print '%d: (%s, %s)' % (e, r, l)
    
    def _getSortedFullChain(self):
        if self.lChain is None:
            self.initLChain()
        fullChain = [(e, r, l) for e, (r, l) in list(self.lChain.items())]
        fullChain.sort()
        return fullChain

    def calcBallOfSamples(self, dataCube, sig=(1.0, 1.0, 1.0)):
        fullChain = self._getSortedFullChain()
        totBallSz = 0
        for _, _, l in fullChain:
            totBallSz += (l + 1) * ((2 * l) + 1)
        result = np.zeros(totBallSz)
        offset = 0
        for _, r, l in fullChain:
            samps = self.calcSphereOfSamples(dataCube, r, sig=sig, maxL=l)
            lenFlat = samps.shape[0] * samps.shape[1]
            result[offset: offset+lenFlat] = samps.flat
            offset += lenFlat
        assert offset == totBallSz, 'Sample count does not match'
        return result

    def getBallStructure(self):
        fullChain = self._getSortedFullChain()
        infoChain = []
        for _, r, l in fullChain:
            thetaglq, phiglq, nodes, weights = self.prepL(l)
            infoChain.append((r, thetaglq, phiglq, nodes, weights))
        return infoChain
            
    def expandBall(self, ballSamples):
        fullChain = self._getSortedFullChain()
        offset = 0
        expChain = []
        totExpSz = 0
        for _, _, l in fullChain:
            dim1 = l + 1
            dim2 = (2 * l) + 1
            blkSz = dim1 * dim2
            thisBlk = np.zeros(blkSz)
            thisBlk[:] = ballSamples[offset: offset+blkSz]
            thisExp = self.expand(thisBlk.reshape((dim1, dim2)))
            expChain.append(thisExp)
            s0, s1, s2 = thisExp.shape
            totExpSz += s0 * s1 * s2
            offset += blkSz
        result = np.zeros(totExpSz)
        offset = 0
        for thisExp in expChain:
            s0, s1, s2 = thisExp.shape
            lenFlat = s0 * s1 *s2
            result[offset: offset + lenFlat] = thisExp.flat
            offset += lenFlat
        assert offset == totExpSz, 'Expansion term count does not match'
        return result
            
    def reconstructBall(self, ballExpansion):
        fullChain = self._getSortedFullChain()
        offset = 0
        sampChain = []
        totSampSz = 0
        for _, _, l in fullChain:
            dim0 = 2
            dim1 = l + 1
            dim2 = l + 1
            blkSz = dim0 * dim1 * dim2
            thisBlk = np.zeros(blkSz)
            thisBlk[:] = ballExpansion[offset: offset+blkSz]
            thisSamp = self.reconstructSamples(thisBlk.reshape((dim0, dim1, dim2)),
                                               maxL = l)
            sampChain.append(thisSamp)
            s1, s2 = thisSamp.shape
            totSampSz += s1 * s2
            offset += blkSz
        result = np.zeros(totSampSz)
        offset = 0
        for thisSamp in sampChain:
            s1, s2 = thisSamp.shape
            lenFlat = s1 * s2
            result[offset: offset + lenFlat] = thisSamp.flat
            offset += lenFlat
        assert offset == totSampSz, 'Sample count does not match'
        return result
    
    def rotateBall(self, ballExpansion, thetaZ0, thetaY, thetaZ1):
        fullChain = self._getSortedFullChain()
        result = np.zeros_like(ballExpansion)
        offset = 0
        for _, _, l in fullChain:
            dim0 = 2
            dim1 = l + 1
            dim2 = l + 1
            blkSz = dim0 * dim1 * dim2
            thisBlk = np.zeros(blkSz)
            thisBlk[:] = ballExpansion[offset: offset+blkSz]
            thisRotBlk = self.rotateHarmonics(thisBlk.reshape((dim0, dim1, dim2)),
                                              thetaZ0, thetaY, thetaZ1, maxL = l)
            result[offset: offset+blkSz] = thisRotBlk.flat
            offset += blkSz
        return result
            
            