#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import numpy as np


class Sampler(object):
    def __init__(self, edgeLen):
        self.edgeLen = edgeLen
        pass

    def sample(self, targetCoords):
        raise RuntimeError('Base class sample method called')


class RandomArraySampler(Sampler):
    """
    Used like:

      sampler = RandomArraySampler(edgeLen, 1234)
      dCube = sampler.sample(None)
    """
    def __init__(self, edgeLen, seed=None):
        super(RandomArraySampler, self).__init__(edgeLen)
        self.rng = np.random.RandomState(seed)

    def sample(self, targetCoords):
        """
        Targetcoords is ignored.
        """
        print(self.edgeLen)
        return self.rng.rand(self.edgeLen, self.edgeLen, self.edgeLen)


class CylinderSampler(Sampler):
    """
    Used like:

      sampler = CylinderSampler(edgeLen)
      dCube = sampler.sample({'radius': 18.0, 'thickness': 4.0,
                              'axisX': 1.0, 'axisY': 1.0, 'axisZ': 1.0})
    """
    def __init__(self, edgeLen, sig=(1.0, 1.0, 1.0)):
        super(CylinderSampler, self).__init__(edgeLen)
        self.sig = sig

    def sample(self, targetCoords):
        """
        targetCoords is a dict, with float keys 'radius', 'thickness', 'axisX',
        'axisY', and 'axisZ'
        """
        rslt = np.zeros((self.edgeLen, self.edgeLen, self.edgeLen),
                        dtype=np.uint8)
        xCtr = yCtr = zCtr = 0.5 * float(self.edgeLen - 1)
        rad = targetCoords['radius']
        thk = targetCoords['thickness']
        axis = np.matrix([[targetCoords['axisX'], targetCoords['axisY'],
                           targetCoords['axisZ']]]).T
        axis /= np.linalg.norm(axis)
        sx, sy, sz = self.sig
        for i in range(self.edgeLen):
            x = sx * (float(i) - xCtr)
            for j in range(self.edgeLen):
                y = sy * (float(j) - yCtr)
                for k in range(self.edgeLen):
                    z = sz * (float(k) - zCtr)
                    xVec = np.matrix([[x, y, z]]).transpose()
                    rVec = xVec - (axis.T.dot(xVec)[0, 0] * axis)
                    r2 = rVec.T.dot(rVec)
                    if r2 <= rad*rad and r2 >= (rad-thk) * (rad-thk):
                        rslt[i, j, k] = 255
                    else:
                        rslt[i, j, k] = 0

        return rslt


class CheckerboardSampler(Sampler):
    """
    Used like:

      sampler = Checkerboard(edgeLen)
      dCube = sampler.sample({'spacing': 5})
    """
    def __init__(self, edgeLen, sig=(1.0, 1.0, 1.0)):
        super(CylinderSampler, self).__init__(edgeLen)
        self.sig = sig

    def sample(self, targetCoords):
        """
        targetCoords is a dict, with float keys 'radius', 'thickness', 'axisX',
        'axisY', and 'axisZ'
        """
        rslt = np.zeros((self.edgeLen, self.edgeLen, self.edgeLen),
                        dtype=np.uint8)
        modulus = targetCoords['spacing']
        for i in range(self.edgeLen):
            for j in range(self.edgeLen):
                for k in range(self.edgeLen):
                    if (i % modulus == 0) or (j % modulus == 0):
                        rslt[i, j, k] = 255
                    else:
                        rslt[i, j, k] = 0

        return rslt


class ArraySampler(Sampler):
    def __init__(self, edgeLen, fname, blockX, blockY, blockZ):
        super(ArraySampler, self).__init__(edgeLen)
        rawData = np.fromfile(fname, dtype=np.uint8, 
                              count=(blockX * blockY * blockZ))
        print('loaded %s' % fname)
        self.data = np.reshape(rawData, (blockX, blockY, blockZ),
                               order='F')

    def sample(self, targetCoords):
        """
        targetCoords is a dict, with float keys 'xLoc', 'yLoc', 'zLoc',
        'xOffset', 'yOffset', 'zOffset'.
        """
        xMin = int(round((targetCoords['xLoc'] - self.edgeLen/2)
                         - targetCoords['xOffset']))
        xMax = xMin + self.edgeLen
        yMin = int(round((targetCoords['yLoc'] - self.edgeLen/2)
                         - targetCoords['yOffset']))
        yMax = yMin + self.edgeLen
        zMin = int(round((targetCoords['zLoc'] - self.edgeLen/2)
                         - targetCoords['zOffset']))
        zMax = zMin + self.edgeLen
        rslt = self.data[xMin:xMax, yMin:yMax, zMin:zMax]
#         print 'sampling x: %s %s mean = %s' % (xMin, xMax, (xMin + xMax) / 2)
#         print 'sampling y: %s %s mean = %s' % (yMin, yMax, (yMin + yMax) / 2)
#         print 'sampling z: %s %s mean = %s' % (zMin, zMax, (zMin + zMax) / 2)
        return rslt


class BlockSampler(Sampler):
    def __init__(self, infoDict):
        self.bovFName = infoDict['bovFiles'][0]
        self.fname = infoDict['bovFiles'][1]
        with open(self.bovFName, 'rU') as f:
            for line in f:
                words = line.split()
                if words[0] == 'DATA_FORMAT:':
                    tStr = words[1]
                    self.dtype = {'DOUBLE': np.float64,
                                  'FLOAT': np.float32,
                                  'BYTE': np.uint8}[tStr]
                elif words[0] == 'DATA_SIZE:':
                    xSz = int(words[1])
                    ySz = int(words[2])
                    zSz = int(words[3])
                    assert xSz == ySz and ySz == zSz, 'Not a cube'
                    edgeLen = xSz
        super(BlockSampler, self).__init__(edgeLen)
        
    def sample(self, targetCoords):
        """coords are ignored"""
        rawData = np.fromfile(self.fname, dtype=self.dtype, 
                              count=(self.edgeLen * self.edgeLen
                                     * self.edgeLen))
        byteCube = np.reshape(rawData,
                              (self.edgeLen, self.edgeLen, self.edgeLen),
                              order='F')
        return byteCube


