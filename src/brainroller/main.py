#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import sys
import numpy as np

from .transforms import eulerRzRyRzToTrans, transToEulerRzRyRz, makeAligningRotation
from .writegeom import plotSphere
from .sampler import ArraySampler, BlockSampler
from .yamlblocks import parseBlock

sys.path.extend(['/home/welling/git/Fiasco/src/fmri'])
# sys.path.extend(['/home/welling/Fiasco/fiasco_final/bin/LINUXX86_64',
#                  '/home/welling/shtools/SHTOOLS-3.2'])
sys.path.extend(['/home/welling/Fiasco/fiasco_final/bin/LINUXX86_64',
                 '/home/welling/shtools/SHTOOLS-3.2'])
# sys.path.extend(['/home/welling/Fiasco/Fiasco_final/src/fmri',
#                  '/home/welling/Fiasco/Fiasco_final/bin/LINUXX86_64',
#                  '/home/welling/git/SHTOOLS'])

import pyshtools as psh
import fiasco_numpy as fiasco
from .shtransform import SHTransformer
from .sampler import Sampler

radPixels = 20
maxL = 48


def main():
    edgeLen = 2*radPixels + 1
    rMax = float(radPixels)
    transformer = SHTransformer(edgeLen, maxL)

    # baseName = 'block'
    # vId = 5619494
    baseName = 'synth'
    vId = 0
    yamlDict = parseBlock(vId, edgeLen, baseName=baseName)

    sampler = BlockSampler(yamlDict)
    byteCube = sampler.sample(None)
    assert byteCube.shape == (edgeLen, edgeLen, edgeLen), 'wrong size cube'
    doubleCube = np.require(byteCube, dtype=np.double,
                            requirements=['C'])

    zMat = np.outer(np.cos(transformer.thetaglq),
                    np.ones_like(transformer.phiglq))
    xMat = np.outer(np.sin(transformer.thetaglq),
                    np.cos(transformer.phiglq))
    yMat = np.outer(np.sin(transformer.thetaglq),
                    np.sin(transformer.phiglq))
    edgeMat = np.zeros((len(transformer.thetaglq),
                        len(transformer.phiglq)), dtype=np.float64)
    for edge in yamlDict['edges']:
        cosMat = (xMat * edge[0]) + (yMat * edge[1]) + (zMat * edge[2])
        edgeMat += np.power(cosMat, 20.0)

    samples = transformer.calcSphereOfSamples(doubleCube, rMax)
    harmonics = transformer.calcHarmonics(doubleCube, rMax)
    unRotReconSamples = transformer.reconstructSamples(harmonics)
    rotMatrix = psh.djpi2(maxL)
    thetaZ0, thetaY, thetaZ1 = yamlDict['eulerZYZ']
    rotClim = psh.SHRotateRealCoef(harmonics,
                                   np.array([-thetaZ1, -thetaY, -thetaZ0]),
                                   rotMatrix)
    reconHarm = transformer.reconstructSamples(rotClim)

    edgeHarm = psh.SHExpandGLQ(edgeMat,
                               transformer.weights, transformer.nodes)
    rotEdgeHarm = psh.SHRotateRealCoef(edgeHarm,
                                       np.array([-thetaZ1, -thetaY, -thetaZ0]),
                                       rotMatrix)
    rotEdge = transformer.reconstructSamples(rotEdgeHarm)

    plotSphere(transformer.thetaglq, transformer.phiglq,
               {'samples': samples.transpose(),
                'unrotated': unRotReconSamples.transpose(),
                'rotated': reconHarm.transpose(),
                'unrotEdges': edgeMat.transpose(),
                'rotEdges': rotEdge.transpose()
                },
               fname="%s_%d_spheres.vtk" % (baseName, vId))


if __name__ == '__main__':
    main()
