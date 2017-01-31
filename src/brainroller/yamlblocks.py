#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import sys
import os
import math
import numpy as np
import yaml

from traceneighbors import Vtx, UsefulVtx
from transforms import transToEulerRzRyRz, makeAligningRotation
from writegeom import writeBOV, plotSphere, writeVtkPolylines, plotBall
from writegeom import dtypeToExt, extToDtype
from sampler import CylinderSampler, ArraySampler
from shtransform import SHTransformer


def calcMomentTensor(dCube, sig=(1.0, 1.0, 1.0)):
    edgeLen = dCube.shape[0]
    assert dCube.shape[1] == edgeLen and dCube.shape[2] == edgeLen, 'dCube is not a cube!'
    iT = np.zeros((3, 3))  # moment of inertia tensor
    xCtr = yCtr = zCtr = 0.5 * float(edgeLen - 1)
    rMax = 2.0 * xCtr
    for i in xrange(edgeLen):
        x = sig[0]*(float(i) - xCtr)
        for j in xrange(edgeLen):
            y = sig[1]*(float(j) - yCtr)
            for k in xrange(edgeLen):
                z = sig[2]*(float(k) - zCtr)
                rSqr = x*x + y*y + z*z
                if rSqr <= rMax * rMax:
                    v = dCube[i, j, k]
                    iT[0, 0] += (y*y + z*z) * v
                    iT[1, 1] += (x*x + z*z) * v
                    iT[2, 2] += (x*x + y*y) * v
                    iT[0, 1] -= x * y * v
                    iT[1, 2] -= y * z * v
                    iT[0, 2] -= x * z * v
    iT[1, 0] = iT[0, 1]
    iT[2, 1] = iT[1, 2]
    iT[2, 0] = iT[0, 2]
    return iT


def getNeighborhoodPolylines(vtx, radius, vtxDict):
    heads = set()
    v = vtx
    xCtr, yCtr, zCtr = v.realCoords
    vpId = v.parent
    while vpId in vtxDict:
        vp = vtxDict[vpId]
        x, y, z = vp.realCoords
        dx, dy, dz = x - xCtr, y - yCtr, z - zCtr
        if dx * dx + dy * dy + dz * dz > radius * radius:
            break
        else:
            v = vp
            vpId = vp.parent
    heads.add(v)

    vtxListList = []
    while heads:
        vL = []
        v = heads.pop()
        while True:
            x, y, z = v.realCoords
            dx, dy, dz = x - xCtr, y - yCtr, z - zCtr
            if dx * dx + dy * dy + dz * dz > radius * radius:
                break
            vL.append(v)
            kids = [k for k in v.kids if k in vtxDict]
            if kids:
                nxtV = vtxDict[kids.pop()]
                if kids:
                    # There is a split here
                    forkV = v.copyNoKids()
                    for k in kids:
                        forkV.addChild(k)
                    heads.add(forkV)
                v = nxtV
            else:
                break
        vtxListList.append(vL)
        # print 'seg %s has %s verts' % (len(vtxListList), len(vL))

    return vtxListList


def calcAligningEulerZYZ(dCube, sig=(+1.0, +1.0, +1.0)):
    iT = calcMomentTensor(dCube, sig)
    eVals, eVecs = np.linalg.eigh(iT)
    # print 'Eigenvalues and eigenvectors:'
    # print 'Moment tensor:'
    # print iT
    # print eVals
    # print eVecs
    sL = []
    for i in xrange(eVals.shape[0]):
        sL.append((eVals[i], i))
    sL.sort(reverse=True)
    sortedEVecs = [np.matrix(eVecs[:, i]) for a, i in sL]  # @UnusedVariable
    sortedEVals = [a for a, i in sL]  # @UnusedVariable
    # print 'Unrotated eigenvectors: %s' % sortedEVecs
    # print 'eigen test'
    # mIT = np.matrix(iT)
    # for i in xrange(3):
    #     print 'Case %d:' % i
    #     v = np.matrix(sortedEVecs[i]).transpose()
    #     print (mIT * v) - (sortedEVals[i] * v)

    fmZ = sortedEVecs[0].transpose()
    fmX = sortedEVecs[1].transpose()
    fmY = np.cross(fmZ.transpose(), fmX.transpose()).transpose()
    toX = np.matrix([1.0, 0.0, 0.0]).transpose()
    toY = np.matrix([0.0, 1.0, 0.0]).transpose()
    toZ = np.matrix([0.0, 0.0, 1.0]).transpose()
    aligningTrans = makeAligningRotation(fmX, fmY, fmZ,
                                         toX, toY, toZ)

    # print 'aligning rotation:'
    # print aligningTrans

    thetaZ0, thetaY, thetaZ1 = transToEulerRzRyRz(aligningTrans)
    # print 'Aligning Euler angles: %s %s %s' % (thetaZ0, thetaY, thetaZ1)

    return (thetaZ0, thetaY, thetaZ1)


def _basics(sampleVtx):
    yamlDict = {}
    yamlDict['vtxId'] = sampleVtx.id
    yamlDict['vtxParent'] = sampleVtx.parent
    yamlDict['vtxKids'] = [int(vid) for vid in sampleVtx.kids]
    yamlDict['vtxObj'] = sampleVtx.objId
    yamlDict['absCoords'] = list(sampleVtx.absCoords)
    yamlDict['realCoords'] = list(sampleVtx.realCoords)
    yamlDict['edges'] = [list(e) for e in sampleVtx.edges]
    return yamlDict


def _addDerivedEntries(yamlDict, dCube, baseName, sampId,
                       maxL, rMax, transformer, sig):
    thetaZ0, thetaY, thetaZ1 = calcAligningEulerZYZ(dCube, sig=sig)
    yamlDict['eulerZYZ'] = [float(thetaZ0), float(thetaY), float(thetaZ1)]

    # print 'reconstructed aligning rotation:'
    # rR = eulerRzRyRzToTrans(thetaZ0, thetaY, thetaZ1)
    # print rR

    doubleCube = np.require(dCube, dtype=np.double,
                            requirements=['C_CONTIGUOUS'])
    transformer.initLChain()
    thetaglq, phiglq, _, _ = transformer.prepL(maxL)
    zMat = sig[2]*np.outer(np.cos(thetaglq), np.ones_like(phiglq))
    xMat = sig[0]*np.outer(np.sin(thetaglq), np.cos(phiglq))
    yMat = sig[1]*np.outer(np.sin(thetaglq), np.sin(phiglq))
    edgeMat = np.zeros((len(thetaglq), len(phiglq)), dtype=np.float64)
    for edge in yamlDict['edges']:
        cosMat = (xMat * edge[0]) + (yMat * edge[1]) + (zMat * edge[2])
        edgeMat += np.power(cosMat, 20.0)
    addFieldEntry(yamlDict, edgeMat, baseName, 'edgeSamp', sampId)

    edgeHarm = transformer.expand(edgeMat)
    addFieldEntry(yamlDict, edgeHarm, baseName, 'edgeYlm', sampId)

    rotEdgeHarm = transformer.rotateHarmonics(edgeHarm, thetaZ0, thetaY, thetaZ1)
    addFieldEntry(yamlDict, rotEdgeHarm, baseName, 'rotEdgeYlm', sampId)
    rotEdge = transformer.reconstructSamples(rotEdgeHarm)
    addFieldEntry(yamlDict, rotEdge, baseName, 'rotEdgeSamp', sampId)

    samples = transformer.calcSphereOfSamples(doubleCube, rMax, sig=sig)
    addFieldEntry(yamlDict, samples, baseName, 'samp', sampId)
    harmonics = transformer.expand(samples)
    addFieldEntry(yamlDict, harmonics, baseName, 'sampYlm', sampId)

    # unRotReconSamples = transformer.reconstructSamples(harmonics)
    rotHarm = transformer.rotateHarmonics(harmonics, thetaZ0, thetaY, thetaZ1)
    addFieldEntry(yamlDict, rotHarm, baseName, 'rotSampYlm', sampId)
    rotSamples = transformer.reconstructSamples(rotHarm)
    addFieldEntry(yamlDict, rotSamples, baseName, 'rotSamp', sampId)

    sphereFileName = "%s_%d_spheres.vtk" % (baseName, sampId)
    plotSphere(thetaglq, phiglq,
               {'edges': edgeMat.transpose(),
                'rotEdges': rotEdge.transpose(),
                'samples': samples.transpose(),
                'rotSamples': rotSamples.transpose()                
                },
               fname=sphereFileName)
    yamlDict['sphereFile'] = sphereFileName

    ballSamples = transformer.calcBallOfSamples(doubleCube, sig=sig)
    addFieldEntry(yamlDict, ballSamples, baseName, 'ballSamp', sampId)
    ballExpansion = transformer.expandBall(ballSamples)
    addFieldEntry(yamlDict, ballExpansion, baseName, 'ballYlm', sampId)
    rotBallExpansion = transformer.rotateBall(ballExpansion, thetaZ0, thetaY, thetaZ1)
    addFieldEntry(yamlDict, rotBallExpansion, baseName, 'rotBallYlm', sampId)
    ballRecon = transformer.reconstructBall(rotBallExpansion)
    addFieldEntry(yamlDict, ballRecon, baseName, 'rotBallSamp', sampId)

    ballChain = transformer.getBallStructure()
    layerList = [(r, thetaglq, phiglq)
                 for r, thetaglq, phiglq, _, _ in ballChain]
    ballFName = "%s_%d_balls.vtk" % (baseName, sampId)
    plotBall(layerList,
             {'samples': ballSamples,
              'rotsamples': ballRecon},
             fname=ballFName)
    yamlDict['ballFile'] = ballFName
    
    

class BlockGenerator(object):
    def __init__(self, rMax, edgeLen, maxL, usefulVtxDict,
                 fishCoreFile, fishCoreXSize, fishCoreYSize,
                 fishCoreZSize, baseName='block'):
        self.rMax = rMax
        self.edgeLen = edgeLen
        self.maxL = maxL
        self.usefulVtxDict = usefulVtxDict
        self.transformer = SHTransformer(edgeLen, maxL)
        self.sampler = ArraySampler(edgeLen, fishCoreFile,
                                    fishCoreXSize, fishCoreYSize,
                                    fishCoreZSize)
        self.baseName = baseName

    def writeBlock(self, sampId, sampLocInfo):
        sampleVtx = self.usefulVtxDict[sampId]
        grabBlock(sampleVtx, self.usefulVtxDict, self.sampler,
                  self.transformer, self.rMax, self.edgeLen,
                  self.maxL, sampLocInfo['xOffset'],
                  sampLocInfo['yOffset'], sampLocInfo['zOffset'],
                  baseName=self.baseName)

def grabBlock(sampleVtx, usefulVtxDict, sampler, transformer,
              rMax, edgeLen, maxL, xOffset, yOffset, zOffset,
              baseName='block'):
    yamlDict = _basics(sampleVtx)

    xLoc = sampleVtx.x
    yLoc = sampleVtx.y
    zLoc = sampleVtx.z
    dCube = sampler.sample({'xLoc': xLoc, 'yLoc': yLoc, 'zLoc': zLoc,
                            'xOffset': xOffset, 
                            'yOffset': yOffset,
                            'zOffset': zOffset})

    bovBaseName = '%sCube_%d' % (baseName, sampleVtx.id)
    writeBOV(bovBaseName, dCube, 1.0, 1.0, 1.0, sig=(1.0, -1.0, 1.0))
    yamlDict['bovFiles'] = [bovBaseName + '.bov',
                            bovBaseName + '.bytes']
    
    traceName = '%s_trace_%d.vtk' % (baseName, sampleVtx.id)
    halfEdge = float(edgeLen)/2.0
    writeVtkPolylines(traceName,
                      getNeighborhoodPolylines(sampleVtx,
                                               rMax,
                                               usefulVtxDict),
                      shift1 = (-xLoc, -yLoc, -zLoc),
                      shift2 = (halfEdge, halfEdge, halfEdge),
                      sig=(1.0, -1.0, 1.0))
    yamlDict['traceFile'] = traceName

    _addDerivedEntries(yamlDict, dCube, baseName, sampleVtx.id, 
                       maxL, rMax, transformer, sig=(1.0, -1.0, 1.0))

    with open('%s_%d_%d.yaml' % (baseName, edgeLen, sampleVtx.id),
              'w') as f:
        yaml.dump(yamlDict, f)


class SynthBlockGenerator(object):
    def __init__(self, rMax, edgeLen, maxL, baseName='synth',
                 sig=(1.0, 1.0, 1.0)):
        self.rMax = rMax
        self.edgeLen = edgeLen
        self.maxL = maxL
        self.transformer = SHTransformer(edgeLen, maxL)
        self.sampler = CylinderSampler(edgeLen, sig=sig)
        self.baseName = baseName

    def writeBlock(self, sampId, sampLocInfo):
        synthBlock(sampId, sampLocInfo,
                   self.sampler, self.transformer, self.rMax,
                   self.edgeLen, self.maxL,
                   baseName=self.baseName)

def addFieldEntry(yamlDict, matrix, baseName, fieldName, sampId,
                  writeData=True):
    matFName = ('%s_%s_%s%s'
                % (baseName, fieldName, sampId, dtypeToExt(matrix)))
    yamlDict[fieldName + 'File'] = matFName
    yamlDict[fieldName + 'Shape'] = list(matrix.shape)
    yamlDict[fieldName + 'FtnOrder'] = np.isfortran(matrix)
    if writeData:
        matrix.tofile(matFName)


def loadFieldEntry(yamlDict, baseName, fieldName):
    sampId = yamlDict['vtxId']
    fname = yamlDict[fieldName + 'File']
    shape = yamlDict[fieldName + 'Shape']
    isFtnOrder = yamlDict[fieldName + 'FtnOrder']
    dtype = extToDtype(os.path.splitext(fname)[1])
    count = 1
    for dim in yamlDict[fieldName + 'Shape']:
        count *= dim
    rawData = np.fromfile(fname, dtype=dtype, count=count)
    if isFtnOrder:
        order = 'F'
    else:
        order = 'C'
    result = np.reshape(rawData, shape, order=order)
    return result


def synthBlock(sampId, sampLocInfo, sampler, transformer,
              rMax, edgeLen, maxL, baseName='synth'):
    axX = sampLocInfo['axisX']
    axY = sampLocInfo['axisY']
    axZ = sampLocInfo['axisZ']
    xLoc, yLoc, zLoc = 0.0, 0.0, 0.0
    v0 = UsefulVtx.fromVtx(Vtx(sampId, 0, 0, 0.0, 0.0, 0.0), 0)
    v0.addEdge((axX, axY, axZ))
    v0.addEdge((-axX, -axY, -axZ))

    yamlDict= _basics(v0)

    dCube = sampler.sample(sampLocInfo)

    bovBaseName = '%sCube_%d' % (baseName, sampId)
    writeBOV(bovBaseName, dCube, 1.0, 1.0, 1.0, sig=(1.0, 1.0, 1.0))
    yamlDict['bovFiles'] = [bovBaseName + '.bov',
                            bovBaseName + dtypeToExt(dCube)]
    
    traceName = '%s_trace_%d.vtk' % (baseName, sampId)
    v1 = Vtx(1, 0, 0, axX, axY, axZ)
    v2 = Vtx(2, 0, 0, -axX, -axY, -axZ)
    halfEdge = float(edgeLen)/2.0
    writeVtkPolylines(traceName,
                      [[v0, v1], [v0, v2]],
                      shift1=(-xLoc, -yLoc, -zLoc),
                      shift2=(halfEdge, halfEdge, halfEdge),
                      sig=(1.0, 1.0, 1.0))
    yamlDict['traceFile'] = traceName

    _addDerivedEntries(yamlDict, dCube, baseName, sampId, 
                       maxL, rMax, transformer, sig=(1.0, 1.0, 1.0))

    with open('%s_%d_%d.yaml' % (baseName, edgeLen, sampId),
              'w') as f:
        yaml.dump(yamlDict, f)

def parseBlock(vtxId, edgeLen, baseName='block'):
    with open('%s_%d_%d.yaml' % (baseName, edgeLen, vtxId), 'rU') as f:
        yamlDict = yaml.load(f)
    return yamlDict

