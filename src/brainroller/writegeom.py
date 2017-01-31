#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import numpy as np


def writeVtkPolylines(fname, vtxListList, 
                      shift1 = None, shift2 = None,
                      sig = (+1.0, +1.0, +1.0)):
    with open(fname, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('axon trace polylines\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS %d float\n' % sum([len(l) for l in vtxListList]))
        sx, sy, sz = sig
        if shift1 or shift2:
            if shift1:
                s1 = shift1
            else:
                s1 = [0.0, 0.0, 0.0]
            if shift2:
                s2 = shift2
            else:
                s2 = [0.0, 0.0, 0.0]
            for vL in vtxListList:
                for v in vL:
                    vx = (sx * (v.x + s1[0])) + s2[0]
                    vy = (sy * (v.y + s1[1])) + s2[1]
                    vz = (sz * (v.z + s1[2])) + s2[2]
                    f.write('%f %f %f\n' % (vx, vy, vz))
        else:
            for vL in vtxListList:
                for v in vL:
                    f.write('%f %f %f\n' % (v.x, v.y, v.z))
        f.write('\n')
        f.write('LINES %d %d\n' % (len(vtxListList),
                                   sum([(len(l) + 1) for l in vtxListList])))
        offset = 0
        for vL in vtxListList:
            f.write('%d %s\n'
                    % (len(vL),
                       ' '.join(['%d' % i for i
                                 in xrange(offset, offset+len(vL))])))
            offset += len(vL)
        f.write('\n')
        f.write('POINT_DATA %d\n' % sum([len(l) for l in vtxListList]))
        f.write('SCALARS vID int 1\n')
        f.write('LOOKUP_TABLE default\n')
        for vL in vtxListList:
            for v in vL:
                f.write('%d\n' % v.id)
        f.write('SCALARS objID int 1\n')
        f.write('LOOKUP_TABLE default\n')
        for vL in vtxListList:
            for v in vL:
                f.write('%d\n' % v.objId)
            
# def writeVtkPolylines(fname, vtxList, polyList, ptArrayDict):
#     with open(fname, "w") as f:
#         f.write("# vtk DataFile Version 2.0\n")
#         f.write("Output of brainroller.py\n")
#         f.write("ASCII\n")
#         f.write("DATASET POLYDATA\n")
#         f.write("POINTS %d float\n" % len(vtxList))
#         for x, y, z in vtxList:
#             f.write("%f %f %f\n" % (x, y, z))
#         f.write("\n")
#         f.write("LINES %d %d\n" % (len(polyList),
#                                       sum([1 + len(l) for l in polyList])))
#         for ply in polyList:
#             f.write('%d %s\n' % (len(ply), ' '.join(['%d' % idx for idx in ply])))
#         f.write('\n')
#         ptArrayLen = None
#         for nm, ptArray in ptArrayDict.items():
#             if ptArrayLen is None:
#                 ptArrayLen = len(ptArray)
#                 f.write("POINT_DATA %d\n" % ptArrayLen)
#             else:
#                 assert len(ptArray) == ptArrayLen, 'Scalar arrays do not match'
#             f.write("SCALARS %s float 1\n" % nm)
#             f.write("LOOKUP_TABLE default\n")
#             for v in ptArray:
#                 f.write('%f\n' % v)
#         f.write('\n')


# def plotLines(fname, lineList):
#     """
#     lineList is of the form [[vtx01 vtx02...][vtx11 vtx12...]...] where the
#     vtx objects have attributes x, y, and z.
#     """
#     vtxList = []
#     segList = []
#     offset = 0
#     for line in lineList:
#         seg = []
#         for vtx in line:
#             vtxList.append((vtx.x, vtx.y, vtx.z))
#             seg.append(offset)
#             offset += 1
#         segList.append(seg)
#     writeVtkPolylines(fname, vtxList, segList, {})


def writeVtkPolygons(fname, vtxList, polyList, ptArrayDict,
                     sig = (+1.0, +1.0, +1.0)):
    with open(fname, "w") as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Output of brainroller.py\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write("POINTS %d float\n" % len(vtxList))
        sx, sy, sz = sig
        for x, y, z in vtxList:
            f.write("%f %f %f\n" % (sx * x, sy * y, sz * z))
        f.write("\n")
        f.write("POLYGONS %d %d\n" % (len(polyList),
                                      sum([1 + len(l) for l in polyList])))
        for ply in polyList:
            f.write('%d %s\n' % (len(ply), ' '.join(['%d' % idx for idx in ply])))
        f.write('\n')
        ptArrayLen = None
        for nm, ptArray in ptArrayDict.items():
            if ptArrayLen is None:
                ptArrayLen = len(ptArray)
                f.write("POINT_DATA %d\n" % ptArrayLen)
            else:
                assert len(ptArray) == ptArrayLen, 'Scalar arrays do not match'
            f.write("SCALARS %s float 1\n" % nm)
            f.write("LOOKUP_TABLE default\n")
            for v in ptArray:
                f.write('%f\n' % v)
        f.write('\n')


def plotSphere(thetaVec, phiVec, magArrayDict, fname='test.vtk'):
    x = 10 * np.outer(np.cos(phiVec), np.sin(thetaVec))
    y = 10 * np.outer(np.sin(phiVec), np.sin(thetaVec))
    z = 10 * np.outer(np.ones(np.size(phiVec)), np.cos(thetaVec))
    vtxList = []
    for tpl in zip(list(x.flatten('F')), list(y.flatten(['F'])), list(z.flatten('F'))):
        vtxList.append(tpl)
    polyList = []
    rowStride = len(phiVec)
    rowOffset = 0
    topCapVerts = range(rowStride)  # top end cap
    polyList.append(topCapVerts)
    for j in xrange(len(thetaVec) - 1):
        rowOffset = j * rowStride
        offset = rowOffset
        for i in xrange(len(phiVec) - 1):  # @UnusedVariable
            polyList.append([offset + rowStride, offset + 1 + rowStride, offset+1, offset])
            offset += 1
        offset = rowOffset + len(phiVec) - 1
        polyList.append([offset + rowStride, rowOffset + rowStride, rowOffset, offset])
    rowOffset = (len(thetaVec) - 1) * rowStride
    botCapVerts = range(rowOffset, rowOffset+rowStride)  # bottom end cap
    botCapVerts.reverse()  # for right hand rule
    polyList.append(botCapVerts)
    flatMagArrayDict = {}
    for nm, arr in magArrayDict.items():
        flatMagArrayDict[nm] = arr.flatten('F')
    writeVtkPolygons(fname, vtxList, polyList, flatMagArrayDict)


def plotBall(layerList, magArrayDict, fname='test.vtk'):
    """
    layerList is a list of tuples, each of the form (r, thetaVec, phiVec)
    magListDict is a dict of the form {name: magArray} where magArray
      is a flattened array of magnitudes in radius order.
    """
    layerDict = {}
    vtxList = []
    polyList = []
    for r, thetaVec, phiVec in layerList:
        vtxBase = len(vtxList)
        polyBase = len(polyList)
        layerDict[r] = (vtxBase, polyBase)
        x = r * np.outer(np.cos(phiVec), np.sin(thetaVec))
        y = r * np.outer(np.sin(phiVec), np.sin(thetaVec))
        z = r * np.outer(np.ones(np.size(phiVec)), np.cos(thetaVec))
        for tpl in zip(list(x.flatten('F')), list(y.flatten(['F'])), list(z.flatten('F'))):
            vtxList.append(tpl)
        rowStride = len(phiVec)
        rowOffset = 0
        lclPolyList = []
        topCapVerts = range(rowStride)  # top end cap
        lclPolyList.append(topCapVerts)
        for j in xrange(len(thetaVec) - 1):
            rowOffset = j * rowStride
            offset = rowOffset
            for i in xrange(len(phiVec) - 1):  # @UnusedVariable
                lclPolyList.append([offset + rowStride,
                                    offset + 1 + rowStride,
                                    offset+1, offset])
                offset += 1
            offset = rowOffset + len(phiVec) - 1
            lclPolyList.append([offset + rowStride,
                                rowOffset + rowStride,
                                rowOffset, offset])
        rowOffset = (len(thetaVec) - 1) * rowStride
        botCapVerts = range(rowOffset, rowOffset+rowStride)  # bottom end cap
        botCapVerts.reverse()  # for right hand rule
        lclPolyList.append(botCapVerts)
        for lclSubList in lclPolyList:
            subList = [idx + vtxBase for idx in lclSubList]
            polyList.append(subList)
    flatMagArrayDict = {}
    for nm, arr in magArrayDict.items():
        flatMagArrayDict[nm] = arr.flatten('F')
    writeVtkPolygons(fname, vtxList, polyList, flatMagArrayDict)


# def plotSphere(thetaVec, phiVec, magArray):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     x = 10 * np.outer(np.cos(phiVec), np.sin(thetaVec))
#     y = 10 * np.outer(np.sin(phiVec), np.sin(thetaVec))
#     z = 10 * np.outer(np.ones(np.size(phiVec)), np.cos(thetaVec))
#     mMin = magArray.min()
#     mMax = magArray.max()
#     fcVals = (magArray - mMin)/(mMax - mMin)
#     surf = ax.plot_surface(x, y, z, rstride=4, cstride=4,
#                            facecolors=cm.coolwarm(fcVals))
#     m = cm.ScalarMappable(cmap=cm.coolwarm)
#     m.set_array(magArray)
#     fig.colorbar(m, shrink=0.5, aspect=5)
#
#     plt.show()


def dtypeToExt(npArray):
    if npArray.dtype == np.float64:
        ext = '.doubles'
    elif npArray.dtype == np.float32:
        ext = '.floats'
    elif (npArray.dtype == np.byte or npArray.dtype == np.uint8):
        ext = '.bytes'
    return ext

def extToDtype(ext):
    if ext == '.doubles':
        dtype = np.float64
    elif ext == '.floats':
        dtype = np.float32
    elif ext == '.bytes':
        dtype = np.uint8
    return dtype


def writeBOV(fnameRoot, data, xVox, yVox, zVox, sig = (+1.0, +1.0, +1.0)):
    ext = dtypeToExt(data)
    if data.dtype == np.float64:
        typeStr = 'DOUBLE'
    elif data.dtype == np.float32:
        typeStr = 'FLOAT'
    elif (data.dtype == np.byte or data.dtype == np.uint8):
        typeStr = 'BYTE'
    if sig[0] < 0:
        data = data[::-1, :, :]
    if sig[1] < 0:
        data = data[:, ::-1, :]
    if sig[2] < 0:
        data = data[:, :, ::-1]
    np.ravel(data, order='F').tofile(fnameRoot + ext)
    with open(fnameRoot + '.bov', 'w') as f:
        f.write('TIME: 0\n')
        f.write('DATA_FILE: %s%s\n' % (fnameRoot, ext))
        f.write('DATA_SIZE: %d %d %d\n' % data.shape)
        f.write('DATA_FORMAT: %s\n' % typeStr)
        f.write('VARIABLE: values\n')
        f.write('DATA_ENDIAN: LITTLE\n')
        f.write('CENTERING: ZONAL\n')
        f.write('BRICK_ORIGIN: 0.0 0.0 0.0\n')
        sx, sy, sz = sig
        f.write('BRICK_SIZE: %f %f %f\n' % (data.shape[0] * xVox,
                                            data.shape[1] * yVox,
                                            data.shape[2] * zVox))


