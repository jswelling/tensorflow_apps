#! /usr/bin/env python

import cPickle
import numpy as np
from writegeom import plotLines

traceIn = 'traces.pkl'
vtkOut = 'trace_lines.vtk'

# llcLoc = np.array([4750. - 512., 2150. - 512, 4000])
# trcLoc = np.array([4750. + 512., 2150. + 512, 8999])
llcLoc = np.array([4750. - 492., 2150. - 492, 4020])
trcLoc = np.array([4750. + 492., 2150. + 492, 8979])


class Vtx(object):
    def __init__(self, vId, objId, parent, x, y, z):
        self.id = vId
        self.objId = objId
        self.parent = parent
        self.absCoords = (x, y, z)
        self.kids = []

    def addChild(self, vtxId):
        self.kids.append(vtxId)

    def copyNoKids(self):
        x, y, z = self.absCoords
        return Vtx(self.id, self.objId, self.parent, x, y, z)

    @property
    def x(self):
        return self.absCoords[0]

    @property
    def y(self):
        return self.absCoords[1]

    @property
    def z(self):
        return self.absCoords[2]


def gather(vtxDict):
    tails = []
    for vtx in vtxDict.values():
        culledKids = [k for k in vtx.kids if k in vtxDict]
        if not culledKids:
            tails.append(vtx.copyNoKids())
    segList = []
    while tails:
        tail = tails.pop()
        seg = []
        seg.append(tail)
        parent = tail.parent
        while True:
            if parent in vtxDict:
                vtx = vtxDict[parent]
                seg.append(vtx)
                parent = vtx.parent
            else:
                segList.append(seg)
                break
    return segList


def main():
    with open(traceIn, 'r') as f:
        vtxDict, objDict = cPickle.load(f)
    print '%d vertices in %d objects' % (len(vtxDict), len(objDict))

    segList = gather(vtxDict)
    for seg in segList:
        print [v.id for v in seg]
    #plotLines(vtkOut, lineList)

if __name__ == '__main__':
    main()
