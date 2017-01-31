#! /usr/bin/env python

import cPickle
import numpy as np

traceIn = 'traces.pkl'
traceOut = 'traces_clipped_tight.pkl'

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


def main():
    with open(traceIn, 'r') as f:
        vtxDict, objDict = cPickle.load(f)
    print '%d vertices in %d objects' % (len(vtxDict), len(objDict))

    locs = np.zeros((len(vtxDict), 3))
    vIds = np.zeros((len(vtxDict)), dtype=np.int_)
    offset = 0
    for vId, vtx in vtxDict.items():
        x, y, z = vtx.absCoords
        locs[offset, :] = (x, y, z)
        vIds[offset] = vId
        offset += 1

    keepVIds = vIds[np.logical_and(np.all(np.greater_equal(locs, llcLoc), axis=1),
                                   np.all(np.less_equal(locs, trcLoc), axis=1))]
    keepVtxDict = {}
    keepObjDict = {}
    for vId in keepVIds:
        vtx = vtxDict[vId].copyNoKids()
        keepVtxDict[vId] = vtx
        if vtx.objId in keepObjDict:
            keepObjDict[vtx.objId].append(vtx)
    for vId, vtx in keepVtxDict.items():
        if vtx.parent in keepVtxDict:
            keepVtxDict[vtx.parent].addChild(vId)

    with open(traceOut, 'w') as f:
        cPickle.dump((keepVtxDict, keepObjDict), f)

if __name__ == '__main__':
    main()
