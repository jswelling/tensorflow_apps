#! /usr/bin/env python

import cPickle
import numpy as np

traceIn = 'traces.pkl'


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

    nanVIds = vIds[np.any(np.isnan(locs), axis=1)]
    with open('nans.txt', 'w') as f:
        for vId in nanVIds:
            x, y, z = vtxDict[vId].absCoords
            f.write('%s: %s %s %s\n' % (vId, x, y, z))

if __name__ == '__main__':
    main()
