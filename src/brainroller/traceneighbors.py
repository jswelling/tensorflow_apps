#! /usr/bin/env python

import pickle
import math
import numpy as np

traceIn = 'traces_clipped.pkl'
skipsIn = 'skips.txt'
traceOut = 'useful_trace_neighborhoods.pkl'

# llcLoc = np.array([4750. - 512., 2150. - 512, 4000])
# trcLoc = np.array([4750. + 512., 2150. + 512, 8999])
llcLoc = np.array([4750. - 492., 2150. - 492, 4020])
trcLoc = np.array([4750. + 492., 2150. + 492, 8979])

radCutoff = 20.0
minNbrsToKeep = 30

class Vtx(object):
    def __init__(self, vId, objId, parent, absX, absY, absZ):
        self.id = vId
        self.objId = objId
        self.parent = parent
        self.absCoords = (absX, absY, absZ)
        self.kids = []
        self.skipTable = None
        
    def setSkipTable(self, tbl):
        self.skipTable = tbl

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
        if self.skipTable:
            return (self.absCoords[2]
                    - self.skipTable[int(self.absCoords[2])])
        else:
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


def loadSkipTable(skipF, maxSlice):
    if skipF:
        skipL = [int(line) for line in skipF]
        skipS = set(skipL)
        skipTbl = {}
        offset = 0
        for i in range(maxSlice + 1):
            if i in skipS:
                offset += 1
            skipTbl[i] = offset
        return skipTbl
    else:
        return None
    


class UsefulVtx(Vtx):
    @classmethod
    def load(cls, pklF, maxSlice, skipF):
        usefulVtxDict = pickle.load(pklF)
        skipTbl = loadSkipTable(skipF, maxSlice)
        for v in list(usefulVtxDict.values()):
            v.setSkipTable(skipTbl)
        return usefulVtxDict
        
    def __init__(self, vId, objId, parent, x, y, z, nbrCt=0):
        super(UsefulVtx, self).__init__(vId, objId, parent, x, y, z)
        self.nbrCt = nbrCt
        self.edges = []

    @classmethod
    def fromVtx(cls, vtx, nbrCt):
        vx, vy, vz = vtx.absCoords
        v = cls(vtx.id, vtx.objId, vtx.parent, vx, vy, vz,
                nbrCt=nbrCt)
        for kId in vtx.kids:
            v.addChild(kId)
        if vtx.skipTable is not None:
            v.setSkipTable(vtx.skipTable)
        return v

    def addEdge(self, xyzTuple):
        if xyzTuple:
            x, y, z = xyzTuple
            xCtr, yCtr, zCtr = self.realCoords
            dx, dy, dz = x - xCtr, y - yCtr, z - zCtr
            norm = math.sqrt(dx * dx + dy * dy + dz * dz)
            if norm > 0.0:
                dx /= norm
                dy /= norm
                dz /= norm
            else:
                print('zero length edge')
            self.edges.append((dx, dy, dz))
            

def countParents(v, vtxDict):
    ctrX, ctrY, ctrZ = v.realCoords
    nbrCt = 0
    vpId = v.parent
    while vpId and vpId in vtxDict:
        vp = vtxDict[vpId]
        px, py, pz = vp.realCoords
        dx, dy, dz = px - ctrX, py - ctrY, pz - ctrZ
        if dx * dx + dy * dy + dz * dz <= radCutoff * radCutoff:
            vpId = vp.parent
            nbrCt += 1
        else:
            break
    return nbrCt


def addParentEdge(v, vtxDict):
    ctrX, ctrY, ctrZ = v.realCoords
    edgeTuple = None
    vpId = v.parent
    while vpId and vpId in vtxDict:
        vp = vtxDict[vpId]
        px, py, pz = vp.realCoords
        dx, dy, dz = px - ctrX, py - ctrY, pz - ctrZ
        if dx * dx + dy * dy + dz * dz <= radCutoff * radCutoff:
            vpId = vp.parent
            edgeTuple = vp.realCoords
        else:
            break
    v.addEdge(edgeTuple)


def innerAddKidEdges(vRoot, vHere, ctrTuple, vtxDict, edgeTuple=None):
    ctrX, ctrY, ctrZ = ctrTuple
    for kId in vHere.kids:
        if kId in vtxDict:
            kv = vtxDict[kId]
            kx, ky, kz = kv.realCoords
            dx, dy, dz = kx - ctrX, ky - ctrY, kz - ctrZ
            if dx * dx + dy * dy + dz * dz <= radCutoff * radCutoff:
                innerAddKidEdges(vRoot, kv, ctrTuple, vtxDict,
                                 edgeTuple=kv.realCoords)
            else:
                vRoot.addEdge(edgeTuple)


def addKidEdges(v, vtxDict):
    innerAddKidEdges(v, v, v.realCoords, vtxDict)


def innerCountKids(v, ctrTuple, vtxDict):
    ctrX, ctrY, ctrZ = ctrTuple
    nbrCt = 0
    for kId in v.kids:
        if kId in vtxDict:
            kv = vtxDict[kId]
            kx, ky, kz = kv.realCoords
            dx, dy, dz = kx - ctrX, ky - ctrY, kz - ctrZ
            if dx * dx + dy * dy + dz * dz <= radCutoff * radCutoff:
                nbrCt += (1 + innerCountKids(kv, ctrTuple, vtxDict))
    return nbrCt


def countKids(v, vtxDict):
    ctrX, ctrY, ctrZ = v.realCoords
    nbrCt = innerCountKids(v, (ctrX, ctrY, ctrZ), vtxDict)
    return nbrCt


def main():
    with open(traceIn, 'r') as f:
        vtxDict, objDict = pickle.load(f)
    print('%d vertices in %d objects' % (len(vtxDict), len(objDict)))
    with open(skipsIn, 'rU') as skipF:
        skipTbl = loadSkipTable(skipF, 10000)
    for v in list(vtxDict.values()):
        v.setSkipTable(skipTbl)

    locs = np.zeros((len(vtxDict), 3))
    vIds = np.zeros((len(vtxDict)), dtype=np.int_)
    offset = 0
    for vId, vtx in list(vtxDict.items()):
        x, y, z = (vtx.x, vtx.y, vtx.z)
        locs[offset, :] = (x, y, z)
        vIds[offset] = vId
        offset += 1

    clippedVIds = vIds[np.logical_and(np.all(np.greater_equal(locs, llcLoc),
                                             axis=1),
                                      np.all(np.less_equal(locs, trcLoc),
                                             axis=1))]

    nbrCtDict = {}
    usefulVtxDict = {}
    for vId in clippedVIds:
        v = vtxDict[vId]
        nParents = countParents(v, vtxDict)
        nKids = countKids(v, vtxDict)
        nNbrs = nParents + nKids
        nbrCtDict[vId] = nNbrs
        if nNbrs >= minNbrsToKeep:
            uVtx = UsefulVtx.fromVtx(v, nbrCt=nNbrs)
            addParentEdge(uVtx, vtxDict)
            addKidEdges(uVtx, vtxDict)
            usefulVtxDict[vId] = uVtx
        
    nbrCtHisto = {}
    for n in list(nbrCtDict.values()):
        if n in nbrCtHisto:
            nbrCtHisto[n] += 1
        else:
            nbrCtHisto[n] = 1

    pairL = list(nbrCtHisto.items())[:]
    pairL.sort()
    for n, ct in pairL:
        print('%d: %d' % (n, ct))

    with open(traceOut, 'w') as f:
        pickle.dump(usefulVtxDict, f)

if __name__ == '__main__':
    main()
