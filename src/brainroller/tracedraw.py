#! /usr/bin/env python

import cPickle
import numpy as np

traceIn = 'traces_clipped_tight.pkl'

def writeVTKLines(fname, vtxListList):
    with open(fname, 'w') as f:
        f.write('# vtk DataFile Version 3.0\n')
        f.write('axon trace polylines\n')
        f.write('ASCII\n')
        f.write('DATASET POLYDATA\n')
        f.write('POINTS %d float\n' % sum([len(l) for l in vtxListList]))
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


def main():
    with open(traceIn, 'r') as f:
        vtxDict, objDict = cPickle.load(f)
    print '%d vertices in %d objects' % (len(vtxDict), len(objDict))

    heads = set()
    for v in vtxDict.values():
        if not v.kids or all([(kId not in vtxDict) for kId in v.kids]):
            while (v.parent is not None and v.parent != 0 
                   and v.parent in vtxDict):
                v = vtxDict[v.parent]
            heads.add(v)

    vtxListList = []
    while heads:
        vL = []
        v = heads.pop()
        while True:
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
        print 'seg %s has %s verts' % (len(vtxListList), len(vL))
        
    writeVTKLines('traces.vtk', vtxListList)


if __name__ == '__main__':
    main()
