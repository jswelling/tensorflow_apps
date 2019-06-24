#! /usr/bin/env python

import os
import pickle

#srcDir = '/pylon1/pscstaff/awetzel/ZF-test-files/60nm-remapped-traces'
srcDir = '/pylon1/pscstaff/awetzel/ZF-test-files/60nm-July01-remapped-traces'

class Vtx(object):
    def __init__(self, vId, objId, parent, x, y, z):
        self.id = vId
        self.objId = objId
        self.parent = parent
        self.absCoords = (x, y, z)
        self.kids = []

    def addChild(self, vtxId):
        self.kids.append(vtxId)


def main():
    vtxDict = {}
    objDict = {}
    for path in os.listdir(srcDir):
        print(path)
        if path.endswith('.re'):
            with open(os.path.join(srcDir, path), 'rU') as f:
                for line in f:
                    words = line.split()
                    vId = int(words[1])
                    objId = int(words[0])
                    parent = int(words[2])
                    x = float(words[3])
                    y = float(words[4])
                    z = float(words[5])
                    vtx = Vtx(vId, objId, parent, x, y, z)
                    vtxDict[vId] = vtx
                    if objId in objDict:
                        objDict[objId].append(vtx)
                    else:
                        objDict[objId] = [vtx]
                    if parent in vtxDict:
                        vtxDict[parent].addChild(vId)

    print('%d vertices in %d objects' % (len(vtxDict), len(objDict)))
    with open('traces.pkl', 'w') as f:
        pickle.dump((vtxDict, objDict), f)

if __name__ == '__main__':
    main()
