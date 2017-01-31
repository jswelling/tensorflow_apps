#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import sys
import random

sys.path.extend(['/home/welling/Fiasco/fiasco_final/bin/LINUXX86_64',
                 '/home/welling/shtools/SHTOOLS-3.2'])
# sys.path.extend(['/home/welling/Fiasco/Fiasco_final/src/fmri',
#                  '/home/welling/Fiasco/Fiasco_final/bin/LINUXX86_64',
#                  '/home/welling/git/SHTOOLS'])

from yamlblocks import SynthBlockGenerator

radPixels = 20
# radPixels = 100
maxL = 48

def main():
    edgeLen = 2*radPixels + 1
    rMax = float(radPixels)
    random.seed(1234)
    blockGen = SynthBlockGenerator(rMax, edgeLen, maxL,
                                   sig=(1.0, -1.0, 1.0))

    for i in xrange(1):
        axX, axY, axZ = (1.0, 1.0, 1.0)
#         radius = 18.0
        radius = 10
        thickness = 2.0
        try:
            blockGen.writeBlock(i, {'radius': radius, 'thickness': thickness,
                                    'axisX': axX, 'axisY': axY, 'axisZ': axZ})
        except Exception, e:
            print 'Sample id %s failed: %s' % (i, e)

if __name__ == '__main__':
    main()
