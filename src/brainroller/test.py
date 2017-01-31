#! /usr/bin/env python

import math
import numpy as np
from main import transToEulerRzRyRz, eulerRzRyRzToTrans, transToEulerRxRyRz, eulerRxRyRzToTrans


dToR = np.pi/180.0


def faker(thetaZ0, thetaY, thetaZ1):
    cphi = math.cos(thetaZ0)
    sphi = math.sin(thetaZ0)
    ctheta = math.cos(thetaY)
    stheta = math.sin(thetaY)
    calpha = math.cos(thetaZ1)
    salpha = math.sin(thetaZ1)
    f = np.asmatrix([[cphi*ctheta*calpha - sphi*salpha, -cphi*ctheta*salpha - sphi*calpha, cphi*stheta],
                     [sphi*ctheta*calpha + cphi*salpha, -sphi*ctheta*salpha + cphi*calpha, sphi*stheta],
                     [-stheta*calpha, stheta*salpha, ctheta]])
    return f


def testCase(phiIn, thetaIn, alphaIn, hush=True):
    try:
        t1 = eulerRzRyRzToTrans(phiIn, thetaIn, alphaIn)
        if not hush:
            print 'from main.eulerRzRyRzToTrans'
            print t1

        t2 = faker(phiIn, thetaIn, alphaIn)
        if not hush:
            print 'from faker'
            print t2

#         phi, theta, alpha = transToEulerRzRyRz(t1)
        phi, theta, alpha = transToEulerRzRyRz(t2)
        t3 = eulerRzRyRzToTrans(phi, theta, alpha)
        if not hush:
            print 'recovered phi=%s, theta=%s, alpha=%s' % (phi, theta, alpha)
            print 'reconstruction:'
            print t3

        if not np.allclose(t1, t2):
            if not np.allclose(t1, t3):
                print 'BOTH FAIL for %s %s %s' % (phiIn, thetaIn, alphaIn)
            else:
                print 'TEST! FAIL for %s %s %s' % (phiIn, thetaIn, alphaIn)
        elif not np.allclose(t1, t3):
            print 'TEST2 FAIL for %s %s %s -> %s %s %s' % (phiIn, thetaIn, alphaIn, phi, theta, alpha)
        else:
            print 'OK for %s %s %s' % (phiIn, thetaIn, alphaIn)
    except ValueError, e:
        print 'FAIL: ValueError: %s for %s %s %s' % (e, phiIn, thetaIn, alphaIn)


def testCase2(phiIn, thetaIn, alphaIn, hush=True):
    try:
        t1 = eulerRxRyRzToTrans(phiIn, thetaIn, alphaIn)
        if not hush:
            print 'from main.eulerRxRyRzToTrans'
            print t1

        phi, theta, alpha = transToEulerRxRyRz(t1)
        t3 = eulerRxRyRzToTrans(phi, theta, alpha)
        if not hush:
            print 'recovered phi=%s, theta=%s, alpha=%s' % (phi, theta, alpha)
            print 'reconstruction:'
            print t3

        if not np.allclose(t1, t3):
            print 'TEST FAIL for %s %s %s -> %s %s %s' % (phiIn, thetaIn, alphaIn, phi, theta,
                                                          alpha)
        else:
            print 'OK for %s %s %s' % (phiIn, thetaIn, alphaIn)
    except ValueError, e:
        print 'FAIL: ValueError: %s for %s %s %s' % (e, phiIn, thetaIn, alphaIn)


def main():
    allAngles = [0.0, 0.25*math.pi, 0.5*math.pi, 0.75*math.pi, 1.0*math.pi, 1.25*math.pi,
                 1.5*math.pi, 1.75*math.pi, 2.0*math.pi]

    for phi in allAngles:
        for theta in allAngles:
            for alpha in allAngles:
                testCase(phi, theta, alpha)

if __name__ == '__main__':
    main()
