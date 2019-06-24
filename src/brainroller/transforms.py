#! /usr/bin/env python

'''
Created on May 31, 2016

@author: welling
'''

import math
import numpy as np


def checkDots(lbl, qTrans, vec, basisX, basisY, basisZ):
    dots = []
    for b in (basisX, basisY, basisZ):
        rVec = qTrans * vec
        dots.append(rVec.transpose().dot(b))
    print("%s: %f %f %f " % (lbl, dots[0], dots[1], dots[2]))


class Quaternion(object):
    def __init__(self, x, y, z, w):
        self.a = np.array([x, y, z, w])
        self.normalize()

    def normalize(self):
        mag = np.linalg.norm(self.a)
        if mag != 0.0:
            self.a /= mag

    @classmethod
    def fromAxisAngle(cls, axisVec, theta):
        assert np.allclose(np.linalg.norm(axisVec), 1.0), 'axisVec must be normalized'
        sin_a = math.sin(0.5 * theta)
        cos_a = math.cos(0.5 * theta)
        return cls(axisVec[0, 0] * sin_a, axisVec[1, 0] * sin_a, axisVec[2, 0] * sin_a,
                   cos_a)

    def toAxisAngle(self):
        length = np.linalg.norm(self.a[0:3])
        if length > 0.0:
            axis = np.matrix(self.a[0:3] / length).transpose()
            theta = 2.0 * math.acos(self.a[3])
        else:
            axis = np.matrix([0.0, 0.0, 1.0]).transpose()
            theta = 0.0
        return axis, theta

    def toTransform(self):
        t = np.outer(self.a, self.a)
        trans = np.matrix([[1.0 - 2.0 * (t[1, 1] + t[2, 2]),
                            2.0 * (t[0, 1] - t[2, 3]),
                            2.0 * (t[0, 2] + t[1, 3])],
                           [2.0 * (t[0, 1] + t[2, 3]),
                            1.0 - 2.0 * (t[0, 0] + t[2, 2]),
                            2.0 * (t[1, 2] - t[0, 3])],
                           [2.0 * (t[0, 2] - t[1, 3]),
                            2.0 * (t[1, 2] + t[0, 3]),
                            1.0 - 2.0 * (t[0, 0] + t[1, 1])]])
        return trans

    @classmethod
    def fromTransform(cls, t):
        trace = t.trace() + 1.0
        if trace > 0.5:
            s = 0.5 / math.sqrt(trace)
            return cls((t[2, 1] - t[1, 2]) * s,
                       (t[0, 2] - t[2, 0]) * s,
                       (t[1, 0] - t[0, 1]) * s,
                       0.25 / s)
        else:
            xd = 1.0 + t[0, 0] - t[1, 1] - t[2, 2]
            yd = 1.0 + t[1, 1] - t[0, 0] - t[2, 2]
            zd = 1.0 + t[2, 2] - t[0, 0] - t[1, 1]
            if xd > 1.0:
                S = 2.0 / math.sqrt(xd)  # = 1/X
                x = 1.0 / S
                y = 0.25*(t[0, 1] + t[1, 0]) * S
                z = 0.25*(t[0, 2] + t[2, 0]) * S
                w = 0.25*(t[2, 1] - t[1, 2]) * S
            elif yd > 1.0:
                S = 2.0 / math.sqrt(yd)  # = 1/Y
                x = 0.25 * (t[0, 1] + t[1, 0]) * S
                y = 1.0 / S
                z = 0.25*(t[1, 2] + t[2, 1]) * S
                w = 0.25*(t[0, 2] - t[2, 0]) * S
            else:
                S = 2.0 / math.sqrt(zd)  # = 1/Z
                x = 0.25*(t[0, 2] + t[2, 0]) * S
                y = 0.25*(t[1, 2] + t[2, 1]) * S
                z = 1.0 / S
                w = 0.25*(t[1, 0] - t[0, 1]) * S
            return cls(x, y, z, w)


def makeAligningRotation(fromX, fromY, fromZ, toX, toY, toZ):
    """
    Stolen from Fiasco coregister_struct_to_inplane.py with some minor tweaks
    """
    axis1 = np.cross(fromZ.transpose(), toZ.transpose()).transpose()
#     print 'fromZ'
#     print fromZ
#     print 'toZ'
#     print toZ
#     print 'axis1'
#     print axis1
    sinTheta = np.linalg.norm(axis1)
    if sinTheta == 0.0:  # fromZ is parallel to toZ
        Q1 = Quaternion.fromAxisAngle(fromZ, 0.0)  # identity rotation
    else:
        cosTheta = fromZ.transpose().dot(toZ)
        axis1 /= np.linalg.norm(axis1)
        theta = math.atan2(sinTheta, cosTheta)
        # print axis1
        # print theta
        Q1 = Quaternion.fromAxisAngle(axis1, theta)
    Q1Trans = Q1.toTransform()
    # print Q1Trans
    # print fromX
    rotFromX = Q1Trans * fromX
    axis2 = np.cross(rotFromX.transpose(), toX.transpose()).transpose()  # direction same as toZ
    if np.linalg.norm(axis2) == 0.0:
        phi = 0.0
        Q2 = Quaternion(0.0, 0.0, 0.0, 1.0)
    else:
        cosPhi = rotFromX.transpose().dot(toX)
        sinPhi = np.linalg.norm(axis2)
        axis2 /= np.linalg.norm(axis2)
        phi = math.atan2(sinPhi, cosPhi)
        Q2 = Quaternion.fromAxisAngle(axis2, phi)
    Q2Trans = Q2.toTransform()
    QTrans = Q2Trans * Q1Trans

#     identityTrans= np.matrix(np.identity(3))
#     checkDots("     Q2 rfX", Q2Trans, rotFromX, toX, toY, toZ)
#     checkDots("      raw X",identityTrans,fromX,toX,toY,toZ)
#     checkDots("  aligned X",QTrans,fromX,toX,toY,toZ)
#     checkDots("      raw Y",identityTrans,fromY,toX,toY,toZ)
#     checkDots("  aligned Y",QTrans,fromY,toX,toY,toZ)
#     checkDots("      raw Z",identityTrans,fromZ,toX,toY,toZ)
#     checkDots("  aligned Z",QTrans,fromZ,toX,toY,toZ)

    return QTrans


def transToEulerRzRyRz(R):
    if R[2, 2] == 1.0:
        theta = 0.0
        phi = 0.0
        alpha = math.atan2(R[1, 0], R[0, 0])
        if math.sin(alpha) * R[1, 0] < 0.0:
            alpha += math.pi
    elif R[1, 2] == 0.0:
        phi = 0.0
        alpha = math.atan2(R[1, 0], R[1, 1])
        if math.sin(alpha) * R[1, 0] < 0.0:
            alpha += math.pi
        theta = math.atan2(R[0, 2], R[2, 2])
        if math.sin(theta)*R[0, 2] < 0.0:
            theta += math.pi
        if R[0, 2] > 0.0:
            if theta < 0.0 or theta > math.pi:
                theta += math.pi
        else:
            if theta > 0.0 and theta < math.pi:
                theta += math.pi
    elif R[2, 1] == 0.0 and R[2, 0] != 0.0:
        alpha = 0.0
        phi = -math.atan2(R[0, 1], R[1, 1])
        if R[0, 1] * math.sin(phi) > 0.0:
            phi += math.pi
        theta = math.atan2(R[0, 2], R[0, 0])
        if math.cos(phi)*math.sin(theta)*R[0, 2] < 0.0:
            theta += math.pi
    else:
        theta = math.acos(R[2, 2])
        phi = math.atan2(R[1, 2], R[0, 2])
        if math.sin(theta)*math.sin(phi)*R[1, 2] < 0.0:
            phi += math.pi
        alpha = math.atan2(-R[2, 1], R[2, 0])
        if math.sin(theta)*math.sin(alpha)*R[2, 1] < 0.0:
            alpha += math.pi

    return phi, theta, alpha


def eulerRzRyRzToTrans(thetaZ0, thetaY, thetaZ1):
    cphi = math.cos(thetaZ0)
    sphi = math.sin(thetaZ0)
    ctheta = math.cos(thetaY)
    stheta = math.sin(thetaY)
    calpha = math.cos(thetaZ1)
    salpha = math.sin(thetaZ1)
    f = np.asmatrix([[cphi*ctheta*calpha - sphi*salpha, -cphi*ctheta*salpha - sphi*calpha,
                      cphi*stheta],
                     [sphi*ctheta*calpha + cphi*salpha, -sphi*ctheta*salpha + cphi*calpha,
                      sphi*stheta],
                     [-stheta*calpha, stheta*salpha, ctheta]])
    return f


def eulerRzRyRzToTrans_v2(thetaZ0, thetaY, thetaZ1):
    """
    This version has more rounding error
    """
    vZ = np.matrix([0.0, 0.0, 1.0]).transpose()
    vY = np.matrix([0.0, 1.0, 0.0]).transpose()
    r0 = Quaternion.fromAxisAngle(vZ, thetaZ0).toTransform()
    r0vY = r0 * vY
    r0vZ = r0 * vZ
    r1 = Quaternion.fromAxisAngle(r0vY, thetaY).toTransform()
    r1r0vZ = r1 * r0vZ
    r2 = Quaternion.fromAxisAngle(r1r0vZ, thetaZ1).toTransform()
    return (r2 * r1 * r0)


def transToEulerRzRyRx(R):
    """
    This is based on 'Computing Euler angles from a rotation matrix',
    Gregory G. Slabaugh
    """
    if math.fabs(R[2, 0]) != 1.0:
        theta1 = -math.asin(R[2, 0])
        psi1 = math.atan2(R[2, 1]/math.cos(theta1), R[2, 2]/math.cos(theta1))
        phi1 = math.atan2(R[1, 0]/math.cos(theta1), R[0, 0]/math.cos(theta1))
    else:
        phi1 = 0.0  # can be anything
        if (R[2, 0] == -1.0):
            theta1 = 0.5 * math.pi
            psi1 = phi1 + math.atan2(R[0, 1], R[0, 2])
        else:
            theta1 = -0.5 * math.pi
            psi1 = -phi1 + math.atan2(-R[0, 1], -R[0, 2])

    return phi1, theta1, psi1


def eulerRzRyRxToTrans(thetaZ, thetaY, thetaX):
    """
    This is based on 'Computing Euler angles from a rotation matrix',
    Gregory G. Slabaugh
    """
    cphi = math.cos(thetaZ)
    sphi = math.sin(thetaZ)
    ctheta = math.cos(thetaY)
    stheta = math.sin(thetaY)
    cpsi = math.cos(thetaX)
    spsi = math.sin(thetaX)
    return np.asmatrix([[ctheta*cphi, spsi*stheta*cphi - cpsi*sphi, cpsi*stheta*cphi + spsi*sphi],
                        [ctheta*sphi, spsi*stheta*sphi + cpsi*cphi, cpsi*stheta*sphi - spsi*cphi],
                        [-stheta, spsi*ctheta, cpsi*ctheta]])


def testCase(phiIn, thetaIn, alphaIn, testThis, testThat, hush=True):
    try:
        t1 = testThis(phiIn, thetaIn, alphaIn)
        if not hush:
            print('generated transform')
            print(t1)

        phi, theta, alpha = testThat(t1)
        t3 = testThis(phi, theta, alpha)
        if not hush:
            print('recovered phi=%s, theta=%s, alpha=%s' % (phi, theta, alpha))
            print('reconstruction:')
            print(t3)

        if np.allclose(t1, t3):
            print('OK for %s %s %s' % (phiIn, thetaIn, alphaIn))
        else:
            print('TEST FAIL for %s %s %s -> %s %s %s' % (phiIn, thetaIn, alphaIn, phi, theta,
                                                          alpha))
    except ValueError as e:
        print('FAIL: ValueError: %s for %s %s %s' % (e, phiIn, thetaIn, alphaIn))


def main():
    allAngles = [0.0, 0.25*math.pi, 0.5*math.pi, 0.75*math.pi, 1.0*math.pi, 1.25*math.pi,
                 1.5*math.pi, 1.75*math.pi, 2.0*math.pi]

    print('Testing RzRyRz <-> Trans')
    for phi in allAngles:
        for theta in allAngles:
            for alpha in allAngles:
                testCase(phi, theta, alpha, eulerRzRyRzToTrans, transToEulerRzRyRz)
    print('Testing RxRyRz <-> Trans')
    for phi in allAngles:
        for theta in allAngles:
            for alpha in allAngles:
                testCase(phi, theta, alpha, eulerRzRyRxToTrans, transToEulerRzRyRx)
    print('Done with tests')

if __name__ == '__main__':
    main()
