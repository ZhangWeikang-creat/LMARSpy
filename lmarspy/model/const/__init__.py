# Purpose: Constants in the atmosphere for LMARS model
from lmarspy.backend.field import Field
from lmarspy.backend.cal import cal

class Const:

    RADIUS = Field(data=6.3712e6)
    PI = Field(data=3.1415926535897931)
    OMEGA = Field(data=7.2921e-5)
    GRAV = Field(data=9.80665)
    RDGAS = Field(data=287.05)
    RVGAS = Field(data=461.50)
    CP_AIR = Field(data=1004.6)
    KAPPA = Field(data=RDGAS/CP_AIR)

    E2m1 = Field(data=-1./6.)
    E2c0  = Field(data=5./6.)
    E2p1 = Field(data=1./3.)

    W2m1 = Field(data=E2p1)
    W2c0  = Field(data=E2c0)
    W2p1 = Field(data=E2m1)

    E3m2 = Field(data=1./30.)
    E3m1 = Field(data=-13./60.)
    E3c0  = Field(data=47./60.)
    E3p1 = Field(data=9./20.)
    E3p2 = Field(data=-1./20.)

    W3m2 = Field(data=E3p2)
    W3m1 = Field(data=E3p1)
    W3c0  = Field(data=E3c0)
    W3p1 = Field(data=E3m1)
    W3p2 = Field(data=E3m2)

    vare = Field(data=0.000001)
    vare12 = Field(data=1E-12)


