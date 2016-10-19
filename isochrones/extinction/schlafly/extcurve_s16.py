from ...config import on_rtd

if not on_rtd:
    import numpy
    from . import cubicspline

    # Schlafly+2016
    ra0 = numpy.array([ 0.65373283,  0.39063843,  0.20197893,  0.07871701, -0.00476316,
                       -0.14213929, -0.23660605, -0.28522577, -0.321301  , -0.33503192])
    dra0 = numpy.array([-0.54278669,  0.03404903,  0.36841725,  0.42265873,  0.38247769,
                         0.14148814, -0.04020524, -0.13457319, -0.26883343, -0.36269229])

    # "isoreddening wavelengths" for extinction curve, at E(g-r) = 0.65 reddening
    # T_eff = 4500, Fe/H = 0, log g = 2.5
    lam0 = numpy.array([  5032.36441067,   6280.53335141,   7571.85928312,   8690.89321059,
                          9635.52560909,  12377.04268274,  16381.78146718,  21510.20523237,
                         32949.54009328,  44809.4919175 ])


    rhk = 1.55 # Indebetouw (2005)

    def extcurve(x, ra=None, dra=None, lam=None):
        """ Return extinction curve, for R(V)-like parameter x.

        Returns the extinction curve, A(lambda)/A(5420 A), according to
        Schlafly+2016, for the parameter "x," which controls the overall shape of
        the extinction curve in an R(V)-like way.  The extinction curve returned
        is a callable function, which is then invoked with the wavelength, in
        angstroms, of interest.

        The extinction curve is based on broad band photometry between the PS1 g
        band and the WISE W2 band, which have effective wavelengths between 5000
        and 45000 A.  The extinction curve is blindly extrapolated outside that
        range.  The gray component of the extinction curve is fixed by enforcing
        A(H)/A(K) = 1.55 (Indebetouw+2005).  The gray component is relatively
        uncertain, and its variation with x is largely made up.

        Args:
            x: some number controlling the shape of the extinction curve
            ra: extinction vector at anchor wavelengths, default to Schlafly+2016
            dra: derivative of extinction vector at anchor wavelengths, default to
                 Schlafly+2016
            lam: anchor wavelengths (angstroms), default to Schlafly+2016

        Returns: the extinction curve E, so the extinction alam = A(lam)/A(5420 A)
            is given by: 
            A = extcurve(x)
            alam = A(lam)
        """

        if ra is None:
            ra = ra0
        if dra is None:
            dra = dra0
        if lam is None:
            lam = lam0

        anchors = ra + x*dra
        # fix gray component so that A(H)/A(K) = 1.55
        anchors += (-anchors[6] + rhk*anchors[7])/(1 - rhk)
        cs0 = cubicspline.CubicSpline(lam, anchors, yp='3d=0')
        # normalize at 5420 angstroms
        return cubicspline.CubicSpline(lam, anchors/cs0(5420.), yp='3d=0')
