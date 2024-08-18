import scipy
import numpy as np
import excor
import matplotlib.pyplot as plt
from functools import cmp_to_key

def cmp(a, b):
    return (a > b) ^ (a < b) # - sign to bitwise ^


##############################################################
###     Everything connected with fcc crystal structure  #####
###     Could be generalized to an arbitrary crystal    #####
##############################################################
class FccLattice:
    "Handles fcc crystal. Could be generalized to arbitrary crystal."
    def __init__(self, LatConst):
        self.a0 = np.array([0.5*LatConst,0.5*LatConst,0])
        self.a1 = np.array([0.5*LatConst,0,0.5*LatConst])
        self.a2 = np.array([0,0.5*LatConst,0.5*LatConst])
        Vc = np.dot(np.cross(self.a0,self.a1),self.a2) # Volume
        self.Volume = abs(Vc)
        print("Volume is ", self.Volume)
        self.b0 = (2*np.pi/Vc)*np.cross(self.a1,self.a2)
        self.b1 = (2*np.pi/Vc)*np.cross(self.a2,self.a0)
        self.b2 = (2*np.pi/Vc)*np.cross(self.a0,self.a1)
        # Special points in Brillouin zone
        brs = 2*np.pi/LatConst
        self.GPoint = [0,0,0]
        self.LPoint = np.array([0.5,  0.5,  0.5])*brs
        self.KPoint = np.array([0.75, 0.75, 0])*brs
        self.XPoint = np.array([1.0,  0.0,  0])*brs
        self.WPoint = np.array([1.0,  0.5,  0])*brs
        
    def RMuffinTin(self):
        return 0.5*np.sqrt(np.dot(self.a0,self.a0)) # Spheres just touch

    def GenerateReciprocalVectors(self, q, CutOffK):
        # Many reciprocal vectors are generated and later only the shortest are used
        Kmesh0=[]
        for n in range(-q,q+1):
            for l in range(-q,q+1):
                for m in range(-q, q+1):
                    vec = n*self.b0+l*self.b1+m*self.b2
                    if np.dot(vec,vec) <= CutOffK**2:
                        Kmesh0.append(vec)
                    
        sorted(Kmesh0, key=cmp_to_key(lambda x,y: cmp(np.dot(x,x), np.dot(y,y)))) #cmp this needs fixed
        self.Km = np.array(Kmesh0)
        print("K-mesh size=", len(self.Km))
        
    def ChoosePointsInFBZ(self, nkp, type=0): # Chooses the path in the 1BZ we will use
        
        def kv0(iq, q):
            return (iq-int((q+1.5)/2)+1)/(q+0.0)
        
        if type==0: # Choose mesh in the 1BZ to cover the whole space - for SC calculation
            kp=[]
            for i0 in range(nkp):
                r0 = kv0(i0,nkp)
                print('r0=', r0)
                for i1 in range(nkp):
                    r1 = kv0(i1,nkp)
                    for i2 in range(nkp):
                        r2 = kv0(i2,nkp)
                        k = self.b0*r0+self.b1*r1+self.b2*r2
                        kp.append(k)
            print("Number of all k-points=", len(kp))

            kpc = []
            for k in kp:
                kpc.append(sorted(k))
                
            # ChooseIrreducible k-points only
            # The function performs all symmetry operations of a cubic point-group to each k-point and
            # keeps only thos k-points which can not be obtained from another k-point by group operation.
            # These k-points are obviously irreducible.
            irkp = []       # temporary list where irreducible k points will be stored
            wkp  = []       # temporary weights
            while len(kpc)>0: # continues until all k-points are grouped into irreducible classes 
                tk = kpc[0]               # we concentrate on the k-point which is the first in the list
                irkp.append(tk)          # the first can be stored as irreducible
                wkp.append(0)            # and the weights for this irreducible k-point is set to zero
                # We go over 48 symmetry operations of cubic system:
                # Each wector component can change sign: 2^3=8 possibilities
                # All permutations of components: 3!=6
                # Since the operations are independent, we have 3!*2^3=48 operations == number of cubic point group operations
                for ix in [-1,1]:  # three loops for all possible sign changes 
                    for iy in [-1,1]:
                        for iz in [-1,1]:
                            nk = sorted([ix*tk[0], iy*tk[1], iz*tk[2]]) # sorted so that we do not need to try all permutations
                            ii=0
                            while ii<len(kpc): # This permutation and sign change leads to some element still in the list of k-points?
                                diff = sum(abs(np.array(nk) - np.array(kpc[ii]))) # casting nk and kpc[i] makes this work 
                                if diff<1e-6:
                                    del kpc[ii] # These two k-points are the same
                                    wkp[-1] += 1.
                                else:
                                    ii+=1

            # irreducible k-points are stored in the output vectors
            self.wkp = np.array(wkp)/sum(wkp)
            self.kp = np.array(irkp)

            print("Number of irreducible k points is: ", len(self.kp))
            #for ik,k in enumerate(self.kmesh):
            #    print "%10.6f"*3 % tuple(k), '  ', self.wkp[ik]
            
        else:        # Choose one particular path in the 1BZ - for plotting purposes
            nkp = 4*int(nkp/4.)+1
            print("nkp=", nkp)
            self.kp = np.zeros((nkp,3), dtype=float)
            N0=nkp//4

            self.Points = [(r'$\Gamma$', 0), (r'$X$', N0), (r'$L$', 2*N0), (r'$\Gamma$', 3*N0), (r'$K$', 4*N0)]
            for i in range(N0): self.kp[i,:]      = self.GPoint + (self.XPoint-self.GPoint)*i/(N0-0.)
            for i in range(N0): self.kp[N0+i,:]   = self.XPoint + (self.LPoint-self.XPoint)*i/(N0-0.)
            for i in range(N0): self.kp[N0*2+i,:] = self.LPoint + (self.GPoint-self.LPoint)*i/(N0-0.)
            for i in range(N0): self.kp[N0*3+i,:] = self.GPoint + (self.KPoint-self.GPoint)*i/(N0-0.)
            self.kp[4*N0] = self.KPoint

def Numerov(F, dx, f0=0.0, f1=1e-3):
    Nmax = len(F)
    dx = float(dx)
    Solution = np.zeros(Nmax, dtype=float)
    Solution[0] = f0
    Solution[1] = f1
    h2 = dx*dx;
    h12 = h2/12;
      
    w0 = (1-h12*F[0])*Solution[0];
    Fx = F[1];
    w1 = (1-h12*Fx)*Solution[1];
    Phi = Solution[1];
      
    w2 = 0.0
    for i in range(2, Nmax):
        w2 = 2*w1 - w0 + h2*Phi*Fx;
        w0 = w1;
        w1 = w2;
        Fx = F[i];
        Phi = w2/(1-h12*Fx);
        Solution[i] = Phi;
    return Solution


def NumerovGen(F, U, dx, f0=0.0, f1=1e-3):
    Nmax = len(F)
    dx = float(dx)
    Solution = np.zeros(Nmax, dtype=float)
    Solution[0] = f0
    Solution[1] = f1

    h2 = dx * dx
    h12 = h2 / 12

    w0 = Solution[0] * (1 - h12 * F[0]) - h12 * U[0]
    w1 = Solution[1] * (1 - h12 * F[1]) - h12 * U[1]
    Phi = Solution[1]

    for i in range(2, Nmax):
        Fx = F[i]
        Ux = U[i]
        w2 = 2 * w1 - w0 + h2 * (Phi * Fx + Ux)
        w0 = w1
        w1 = w2
        Phi = (w2 + h12 * Ux) / (1 - h12 * Fx)
        Solution[i] = Phi

    return Solution


def CRHS(E, l, R, Veff):
    """
    Compute the RHS for solving the Schrödinger equations by Numerov.

    Parameters:
    E (float): Energy.
    l (float): Angular momentum quantum number.
    R (numpy.ndarray): Array of radial coordinates.
    Veff (numpy.ndarray): Array of effective potential values.

    Returns:
    numpy.ndarray: Array containing the RHS values.
    """
    N = len(R)
    RHS = np.zeros(N, dtype=float)
    
    for i in range(N):
        RHS[i] = 2 * (-E + 0.5 * l * (l + 1) / (R[i] * R[i]) + Veff[i])
    
    return RHS


def SolvePoisson(Zq, R, rho):
    """
    Given the input density rho, calculates the Hartree potential.
    The boundary conditions used are U(0)=0 and U(S)=Zq.
    """
    # Compute the effective potential U from the density rho
    U = -4 * np.pi * R * np.array(rho)
    
    # Define parameters
    Nmax = len(R)
    dx = float((R[-1] - R[0]) / (Nmax - 1))
    Solution = np.zeros(Nmax, dtype=float)
    
    # Boundary conditions
    Solution[0] = 0
    Solution[1] = R[1] - R[0]  # Assumes the grid spacing is the boundary condition
    
    h2 = dx * dx
    h12 = h2 / 12

    # Numerov integration
    for i in range(2, Nmax):
        w2 = 2 * Solution[i - 1] - Solution[i - 2] + h2 * U[i - 1]
        Solution[i] = w2 + h12 * U[i]
    
    # Adjust boundary condition: U(0)=0, U(infinity)=Zq
    alpha = (Zq - Solution[-1]) / R[-1]
    Solution += alpha * R
    
    return Solution

#############################
#       LAPW Routins        #
#############################
def ComputeInterstitialOverlap(Km, RMuffinTin, Vol):
    """ Overlap in the interstitials can be calculated outside the k-loop
    
        Please see Eq.46 on page 26 for the quantity O_{K'K}^I
    """
    Olap_I = np.zeros((len(Km),len(Km)), dtype=float)
    for i in range(len(Km)):
        Olap_I[i,i] = 1 - 4*np.pi*RMuffinTin**3/(3.*Vol)
        for j in range(i+1, len(Km)):
            KKl = np.sqrt(np.dot(Km[i]-Km[j],Km[i]-Km[j]))
            fbessel = scipy.special.spherical_jn(1,KKl*RMuffinTin) #originally indexed as [0][1], changed to just scalar, i think if inputs are scalars
            Olap_I[i,j] = -4*np.pi*RMuffinTin**2*fbessel/(KKl*Vol)
            Olap_I[j,i] = Olap_I[i,j]
    return Olap_I


def Wave(Z, Enu, R0, Veff):
    """Solves the SCH Eq for Psi(Enu) and its energy derivative
       Returns logarithmic derivative, Psi(l,E) and its energy derivative
       
       Please  see Eq.30 on page 20 for definition, and Eq.49 on page 26
        for S*Psi'(S)/Psi(S) and S*dPsi'(S)/dPsi(S)
    """
    def startSol(Z, l, r):
        "good choice for starting Numerov algorithm"
        return r**(l+1)*(1-Z*r/(l+1))

    logDer=[]
    Psi_l=[]
    Psip_l=[]
    for l in range(len(Enu)):

        # Computes Psi=u/r
        crhs = CRHS(Enu[l], l, R0, Veff)
        crhs[0]=0
        ur = Numerov(crhs, (R0[-1]-R0[0])/(len(R0)-1.), 0.0, startSol(Z,l,R0[1]))
        
        ur *= 1/np.sqrt( scipy.integrate.simpson(ur*ur, x=R0) )  # normalization
        Psi_l.append( ur/R0 ) # storing Psi
        Psi_l[-1][0] = extrapolate(R0[0], R0[1], R0[2], ur[1]/R0[1], ur[2]/R0[2])
        
        # For energy derivative of Psi' = urp/r
        inhom = -2*ur
        urp = NumerovGen(crhs, inhom, (R0[-1]-R0[0])/(len(R0)-1.), 0.0, startSol(Z,l,R0[1]))

        # Energy derivative should be orthogonal
        alpha = scipy.integrate.simpson(ur*urp, x=R0)
        urp -= alpha*ur
        Psip_l.append( urp/R0 ) # storing Psip'
        Psip_l[-1][0] = extrapolate(R0[0], R0[1], R0[2], urp[1]/R0[1], urp[2]/R0[2])
        
        # <\cdot{\psi}|\cdot{\psi}>
        PsipPsip = scipy.integrate.simpson(urp*urp, x=R0)
        
        # Computes the logarithmic derivative
        v1 = crhs[-1]*ur[-1]
        v0 = crhs[-2]*ur[-2]
        w1 = crhs[-1]*urp[-1]+inhom[-1]
        w0 = crhs[-2]*urp[-2]+inhom[-2]
        dh = R0[2]-R0[1]
        dudr  = (ur[-1]-ur[-2])/dh + 0.125*dh*(3*v1+v0)
        dupdr = (urp[-1]-urp[-2])/dh + 0.125*dh*(3*w1+w0)
        
        dlogPsi = RMuffinTin*dudr/ur[-1] - 1
        dlogPsip = RMuffinTin*dupdr/urp[-1] - 1
        Psi = ur[-1]/RMuffinTin
        Psip = urp[-1]/RMuffinTin
        
        logDer.append( (Psi, Psip, dlogPsi, dlogPsip, PsipPsip) )
        
    return (logDer, Psi_l, Psip_l)

def FindCoreStates(core, R, Veff, Z, fraction=5.):
    print("finding core states...")
    "Finds all core states"
    def root(Ex, l, R, Veff):
        "For searching the core bound states"
        rhs = CRHS(Ex, l, R, Veff)
        h = (R[-1]-R[0])/(len(R)-1.)
        u = Numerov(rhs, h, R[0]*np.exp(-R[0]), R[1]*np.exp(-R[1]))
        extraplt = u[-2]*(2+h**2*rhs[-2])-u[-3]
        return u[-1]

    coreRho = np.zeros(len(R), dtype=float)
    coreE = 0
    coreZ = 0

    states=[]
    for l in range(len(core)):
        n=0                           # number of states found
        E = -0.5*Z*Z/(l+1)**2-3.      # here we starts to look for zero
        dE = abs(E)/fraction          # the length of the first step 
        decrease = abs(E)/(abs(E)-dE) # the step will decrease to zero. Check the formula!
        v0 = root(E, l, R, Veff)      # starting value
        while E<0 and n<core[l]:      # we need ncore[l] bound states
#            print("start findcore while loop")
            E += dE
            v1 = root(E, l, R, Veff)
            # print("v0: ", v0)
            # print("v1: ", v1)
            # print("while iteration: ", n)
            if v1*v0<0:
                Energy = scipy.optimize.brentq(root, E-dE, E, args=(l, R, Veff))
                # Density
                rhs = CRHS(Energy, l, R, Veff)
                u = Numerov(rhs, (R[-1]-R[0])/(len(R)-1.), R[0]*np.exp(-R[0]), R[1]*np.exp(-R[1]))
                drho = u*u
                norm = abs(scipy.integrate.simpson(drho, x=R )) #changed call signature from R to x=R
                drho *= 1./(norm*4*np.pi*R**2)
                
                coreRho += drho * (2*(2*l+1.))
                coreE   += Energy*(2*(2*l+1.))
                coreZ   += 2*(2*l+1)
                states.append( (n,l,Energy) )
                n += 1
            dE/=decrease
            v0 = v1

    print('   Found core states for (n,l)=[', end=' ')
    for state in states:
        print('(%d,%d)' % state[:2], end=' ')
    print('] E=[', end=' ')
    for state in states:
        print('%f,' % state[2], end=' ')
    print(']')
    
    return (coreRho[::-1], coreE, coreZ, states)

import scipy.special
from scipy.linalg import eigh

def ComputeEigensystem(k, Km, Olap_I, Enu, logDer, RMuffinTin, Vol, VKSi=0):
    """
    The main part of the LAPW algorithm: Implements valence H[K,K'] and O[K,K'] and diagonalizes them.
    The output are energy bands, eigenvectors and weight functions which can be used to compute
    electronic charge in real space.
    """

    def dlog_bessel_j(lmax, x):
        """Calculates the logarithmic derivative of the spherical Bessel functions."""
        if np.abs(x) < 1e-5:
            return [(l, x**l / scipy.special.factorial2(2*l+1), l*x**l / scipy.special.factorial2(2*l+1)) for l in range(lmax+1)]
        else:
            jls = lambda l : scipy.special.spherical_jn(l, x)  # all jl's and derivatives for l=[0,...lmax]
            djls = lambda l : scipy.special.spherical_jn(l, x, True) 
            return [(x * djls(l) / jls(l), jls(l), x * djls(l)) for l in range(lmax+1)]

    # Prepare omegal, C1, and PP
    omegal = np.zeros((len(Km), len(Enu)), dtype=float)
    C1 = np.zeros((len(Km), len(Enu)), dtype=float)
    PP = np.array([logDer[l][4] for l in range(len(Enu))])

    for iK, K in enumerate(Km):
        Dl_jl = dlog_bessel_j(len(Enu)-1, np.sqrt(np.dot(k + K, k + K)) * RMuffinTin)
        for l in range(len(Enu)):
            Psi, Psip, dlogPsi, dlogPsip, PsipPsip = logDer[l]
            Dl, jl, jlDl = Dl_jl[l]
            omegal[iK, l] = -Psi / Psip * (Dl - dlogPsi) / (Dl - dlogPsip)
            C1[iK, l] = np.sqrt(4 * np.pi * (2 * l + 1) / Vol) * (jlDl - jl * dlogPsip) / (Psi * (dlogPsi - dlogPsip))

    # Compute arguments for Legendre polynomials
    qv = np.zeros((len(Km), 3), dtype=float)
    qvs = np.zeros(len(Km), dtype=float)
    argums = np.zeros((len(Km), len(Km)), dtype=float)

    for iK in range(len(Km)):
        qv[iK] = Km[iK] + k
        qvs[iK] = np.linalg.norm(qv[iK])
    
    for iK in range(len(Km)):
        for jK in range(len(Km)):
            qvqv = np.dot(qv[iK], qv[jK])
            if qvs[iK] * qvs[jK] == 0:
                argums[iK, jK] = 1.0
            else:
                argums[iK, jK] = qvqv / (qvs[iK] * qvs[jK])

    # Compute Legendre polynomials
    lmax = len(Enu) - 1
    Leg = np.zeros((len(Km), len(Km), len(Enu)), dtype=float)

    for iK in range(len(Km)):
        for jK in range(len(Km)):
            x = argums[iK, jK]
            Leg[iK, jK, 0] = 1
            if lmax >= 1:
                Leg[iK, jK, 1] = x
            if lmax >= 2:
                Leg[iK, jK, 2] = 1.5 * x**2 - 0.5
            if lmax >= 3:
                Leg[iK, jK, 3] = x * (2.5 * x**2 - 1.5)
            if lmax >= 4:
                Leg[iK, jK, 4] = 0.375 * (1 - 10 * x**2 * (1 - 1.1666666666666667 * x**2))
            if lmax >= 5:
                Leg[iK, jK, 5] = 1.875 * x * (1 - 4.66666666666666667 * x**2 * (1 - 0.9 * x**2))
            for l in range(6, lmax + 1):
                p0 = 0.375 * (1 - 10 * x**2 * (1 - 1.1666666666666667 * x**2))
                p1 = 1.875 * x * (1 - 4.66666666666666667 * x**2 * (1 - 0.9 * x**2))
                for i in range(6, l + 1):
                    p2 = ((2 * i - 1) * x * p1 - (i - 1) * p0) / i
                    p0 = p1
                    p1 = p2
                Leg[iK, jK, l] = p2

    # Compute Hamiltonian and Overlap matrices
    Olap = np.zeros((len(Km), len(Km)), dtype=float)
    Ham = np.zeros((len(Km), len(Km)), dtype=float)
    C2l = np.zeros((len(Enu), len(Km), len(Km)), dtype=float)
    C2_1 = np.zeros((len(Enu), len(Km), len(Km)), dtype=float)
    C2_2 = np.zeros((len(Enu), len(Km), len(Km)), dtype=float)

    for iK in range(len(Km)):
        for jK in range(len(Km)):
            olapMT = 0
            hamMT = 0
            for l in range(len(Enu)):
                tC2l = C1[iK, l] * C1[jK, l] * Leg[iK, jK, l]
                toop = 1 + omegal[iK, l] * omegal[jK, l] * PP[l]
                olapMT += tC2l * toop
                hamMT += tC2l * (0.5 * (omegal[iK, l] + omegal[jK, l]) + toop * Enu[l])
                C2l[l, iK, jK] = tC2l
                C2_1[l, iK, jK] = tC2l * (omegal[iK, l] + omegal[jK, l])
                C2_2[l, iK, jK] = tC2l * (omegal[iK, l] * omegal[jK, l])
            Olap[iK, jK] = olapMT + Olap_I[iK, jK]
            Ham[iK, jK] = (0.25 * (qvs[iK]**2 + qvs[jK]**2) + VKSi) * Olap_I[iK, jK] + hamMT

    # Diagonalize the Hamiltonian
    Ek, Ar = eigh(Ham, Olap)

    # Calculation of weights for valence density
    w0 = np.zeros((len(Enu), len(Ar)), dtype=float)
    w1 = np.zeros((len(Enu), len(Ar)), dtype=float)
    w2 = np.zeros((len(Enu), len(Ar)), dtype=float)
    for l in range(len(Enu)):
        tw0 = Ar.T @ np.matrix(C2l[l]) @ Ar
        tw1 = Ar.T @ np.matrix(C2_1[l]) @ Ar
        tw2 = Ar.T @ np.matrix(C2_2[l]) @ Ar
        w0[l, :] = np.array([tw0[p, p] for p in range(len(Ar))])
        w1[l, :] = np.array([tw1[p, p] for p in range(len(Ar))])
        w2[l, :] = np.array([tw2[p, p] for p in range(len(Ar))])
    twi = Ar.T @ np.matrix(Olap_I) @ Ar
    wi = np.array([twi[p, p] for p in range(len(Ar))])

    return Ek, Ar, w0, w1, w2, wi

from scipy.special import expit  # Logistic sigmoid function

def rootChemicalPotential(mu, Ek, wkp, Zval, beta=50.):
    """
    Computes the valence density to find the root for the chemical potential.
    
    Parameters:
    - mu: Chemical potential
    - Ek: Energy matrix (2D array)
    - wkp: Weight function (1D array)
    - Zval: Target value
    - beta: Parameter for the Fermi-Dirac distribution

    Returns:
    - Difference between computed and target valence density
    """
    # Compute the Fermi-Dirac distribution
    x = beta * (Ek - mu)
    ferm = np.where(np.abs(x) < 100, 1 / (np.exp(x) + 1), np.where(x < 0, 1, 0))
    
    # Compute the total valence density
    Zt = np.sum(wkp[:, np.newaxis] * ferm)
    
    return 2 * Zt - Zval

def ferm(x):
    """Compute the Fermi-Dirac distribution."""
    if np.abs(x) < 100:
        return 1 / (np.exp(x) + 1)
    return 1 / (np.exp(x) + 1) if x < 0 else 0

def ComputeMTDensity(mu, Ek, wkp, w0, w1, w2, Psi_l, Psip_l, beta=50.):
    """
    Given the coefficients Eqs. 58-61 on page 30, computes the valence charge
    given the chemical potential mu using Eq. 36 on page 31.
    """
    nlmax = len(w0[0])
    wgh = np.zeros((nlmax, 3), dtype=float)

    for l in range(nlmax):
        for ik in range(len(Ek)):
            w0k = w0[ik][l, :]
            w1k = w1[ik][l, :]
            w2k = w2[ik][l, :]
            ek = Ek[ik]
            
            ferm_values = np.array([ferm(beta * (e - mu)) for e in ek])
            wgh[l, 0] += np.sum(w0k * ferm_values)
            wgh[l, 1] += np.sum(w1k * ferm_values)
            wgh[l, 2] += np.sum(w2k * ferm_values)
        
        # Optionally print intermediate results for debugging
        # print(f"{l:3d} {wgh[l,0]:20.10f} {wgh[l,1]:20.10f} {wgh[l,2]:20.10f}")

    nR = len(Psi_l[0])
    MTRho = np.zeros(nR, dtype=float)
    
    for l in range(nlmax):
        Psi_l_l = Psi_l[l]
        Psip_l_l = Psip_l[l]
        MTRho += (wgh[l, 0] * Psi_l_l**2 +
                  wgh[l, 1] * Psi_l_l * Psip_l_l +
                  wgh[l, 2] * Psip_l_l**2)
    
    MTRho *= 2 / (4 * np.pi)  # 2 due to spin

    return MTRho

def ComputeInterstitialCharge(mu, Ek, wi, wkp, beta=50.):
    """
    Computes the interstitial charge using Eq. 64 on page 31.
    """
    sIntRho = 0.0
    for ik in range(len(Ek)):
        dsum = np.sum([ferm((e - mu) * beta) * wi[ik][p] for p, e in enumerate(Ek[ik])])
        sIntRho += dsum * wkp[ik]

    sIntRho *= 2  # due to spin

    return sIntRho

from scipy.interpolate import interp1d

def Atom_cmpb(x, y):
    """Comparison function for sorting of bound states."""
    if abs(x[2] - y[2]) > 1e-4:
        return np.sign(x[2] - y[2])
    else:
        return np.sign(x[1] - y[1])

def Atom_ChargeDensity(states, R, Veff, Z):
    """
    Computes electron charge density, given the bound states and Z.
    
    Args:
        states (list): List of bound states, each represented as a tuple (n, l, E).
        R (np.ndarray): Radial grid points.
        Veff (np.ndarray): Effective potential.
        Z (float): Total number of electrons.
        
    Returns:
        tuple: (rho, Ebs) where rho is the electron charge density and Ebs is the total bound state energy.
    """
    rho = np.zeros(len(R), dtype=float)
    N = 0
    Ebs = 0
    
    for state in states:
        # Unpack the state tuple
        n, l, E = state
        
        # Compute right-hand side for Numerov integration
        rhs = CRHS(E, l, R, Veff)
        
        # Compute the grid spacing
        h = (R[-1] - R[0]) / (len(R) - 1)
        
        # Numerov integration to find u
        u = Numerov(rhs, h, R[0] * np.exp(-R[0]), R[1] * np.exp(-R[1]))
        
        # Normalize u to avoid overflow
        u /= np.sqrt(np.abs(np.sum(u) * h))
        
        # Compute u squared and normalize
        u2 = u**2
        norm = np.abs(scipy.integrate.simpson(u2, x=R))
        u2 /= norm

        # Determine the number of electrons and occupancy
        dN = 2 * (2 * l + 1)
        ferm = 1 if (N + dN) < Z else (Z - N) / dN
        drho = u2 * dN * ferm / (4 * np.pi * R**2)
        
        # Accumulate the charge density and energy
        rho += drho
        N += dN
        Ebs += E * ferm * dN
        
        # Break if the total number of electrons is reached
        if N >= Z:
            break

    # Extrapolate the density at the last point
    if len(R) > 2:
        f = interp1d(R[-3:], rho[-3:], kind='linear', fill_value="extrapolate")
        rho[-1] = f(R[-1])

    return rho, Ebs


from scipy.interpolate import interp1d

def Atom_charge(Z, core, mix=0.3, RmaxAtom=10.0, Natom=3001, precision=1.0, Nitt=100): #originally precision is 1e-5, I changed it to 10 eV for testing purposes
    """
    Computes atomic electronic density and atomic energy.

    Args:
        Z (float): Nucleus charge.
        core (list): States treated as core in LAPW (e.g., [3,2,0] for 1s, 2s, 3s, 1p, 2p, no d).
        mix (float): Mixing parameter for density.
        RmaxAtom (float): End of the radial mesh (maximum r).
        Natom (int): Number of points in radial mesh.
        precision (float): Desired precision for total energy.
        Nitt (int): Maximum number of iterations.

    Returns:
        tuple: (R0, rho) where R0 is the radial mesh and rho is the electron density.
    """

    XC = excor.ExchangeCorrelation(3)  # Exchange-correlation class

    # Radial mesh
    R0 = np.linspace(1e-10, RmaxAtom, Natom)
    Ra = R0[::-1]  # Inverse radial mesh

    # Initial effective potential
    Veff = -np.ones(len(Ra), dtype=float) / Ra

    # Core states
    catm = [c + 1 for c in core]  # Add one more state to core

    Etot_old = 0

    # Find and sort core states
    coreRho, coreE, coreZ, states = FindCoreStates(catm, Ra, Veff, Z)
    sorted(states, key=cmp_to_key(Atom_cmpb))

    # Compute initial charge density
    rho, Ebs = Atom_ChargeDensity(states, Ra, Veff, Z)
    rho = rho[::-1]

    for itt in range(Nitt):
        # Compute Hartree potential
        UHartree = SolvePoisson(Z, R0, rho)
        
        # Compute exchange-correlation potentials
        Vxc = np.array([XC.Vx(rsi) + XC.Vc(rsi) for rsi in rs(rho)])
        ExcVxc = np.array([XC.EcVc(rsi) + XC.ExVx(rsi) for rsi in rs(rho)])
        
        # Update effective potential
        Veff = (UHartree - Z) / R0 + Vxc
        Veff = Veff[::-1]

        # Find and sort core states with updated potential
        coreRho, coreE, coreZ, states = FindCoreStates(catm, Ra, Veff, Z)
        sorted(states, key=cmp_to_key(Atom_cmpb))

        # Compute new charge density
        nrho, Ebs = Atom_ChargeDensity(states, Ra, Veff, Z)

        # Compute total energy
        pot = (ExcVxc * R0**2 - 0.5 * UHartree * R0) * nrho[::-1] * 4 * np.pi
        Etot = scipy.integrate.simpson(pot, x=R0) + Ebs
        Ediff = np.abs(Etot - Etot_old)
        
        print(f'   {itt}) Etot={Etot:.6f} Eband={Ebs:.6f} Ediff={Ediff:.6f}')
        
        # Mixing
        rho = mix * nrho[::-1] + (1 - mix) * rho
        Etot_old = Etot

        # Check for convergence
        if Ediff < precision:
            break

    return R0, rho

################################
### Small utility functions   ###
#################################
def extrapolate(x, x0, x1, f0, f1):
    "linear extrapolation"
    return f0 + (f1-f0)*(x-x0)/(x1-x0)

def rs(rh):
    "rs from density -> an electron radius that corresponds to density"
    return (3/(4*np.pi*rh))**(1/3.)

DEFAULT_COLOR = '\033[0m'
RED = '\033[31;1m'
GREEN = '\033[32;1m'
BLUE = '\033[34;1m'
YELLOW = '\033[33;1m'

def Atom_cmpb(x, y):
    """Comparison function for sorting bound states."""
    return np.sign(x[2] - y[2]) if abs(x[2] - y[2]) > 1e-4 else np.sign(x[1] - y[1])

def main():
    # Input parameters
    Z = 29                    # Number of electrons in the atom
    LatConst = 6.8219117      # Lattice constant
    nkp = 6                   # Number of k-points in 1BZ: (nkp x nkp x nkp)
    core = [3, 2, 0]          # Core states: [3s, 2s, 1s, 2p, 1p, no-d]
    Enu = [0.11682, 0.18794, 0.211145, 0.3, 0.3, 0.3]  # Linearization energies
    N = 1001                  # Number of points in radial mesh
    beta = 50.                # Inverse temperature for chemical potential
    mu_mm = [-10.0, 10.0]        # Range for chemical potential search # changed interval so brentq would work, the default range is -1,1 
    CutOffK = 3.5             # Cutoff for reciprocal vectors
    DRho = 1e-3               # Convergence criteria for density
    Nitt = 100                # Maximum number of iterations
    mixRho = 0.3              # Mixing parameter for charge
    Nkplot = 200              # Number of k-points for plotting bands
    plotMM = [-1., 0.1]       # Energy range for plotting bands

    # Core number of electrons
    Zcor = sum([2 * (2 * l + 1) * nl for l, nl in enumerate(core)])
    # Valence number of electrons
    Zval = Z - Zcor

    print(f"Z core= {Zcor} and Zval= {Zval}")

    # Atomic charge for a good starting guess
    Atom_R0, Atom_rho = Atom_charge(Z, core, mix=0.3)
    AtomRhoSpline = scipy.interpolate.splrep(Atom_R0, Atom_rho, s=0)

    # Exchange-correlation class
    XC = excor.ExchangeCorrelation(3)

    # Generate lattice information
    fcc = FccLattice(LatConst)
    global RMuffinTin # added this line to make the MuffinTin Radius global
    RMuffinTin = fcc.RMuffinTin()
    VMT = 4 * np.pi * RMuffinTin**3 / 3.0
    Vinter = fcc.Volume - VMT
    print(f"Muffin-Tin radius = {RMuffinTin}")
    print(f"Volume of the MT sphere = {VMT}")
    print(f"Volume of the unit cell = {fcc.Volume}")
    print(f"Volume of the interstitial = {Vinter}")

    fcc.GenerateReciprocalVectors(4, CutOffK)
    fcc.ChoosePointsInFBZ(nkp, 0)

    # Radial mesh
    R0 = np.linspace(0, RMuffinTin, N)
    R0[0] = 1e-10
    R = R0[::-1]

    # Compute interstitial overlap
    Olap_I = ComputeInterstitialOverlap(fcc.Km, RMuffinTin, fcc.Volume)

    # Interpolate atomic charge on new mesh
    TotRho = scipy.interpolate.splev(R0, AtomRhoSpline)
    # plt.title("Atomic Charge")
    # plt.plot(R0, TotRho)
    # plt.show()

    for itt in range(Nitt):
        print(f'{itt}) Preparing potential')
        
        # Compute Hartree potential
        UHartree = SolvePoisson(Z, R0, TotRho)
        Vxc = np.array([XC.Vx(rsi) + XC.Vc(rsi) for rsi in rs(TotRho)])
        nVeff = (UHartree - Z) / R0 + Vxc
        zeroMT = nVeff[-1]  # Muffin-Tin zero
        nVeff -= zeroMT
        print(f'   Muffin-Tin zero is {zeroMT}')

        Veff = nVeff

        # Solve for wave functions
        logDer, Psi_l, Psip_l = Wave(Z, Enu, R0, Veff)

        # Find and sort core states
        coreRho, coreE, coreZ, core_states = FindCoreStates(core, R0[::-1], Veff[::-1], Z)
        print(f'   coreZ= {coreZ}, coreE= {coreE}')

        # Main loop over k-points
        Ek, w0, w1, w2, wi = [], [], [], [], []
        for ik, k in enumerate(fcc.kp):
            tEk, tAr, tw0, tw1, tw2, twi = ComputeEigensystem(k, fcc.Km, Olap_I, Enu, logDer, RMuffinTin, fcc.Volume)
            Ek.append(tEk)
            w0.append(tw0)
            w1.append(tw1)
            w2.append(tw2)
            wi.append(twi)
        Ek = np.array(Ek)

        print("root chemical potential: ", rootChemicalPotential)
        print("mu_mm[0]", mu_mm[0])
        print("mu_mm[1]", mu_mm[1])
        # Determine new chemical potential
        mu = scipy.optimize.brentq(rootChemicalPotential, mu_mm[0], mu_mm[1], args=(Ek, fcc.wkp, Zval, beta))
        print(f'\033[92mNew chemical potential is {mu}\033[0m')

        # Compute density
        MTRho = ComputeMTDensity(mu, Ek, fcc.wkp, w0, w1, w2, Psi_l, Psip_l, beta)
        nTotRho = MTRho + coreRho
        
        sMTRho = scipy.integrate.simpson(MTRho * R0**2 * (4 * np.pi), x=R0)
        sIntRho = ComputeInterstitialCharge(mu, Ek, wi, fcc.wkp, beta)
        sCoreRho = scipy.integrate.simpson(coreRho * R0**2 * (4 * np.pi), x=R0)

        print(f'   Zval= {Zval} ~ {sMTRho + sIntRho}')
        print(f'   Weight in the MT sphere = {sMTRho}, in the interstitials = {sIntRho}, in core = {sCoreRho}')

        renorm = Z / (sMTRho + sIntRho + sCoreRho)
        print(f'   Total charge found= {sMTRho + sIntRho + sCoreRho}, should be {Z} -> renormalizing by {renorm}')
        nTotRho *= renorm

        # Check convergence
        DiffRho = scipy.integrate.simpson(np.abs(nTotRho - TotRho), x=R0)
        print(f'\033[94mElectron density difference= {DiffRho}\033[0m')
        if DiffRho < DRho:
            break
        
        TotRho = mixRho * nTotRho + (1 - mixRho) * TotRho

    # Plotting bands
    fcc.ChoosePointsInFBZ(Nkplot, type=1)
    
    Ek = []
    for ik, k in enumerate(fcc.kp):
        tEk, tAr, tw0, tw1, tw2, twi = ComputeEigensystem(k, fcc.Km, Olap_I, Enu, logDer, RMuffinTin, fcc.Volume)
        Ek.append(tEk)
    Ek = np.array(Ek)

    for i in range(Ek.shape[1]):
        if (np.max(Ek[:, i]) - mu > plotMM[0] and np.min(Ek[:, i]) - mu < plotMM[1]):
            plt.plot(Ek[:, i] - mu, 'k-', lw=2)

    plt.axhline(0, color='k', linestyle=':')  # Chemical potential line
    ax = plt.axis()

    xlabs = [p[1] for p in fcc.Points]
    labs = [p[0] for p in fcc.Points]
    plt.xticks(xlabs, labs)
    
    for ix, x in enumerate(xlabs):
        plt.axvline(x, color='k', linestyle=':')

    plt.axis([xlabs[0], xlabs[-1], ax[2], ax[3]])
    plt.show()

if __name__ == '__main__':
    main()
