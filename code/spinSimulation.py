import numpy as np
import scipy.linalg as spla

def mykron(*args):
    '''Returns the Kroneker product of all matrices passed as args
    Adapted from MATLAB script written by Evan Fortunato, April 29th 2000
    '''
    if len(args) < 1:
        print('Please enter at least one product operator!')
        raise
    elif len(args) == 1:
        return(args[0])
    else:
        product = args[0]
        for i in args[1:]:
            product = np.kron(product, i)
        return(product)

##### Global variables #####

z = 0.5  * np.array([[1,0],[0,-1]],  dtype='complex128')
x = 0.5  * np.array([[0,1],[1,0]],   dtype='complex128')
y = 0.5j * np.array([[0,1],[-1,0]], dtype='complex128')

ep = np.array([[1,0],[0,0]], dtype='complex128')
em = np.array([[0,0],[0,1]], dtype='complex128')

p = np.array([[0,1],[0,0]], dtype='complex128')
m = np.array([[0,0],[1,0]], dtype='complex128')

def getTotalSpin(N, dim):
    '''Define the X, Y, and Z total spin observables
    '''
    X = np.zeros((dim, dim), dtype='complex128')
    Y = np.zeros((dim, dim), dtype='complex128')
    Z = np.zeros((dim, dim), dtype='complex128')
    for i in range(N):
        X += mykron(np.eye(2**i), x, np.eye(2**(N-i-1)))
        Y += mykron(np.eye(2**i), y, np.eye(2**(N-i-1)))
        Z += mykron(np.eye(2**i), z, np.eye(2**(N-i-1)))
    return(X,Y,Z)

def getTotalZSpin(N, dim):
    Z = np.zeros((dim, dim), dtype='complex128')
    for i in range(N):
        Z += mykron(np.eye(2**i), z, np.eye(2**(N-i-1)))
    return Z

def getAngMom(theta, phi, N, dim):
    '''Returns a spin angular momentum operator along
    an arbitrary axis specified by theta, phi
    '''
    j = np.cos(theta) * z + np.sin(theta)*np.cos(phi) * x + \
        np.sin(theta)*np.sin(phi) * y
    J = np.zeros((dim, dim), dtype='complex128')
    for i in range(N):
        J += mykron(np.eye(2**i), j, np.eye(2**(N-i-1)))
    return J

def getRandDip(N):
    a = np.abs(np.random.normal(size=(N,N)))
    a = np.triu(a) + np.triu(a).T
    return(a)

def getHdip(N, dim, x, y, z, a):
    '''Get the dipolar Hamiltonian term, which includes
    spin-spin interactions with a specified dipolar coupling strength.
    The Hamiltonian is a sum over terms $a_{i,j} I_z^{(i)}I_z^{(j)}$
    '''
    Hdip = np.zeros((dim, dim), dtype='complex128')
    for i in range(N):
        for j in range(i+1, N):
            # TODO fix bug in here with floats/ints
            Hdip += a[i,j] * \
            (2*mykron(np.eye(2**i), z, np.eye(2**(j-i-1)), z, np.eye(2**(N-j-1))) - \
            mykron(np.eye(2**i), x, np.eye(2**(j-i-1)), x, np.eye(2**(N-j-1))) - \
            mykron(np.eye(2**i), y, np.eye(2**(j-i-1)), y, np.eye(2**(N-j-1))))
    return(Hdip)

def getHint(Hdip, coupling, Z, Delta):
    '''Get the total internal Hamiltonian for a NMR system
    currently not including an offset term (only CS and dipolar terms)
    '''
    return(coupling * Hdip + Delta * Z)

def getAllH(N, dim, coupling, delta):
    '''Get Hdip and Hint with random dipolar coupling strengths
    
    Arguments:
        N:
        dim:
        coupling: Coupling strength (to weight the dipolar coupling matrix).
        delta: Chemical shift strength (for identical spins).
    '''
    a = getRandDip(N) # random dipolar coupling strengths
    Z = getTotalZSpin(N, dim)

    Hdip = getHdip(N, dim, x, y, z, a)
    Hint = getHint(Hdip, coupling, Z, delta)
    return Hdip, Hint

def getHWHH0(X, Y, Z, Delta):
    return(Delta/3 * (X+Y+Z))

def getPropagator(H, t):
    '''Get the propagator from a time-independent Hamiltonian
    evolved for time t
    '''
    return(spla.expm(-1j*H*t))

def getUWHH(Hint, delay, pulse, X, Y):
    Utau   = getPropagator(Hint, delay)
    Ux     = getPropagator(Hint + np.pi/2/pulse*X, pulse)
    Uy     = getPropagator(Hint + np.pi/2/pulse*Y, pulse)
    Uxbar  = getPropagator(Hint - np.pi/2/pulse*X, pulse)
    Uybar  = getPropagator(Hint - np.pi/2/pulse*Y, pulse)

    return Utau @ Ux @ Utau @ Uybar @ Utau @ Utau @ \
            Uy @ Utau @ Uxbar @ Utau

def fidelity(Utarget, Uexp):
    '''Returns the trace of U_target' * U_exp, scaled
    by the dimension of the system to return a value
    between 0 and 1. Fidelity of 1 means the unitary
    operators are equal.
    '''
    if np.shape(Utarget) != np.shape(Uexp):
        print('Utarget and Uexp are not the same shape!')
        raise
    else:
        dim = np.size(Utarget, 0)
        return np.clip(abs(np.trace(Utarget.T.conj() @ Uexp) / dim), 0, 1)

def metric1(Utarget, Uexp):
    '''Returns the Frobenius norm of the difference of
    the two unitary operators
    '''
    if np.shape(Utarget) != np.shape(Uexp):
        print('Utarget and Uexp are not the same shape!')
        raise
    else:
        return(np.linalg.norm(Utarget - Uexp))
    

def metric2(Utarget, Uexp):
    '''Other possible metric definition using trace
    '''
    if np.shape(Utarget) != np.shape(Uexp):
        print('Utarget and Uexp are not the same shape!')
        raise
    else:
        dim = np.size(Utarget, 0)
        return(abs(np.trace(Utarget.T.conj() @ Uexp - \
            np.eye(dim)) / dim))
