# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:47:45 2023

@author: aiden
"""

import numpy.random as rng
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg,sparse,optimize,signal
from scipy.fft import fft,ifft
import time as tim
from mpl_toolkits import mplot3d
from scipy.signal import argrelextrema



def state(n,L):   #Function that takes a state :n> and converts it to the equivelent binary number
    binary = str(bin(n)[2:])
    n=[]
    if len(binary) < L:
        for digit in range(L-len(binary)):
            n.append(0)
    for digit in binary:
        n.append(int(digit))
    return n

def bit_flip(n,i): #flips the i'th element in a state n
    new = list(n)[:]
    if new[i] == 0:
        new[i]=1
    else:
        new[i]=0
    return np.array(new)

def undostate(n): #Converts a binary state of the string back into its single numberical equivelent
    return int(''.join([str(x) for x in n]),2)

def spinbasis(N,cond="PBC"): #Generates the spin basis
    binary_basis=list(np.arange(2**N))
    final_basis=[]
    for i in binary_basis:
        final_basis.append(state(i, N))
    basismatrix=np.array(final_basis)
    return final_basis,binary_basis,basismatrix

def spinbasis_Znum(N,z,cond="PBC"): #Generates the spin basis
    binary_basis=list(np.arange(2**N))
    final_basis=[]
    for i in binary_basis:
        final_basis.append(state(i, N))
    basismatrix=np.array(final_basis)
    Znum=np.sum((basismatrix-0.5)*2,axis=1)/N
    mask=np.where(Znum==z)[0]
    return list(np.array(final_basis)[mask]),list(np.array(binary_basis)[mask]),basismatrix[mask,:]

def magnetisation_vec(N,basis): #Computes the vector for themagnetisation observable
    basis_matrix=basis[-1]
    basis_matrix=(basis_matrix-0.5)*2
    Z=np.sum(basis_matrix,axis=1)/N
    return Z

def chiralact_A(n,u=0,v=1,E=0,cond="OBC"):#Finds the result of the chiral Hamiltonian acting on a state n.
    output=[]
    binary_output=[]
    factor=[]
    L=len(n)
    chain=L
    if type(v)!=list:
        v=np.tile(v,chain)

    #Coupling between two subsystems with coupling E
    new=bit_flip(n,(chain-1)//2 -1)
    new=bit_flip(new,(chain-1)//2)
    element=(-1*E/2)*(1/4)
    output.append(new[:])
    binary_output.append(undostate(new))
    factor.append(element)
        
    new=bit_flip(n,(chain-1)//2 -1)
    new=bit_flip(new,((chain-1)//2))
    element=(-1*E/2)*(1j)*((-1)**(n[((chain-1)//2 -1)]+1))*(1j)*((-1)**(n[((chain-1)//2)]+1))*(1/4)
    output.append(new[:])
    binary_output.append(undostate(new))
    factor.append(element)
    

    #first half being +H
    for i in np.arange(0,(chain-1)//2 -2):
        new=bit_flip(n,i)
        new=bit_flip(new,(i+1)%L)
        element=(1*v[i]/4)*(n[(i+2)%L]-0.5)*2*(1j)*((-1)**(n[(i+1)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,i)
        new=bit_flip(new,(i+2)%L)
        element=(-1*v[i]/4)*(n[(i+1)%L]-0.5)*2*(1j)*((-1)**(n[(i+2)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,i)
        new=bit_flip(new,(i+1)%L)
        element=(-1*v[i]/4)*(n[(i+2)%L]-0.5)*2*(1j)*((-1)**(n[(i)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,i)
        new=bit_flip(new,(i+2)%L)
        element=(1*v[i]/4)*(n[(i+1)%L]-0.5)*2*(1j)*((-1)**(n[(i)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,(i+1)%L)
        new=bit_flip(new,(i+2)%L)
        element=(1*v[i]/4)*(n[(i)%L]-0.5)*2*(1j)*((-1)**(n[(i+2)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,(i+1)%L)
        new=bit_flip(new,(i+2)%L)
        element=(-1*v[i]/4)*(n[(i)%L]-0.5)*2*(1j)*((-1)**(n[(i+1)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        #non-chiral term - XY term
        new=bit_flip(n,i)
        new=bit_flip(new,(i+1)%L)
        element=(-1*u/2)*(1/4)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,i)
        new=bit_flip(new,(i+1)%L)
        element=(-1*u/2)*(1j)*((-1)**(n[(i)%L]+1))*(1j)*((-1)**(n[(i+1)%L]+1))*(1/4)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
    if cond=="OBC" or cond=="obc":
        new=bit_flip(n,(chain-1)//2 -2)
        new=bit_flip(new,(chain-1)//2-1)
        element=(-1*u/2)*(1/4)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
            
        new=bit_flip(n,(chain-1)//2 -2)
        new=bit_flip(new,((chain-1)//2)-1)
        element=(-1*u/2)*(1j)*((-1)**(n[((chain-1)//2 -2)]+1))*(1j)*((-1)**(n[((chain-1)//2-1)]+1))*(1/4)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
    return output,factor,binary_output


def chiralact_B(n,u=0,v=1,E=0,cond="OBC"):#Finds the result of the chiral Hamiltonian acting on a state n.
    output=[]
    binary_output=[]
    factor=[]
    L=len(n)
    chain=L
    if type(v)!=list:
        v=np.tile(v,chain)
    #Adding a stray defect on a single site n_J term with factor E,
    #E=10e-4
    J=0
    #defectstate=bit_flip(n,J)
    #print("stray field on site {}".format(J))
    binary_output.append(undostate(n))
    output.append(n[:])
    factor.append(0)#*(n[J]-0.5)*2)
    
    #second half being -H -- leaves last site to be I
    for i in np.arange((chain-1)//2,chain-3):
        new=bit_flip(n,i)
        new=bit_flip(new,(i+1)%L)
        element=(-1*v[i]/4)*(n[(i+2)%L]-0.5)*2*(1j)*((-1)**(n[(i+1)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,i)
        new=bit_flip(new,(i+2)%L)
        element=(1*v[i]/4)*(n[(i+1)%L]-0.5)*2*(1j)*((-1)**(n[(i+2)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,i)
        new=bit_flip(new,(i+1)%L)
        element=(1*v[i]/4)*(n[(i+2)%L]-0.5)*2*(1j)*((-1)**(n[(i)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,i)
        new=bit_flip(new,(i+2)%L)
        element=(-1*v[i]/4)*(n[(i+1)%L]-0.5)*2*(1j)*((-1)**(n[(i)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,(i+1)%L)
        new=bit_flip(new,(i+2)%L)
        element=(-1*v[i]/4)*(n[(i)%L]-0.5)*2*(1j)*((-1)**(n[(i+2)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,(i+1)%L)
        new=bit_flip(new,(i+2)%L)
        element=(1*v[i]/4)*(n[(i)%L]-0.5)*2*(1j)*((-1)**(n[(i+1)%L]+1))*(1/8)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        #non-chiral term - XY term
        new=bit_flip(n,i)
        new=bit_flip(new,(i+1)%L)
        element=(1*u/2)*(1/4)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
        new=bit_flip(n,i)
        new=bit_flip(new,(i+1)%L)
        element=(1*u/2)*(1j)*((-1)**(n[(i)%L]+1))*(1j)*((-1)**(n[(i+1)%L]+1))*(1/4)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
        
    if cond=="OBC" or cond=="obc":
        new=bit_flip(n,L-3)
        new=bit_flip(new,(L-2))
        element=(1*u/2)*(1/4)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
            
        new=bit_flip(n,L-3)
        new=bit_flip(new,(L-2))
        element=(1*u/2)*(1j)*((-1)**(n[(L-3)]+1))*(1j)*((-1)**(n[(L-2)]+1))*(1/4)
        output.append(new[:])
        binary_output.append(undostate(new))
        factor.append(element)
    return output,factor,binary_output


def hamiltonian(N,basis,u=0,v=1,E=0,cond="OBC"): #Computes the PXP hamiltonian on the Rydberg states of system size N
    computebasis = basis
    states = computebasis[0]
    binarybasis = computebasis[1]
    L=len(binarybasis)
    H_A=sparse.csc_matrix(np.zeros((L,L))+0j)
   # H=np.zeros((L,L))+0j
    #pxp=sparse.dok_array((L,L))
    for b,n in enumerate(states):
        output = chiralact_A(n[:],u,v,E,cond=cond)
        for place,m in enumerate(output[0]):
            if undostate(m[:]) in binarybasis:
                a = binarybasis.index(undostate(m[:]))
                H_A[a,b]=H_A[a,b] + (output[1][place])
    H_B=sparse.csc_matrix(np.zeros((L,L))+0j)
   # H=np.zeros((L,L))+0j
    #pxp=sparse.dok_array((L,L))
    for b,n in enumerate(states):
        output = chiralact_B(n[:],u,v,E,cond=cond)
        for place,m in enumerate(output[0]):
            if undostate(m[:]) in binarybasis:
                a = binarybasis.index(undostate(m[:]))
                H_B[a,b]=H_B[a,b] + (output[1][place])            
    I=sparse.identity(len(basis[1]))            
    return H_A + H_B# + 0.1*I

def anyreduced(vec,N,number): #computes the reduced density matrix at partition number on a vector
    vector=np.reshape(vec,(2**int(N-number),2**number))
    #u,s,v = linalg.svd(vector)
    rho=np.conjugate(np.transpose(vector))@vector
    return rho/np.trace(rho)


def singlex_i(N,basis,i): #Computes the operator x_i
    basis_matrix=basis[-1]
    basis_convert=np.flip(2**np.arange(N))
    mask=np.tile(True,basis_matrix.shape[0])
    acted_matrix=np.copy(basis_matrix)
    acted_matrix[:,i]=np.where(acted_matrix[:,i]==1,0,1)
    acted_matrix_binary=np.sum(basis_convert[np.newaxis,:]*acted_matrix,axis=1)
    old_location=np.where(mask==True)[0]
    new_location=[]
    for j in acted_matrix_binary[np.where(mask==True)[0]]:
        new_location.append(basis[1].index(j))
    matrix=np.zeros([len(basis[1]),len(basis[1])])+0j
    matrix[new_location,old_location]=1
    return matrix
    
    
def effectivetemperature(staten,energies,Ham,basis): #Takes a state H and uses the maths of statistical mechanics to calculate its thermising canonical ensemble temperature
        def energyexpect(Temp,args):
            energies,thermval=args
            beta=Temp
            Z=0
            val=0
            for E in energies:
                if beta>0:
                    Z=Z+np.exp(-1*beta*(E+50))
                    val=val+(E*np.exp(-1*beta*(E+50)))
                elif beta<=0:
                    Z=Z+np.exp(-1*beta*(E-100))
                    val=val+(E*np.exp(-1*beta*(E-100)))
            if val!=0:
                val=val/Z
            return np.abs(val-thermval)
        value=np.conjugate(np.transpose(staten))@Ham@staten
        args=[energies,value]
        optimise=optimize.basinhopping(energyexpect,[-0.1],niter=15,minimizer_kwargs={"args":args,"method":"L-BFGS-B"})#,"bounds":[(0,np.inf)]})
        if optimise.fun>0.1:
            print("There's a problem finding effective temp",optimise.fun)
        print(optimise.success,optimise.fun)
        return optimise.x
        
def OTOCS_singlev_diftemp(N,u,basis,v,betas,t=(0,1,0.01)): #Computes the otoc over time for different Betas and single v and returns the list of OTOC data sets
    Cts_normed_vs=[]
    ham=hamiltonian(N,basis,u,v,cond="OBC")
    ener,eigs=np.linalg.eigh(ham.toarray())
    size=ham.shape[0]
    J=(N//2)-2
    x_j=singlex_i(N,basis,J) 
    x_i=singlex_i(N, basis, J+2)
    #J=(N//2)
    #n_j=np.diag(basis[-1][:,J])
    t0,tm,dt=t
    #e_dt=sparse.linalg.expm(-1j*ham*dt)
    #C_T=[]
    e_dt=sparse.linalg.expm(-1j*ham*dt)
    e_t=np.identity(size)+0j

    for beta in betas:
        e_t=np.identity(size)+0j
        print(v,beta)
        if beta<20:
            rho=sparse.linalg.expm(-(beta/4)*ham.toarray())
            rho=rho/np.trace(rho)
            normalization=1/(np.trace(rho@rho@rho@rho))
        elif beta>=100:
            rho=sparse.linalg.expm(-(beta/4)*ham.toarray()/5)
            rho=rho/np.trace(rho)
            rho=np.linalg.matrix_power(rho,5)
            rho=rho/np.trace(rho)
            normalization=1/(np.trace(rho@rho@rho@rho))
        elif beta>=20 and beta<100:
            rho=sparse.linalg.expm(-(beta/4)*ham.toarray()/2)
            rho=rho/np.trace(rho)
            rho=np.linalg.matrix_power(rho,2)
            rho=rho/np.trace(rho)
            normalization=1/(np.trace(rho@rho@rho@rho))
        #normalization=1/(np.trace(rho))
        #rho_square=linalg.sqrtm(linalg.sqrtm(rho*normalization))
        C_T=[]
        for time in np.arange(*t):
            #e_t=sparse.linalg.expm(-1j*ham*time)
            x_t=np.transpose(np.conjugate(e_t))@x_i@e_t
            product=normalization*x_t@rho@x_j@rho@x_t@rho@x_j@rho
            C_T.append(np.trace(product).real)
            e_t=e_t@e_dt
        C_T_norm=np.array(C_T)/C_T[0]
        Cts_normed_vs.append(C_T_norm)
    return Cts_normed_vs



def lyaponov_fit(dataset,t=(0,1,0.025)): #Given a data set, computes the Lyaponov exponent up by fitting an exponential to the data set (up to a maxima in the data) e.g. reduces data down to a maxima
    Time=np.arange(*t)
    from scipy.signal import argrelextrema,find_peaks
    maxima=find_peaks(dataset,prominence=0.01)[0]
    time_red=Time
    if maxima.shape[0]>1:
        dataset=dataset[:maxima[0]]
        time_red=Time[:maxima[0]]
    def fit(x,a,b):
        return 1*(np.exp(a*x)-1)
    ps,cov=optimize.curve_fit(fit,time_red,dataset)
    return ps[0]


def hyperfunc(N,lam,t): #For a given exponent and system size, computes the confluent hypergeometric function from the SYK paper in a given t
    import scipy
    z=np.exp(lam*t)/N
    F=scipy.special.hyperu(0.5,1,1/z)
    F=F/(z**0.5)
    return F/F[0]


def entangle_qubits_10(vec,n,m,basis): #For a given vector, creates an entangled pair on sites n and m in the bell basis of the type 1/root2 * (01 - 10)_nm
    newvecn=np.zeros(len(vec))+0j
    newvecm=np.zeros(len(vec))+0j
    newbasismatrix=basis[-1]
    newbasismatrixn=np.copy(newbasismatrix)
    newbasismatrixm=np.copy(newbasismatrix)
    
    newbasismatrixn[:,n]=newbasismatrix[:,n]+1
    maskn=np.where(newbasismatrixn[:,n]!=2)
    newbasis=np.sum(np.flip(2**np.arange(0,newbasismatrix.shape[1]))[np.newaxis,:]*newbasismatrixn,axis=1)
    newbasis=newbasis[maskn]
    newvecn[newbasis]=vec[maskn]
    
    newbasismatrixm[:,m]=newbasismatrix[:,m]+1
    maskm=np.where(newbasismatrixm[:,m]!=2)
    newbasis=np.sum(np.flip(2**np.arange(0,newbasismatrix.shape[1]))[np.newaxis,:]*newbasismatrixm,axis=1)
    newbasis=newbasis[maskm]
    newvecm[newbasis]=vec[maskm]

    newvec=(1/np.sqrt(2)) * (newvecn - newvecm)
    return newvec

def entangle_qubits_11(vec,n,m,basis): #For a given vector, creates an entangled pair on sites n and m in the bell basis of the type 1/root2 * (01 - 10)_nm
    newvecn=np.zeros(len(vec))+0j
    newbasismatrix=basis[-1]
    newbasismatrixn=np.copy(newbasismatrix)
    
    newbasismatrixn[:,n]=newbasismatrix[:,n]+1
    newbasismatrixn[:,m]=newbasismatrix[:,m]+1
    maskn=np.where(newbasismatrixn[:,n]!=2,True,False)
    maskm=np.where(newbasismatrixn[:,m]!=2,True,False)
    mask=maskn*maskm
    mask=np.where(mask==True)    
    newbasis=np.sum(np.flip(2**np.arange(0,newbasismatrix.shape[1]))[np.newaxis,:]*newbasismatrixn,axis=1)
    newbasis=newbasis[mask]
    newvecn[newbasis]=vec[mask]
    
    
    newvec=(1/np.sqrt(2)) * (vec + newvecn)
    return newvec

def projective_measure_10(vec,n,m,basis): #For a vec, performs a projective measure on the state vec | A><A | vec where A is the bell pair 1/root2 * (01 - 10)_nm
    #for first bell pair half
    basismatrix=basis[-1]
    P_n=basismatrix[:,n]
    P_m=basismatrix[:,m]
    maskn=np.where(P_n==0,True,False)
    maskm=np.where(P_m==1,True,False)
    mask=maskn*maskm
    firstvec=mask*vec
    
    P_n=basismatrix[:,n]
    P_m=basismatrix[:,m]
    maskn=np.where(P_n==1,True,False)
    maskm=np.where(P_m==0,True,False)
    mask=maskn*maskm
    secondvec=mask*vec

    newvec=(1/np.sqrt(2))*(firstvec - secondvec)
    newvec=newvec/np.sqrt(np.conjugate(np.transpose(newvec))@newvec)
    return newvec

def projective_measure_10_notnormal(vec,n,m,basis): #For a vec, performs a projective measure on the state vec | A><A | vec where A is the bell pair 1/root2 * (01 - 10)_nm
    #for first bell pair half
    basismatrix=basis[-1]
    P_n=basismatrix[:,n]
    P_m=basismatrix[:,m]
    maskn=np.where(P_n==0,True,False)
    maskm=np.where(P_m==1,True,False)
    mask=maskn*maskm
    firstvec=mask*vec
    
    P_n=basismatrix[:,n]
    P_m=basismatrix[:,m]
    maskn=np.where(P_n==1,True,False)
    maskm=np.where(P_m==0,True,False)
    mask=maskn*maskm
    secondvec=mask*vec

    newvec=(firstvec + secondvec)
    newvec=newvec/np.sqrt(np.conjugate(np.transpose(newvec))@newvec)
    return newvec    


def projective_measure_11(vec,n,m,basis): #For a vec, performs a projective measure on the state vec | A><A | vec where A is the bell pair 1/root2 * (01 - 10)_nm
    #for first bell pair half
    basismatrix=basis[-1]
    P_n=basismatrix[:,n]
    P_m=basismatrix[:,m]
    maskn=np.where(P_n==0,True,False)
    maskm=np.where(P_m==0,True,False)
    mask=maskn*maskm
    firstvec=mask*vec
    
    P_n=basismatrix[:,n]
    P_m=basismatrix[:,m]
    maskn=np.where(P_n==1,True,False)
    maskm=np.where(P_m==1,True,False)
    mask=maskn*maskm
    secondvec=mask*vec

    newvec=(1/np.sqrt(2))*(firstvec + secondvec)
    newvec=newvec/np.sqrt(np.conjugate(np.transpose(newvec))@newvec)
    return newvec

def define_A_state(vec,a,basis): #For a given vec (polarised) constructs the state a|1> + sqrt(|1-a^2|)|0> on the first qubit. Assumes first qubit is |0> 
    newvec=np.zeros(len(vec))+0j
    newbasismatrix=np.copy(basis[-1])
    
    newbasismatrix[:,0]=newbasismatrix[:,0]+1
    mask=np.where(newbasismatrix[:,0]!=2)
    newbasis=np.sum(np.flip(2**np.arange(0,newbasismatrix.shape[1]))[np.newaxis,:]*newbasismatrix,axis=1)
    newbasis=newbasis[mask]
    newvec[newbasis]=vec[mask]
    
    finalvec = np.sqrt(1-np.abs(a)**2)*vec + a*newvec
    return finalvec        
        
def measure_lastcubit(vec,basis): #for a given vec, measures the last cubit and returns a and b for a|1> + b|0>
    basis_cut=basis[-1][:,-1]
    zeroplace=np.where(basis_cut==0)
    oneplace=np.where(basis_cut==1)
    b=np.sum(np.abs(vec[zeroplace])**2)
    a=np.sum(np.abs(vec[oneplace])**2)
    return [a,b]

def fullbasis_toparticlebasis(vec,basis,basisZ): #Converts a full basis vector to a reduced basis vector (or vise versa depending on vector size)
    vecdim=len(vec)
    if vecdim==len(basis[1]):
        newvec=vec[basisZ[1]]
        newvec=newvec/np.sqrt(np.conjugate(np.transpose(newvec))@newvec)
    if vecdim==len(basisZ[1]):
        newvec=np.zeros(len(basis[1]))+0j
        newvec[basisZ[1]]=vec
        newvec=newvec/np.sqrt(np.conjugate(np.transpose(newvec))@newvec)
    return newvec


N=9
found_As_ps=[]
A_overtime_vs_ps=[]
probabilities_vs_ps=[]
for k in np.arange(0,(N+1)//2): #Loops over possible number of measurements
    found_as=[]
    A_overtime_vs=[]
    probabilities_vs=[]
    Vs=np.linspace(0,20,21) 
    for v in Vs:
        u=1
        E=1 #Defines an XY coupling between the two chain halves
        time=2 #Evolution time
        numprojectivemeasurements=k
    
        basis=spinbasis(N)#_Znum(N, (N-1)//2 + 1)
        basisZ=spinbasis_Znum(N, 1/N)
        cutbasis=basis[-1][:,:-1]
        ham=hamiltonian(N, basisZ,u,float(v),E=E,cond="OBC")
        ener=np.linalg.eigvalsh(ham.todense())
        
        initialstate=np.zeros(len(basis[1]))+0j
        initialstate[0]=1 #construct initial polarised
    
        a=1 #Selects the initial state to be teleportedon the first qubit. a is the coefficient of |1> on the first site
        initialstate=define_A_state(initialstate, a, basis) #constructs the teleported state on the first qubit
    
        numentangledpairs = (N-1)//2 #defined indices of the entangled pairs
        pairindices=[]
        initial=1
        final=N-3
        for i in np.arange(numentangledpairs-1):
            pairindices.append([initial+i,final-i])
        pairindices.append([N-2,N-1])
    
        #construct entangled state
        for i in pairindices:
            initialstate=entangle_qubits_10(initialstate, i[0], i[1], basis)
    
        initialstate=initialstate/np.sqrt(np.conjugate(np.transpose(initialstate))@initialstate)
        #evolution step
        initialstate=fullbasis_toparticlebasis(initialstate,basis,basisZ)
        evolved=sparse.linalg.expm_multiply(-1j*ham,initialstate,0,time,101).T

        pairindices[-1]=[0,-2]
        pairindices=pairindices[:-1][::-1] + [pairindices[-1]]
    
        pairindices=list(np.flip(pairindices[:-1])) + [pairindices[-1]]
        
        A_overtime=[]
        for i in np.arange(101):
            vec=evolved[:,i]
            finalvec=fullbasis_toparticlebasis(vec,basis,basisZ)
            
            firsthalf=np.sum(cutbasis[:,:(N-1)//2],axis=1)
            secondhalf=np.sum(cutbasis[:,(N-1)//2:],axis=1)
            lastsite=basis[-1][:,-1]
                       
          
        
            projected_vec=np.copy(finalvec)
        
        
         
            #projective measurements
            for i in np.arange(numprojectivemeasurements):
                projected_vec=projective_measure_10(projected_vec, pairindices[i][0], pairindices[i][1], basis)
            
            
            projected_vec=projected_vec/np.sqrt(np.conjugate(np.transpose(projected_vec))@projected_vec)
            

            
            
            #measure last qubit
            found_a,found_b=measure_lastcubit(projected_vec, basis)
            A_overtime.append(found_a)
            
            temp=[]
            for bit in np.arange(1,5):
                densitymatrix=anyreduced(projected_vec, N, bit)
                size=densitymatrix.shape[0]
                likeness=np.trace(linalg.sqrtm(densitymatrix))**2
                likeness=likeness/size
                temp.append(likeness)
            firstpart=basis[-1][:,0]
            
            projected_vec=np.copy(finalvec)
            #projective measurements
            for i in np.arange(numprojectivemeasurements):
                projected_vec=projective_measure_10_notnormal(projected_vec, pairindices[i][0], pairindices[i][1], basis)
            
            
            
            
        A_overtime_vs.append(A_overtime)

        
    found_As_ps.append(found_as)
    A_overtime_vs_ps.append(A_overtime_vs) #Stores fidelity results as an embedded list. First index is the number of measurements. Second index is the value of v
  





