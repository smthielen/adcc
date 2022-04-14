import adcc
import numpy as np
from adcc.misc import cached_member_function
from adcc.functions import einsum, direct_sum, zeros_like
import adcc.block as b
from diis import Diis
from pyscf import gto, scf

from geometries import geoms
from ucc3_itm import ucc3_intermediates

use_itm = True

def update_ucc3_t1(t1_in, t2_in, hf, df1):
    return (
        einsum('jaib,jb->ia', -1.0 * hf.ovov, t1_in)        
        + 0.5 * einsum('ijab,jb->ia', hf.oovv, t1_in)        
        + 0.5 * einsum('jacb,ijcb->ia', -1.0 * hf.ovvv, t2_in)        
        - 0.5 * einsum('kjib,jkba->ia', hf.ooov, t2_in)        
        - 0.5 * einsum('kila,jkbc,jlbc->ia', hf.ooov, t2_in, t2_in)        
        + 0.5 * einsum('icad,jkbd,jkbc->ia', hf.ovvv, t2_in, t2_in)        
        - einsum('klca,ijlb,jkbc->ia', t2_in, hf.ooov, t2_in)        
        + einsum('kicd,jdba,jkbc->ia', t2_in, hf.ovvv, t2_in)        
        - 0.25 * einsum('ilac,kjlb,jkbc->ia', t2_in, hf.ooov, t2_in)        
        + 0.25 * einsum('ikac,jcbd,jkbd->ia', t2_in, hf.ovvv, t2_in)        
        + 0.25 * einsum('jkca,icbd,jkbd->ia', t2_in, hf.ovvv, t2_in)        
        - 0.25 * einsum('ilcb,kjla,jkbc->ia', t2_in, hf.ooov, t2_in)        
        ) / df1

def update_ucc3_t2(t1_in, t2_in, hf, df2):
    return (
        hf.oovv        
        - 2.0 * einsum('jika,kb->ijab', hf.ooov, t1_in).antisymmetrise(2, 3)
        + 2.0 * einsum('icab,jc->ijab', hf.ovvv, t1_in).antisymmetrise(0, 1)
        + 0.5 * einsum('klij,klab->ijab', hf.oooo, t2_in)
        + 0.5 * einsum('abcd,ijcd->ijab', hf.vvvv, t2_in)
        + 4.0 * einsum('kaic,jkbc->ijab', -1.0 * hf.ovov, t2_in).antisymmetrise(2, 3).antisymmetrise(0, 1)
        + 4.0 * einsum('jlbd,klcd,ikac->ijab', t2_in, 1.0/3.0 * hf.oovv, t2_in).antisymmetrise(2, 3).antisymmetrise(0, 1)
        + einsum('klab,klcd,ijcd->ijab', t2_in, 1.0/6.0 * hf.oovv, t2_in)
        - 2.0 * einsum('klcb,klcd,ijad->ijab', t2_in, 1.0/3.0 * hf.oovv, t2_in).antisymmetrise(2, 3)
        - 2.0 * einsum('jkdc,klcd,ilab->ijab', t2_in, 1.0/3.0 * hf.oovv, t2_in).antisymmetrise(0, 1)
        + 4.0 * einsum('jkbc,ilad,klcd->ijab', t2_in, 1.0/3.0 * hf.oovv, t2_in).antisymmetrise(2, 3).antisymmetrise(0, 1)
        + einsum('klab,ijcd,klcd->ijab', t2_in, 1.0/12.0 * hf.oovv, t2_in)
        + einsum('ijcd,klab,klcd->ijab', t2_in, 1.0/12.0 * hf.oovv, t2_in)
        - 2.0 * einsum('klcb,ijad,klcd->ijab', t2_in, 1.0/6.0 * hf.oovv, t2_in).antisymmetrise(2, 3)
        - 2.0 * einsum('jkdc,ilab,klcd->ijab', t2_in, 1.0/6.0 * hf.oovv, t2_in).antisymmetrise(0, 1)
        - 2.0 * einsum('ilab,kjcd,klcd->ijab', t2_in, 1.0/6.0 * hf.oovv, t2_in).antisymmetrise(0, 1)
        - 2.0 * einsum('ijad,lkbc,klcd->ijab', t2_in, 1.0/6.0 * hf.oovv, t2_in).antisymmetrise(2, 3) 
        ) / df2

def update_ucc3_t1_itm(t1_in, t2_in, hf, df1, itm1_ooov, itm2_ovvv):
    return (
        einsum('jaib,jb->ia', -1.0 * hf.ovov, t1_in) 
        + 0.5 * einsum('ijab,jb->ia', hf.oovv, t1_in) 
        + einsum('kjib,jkba->ia', itm1_ooov, t2_in) 
        + einsum('jabc,ijcb->ia', itm2_ovvv, t2_in) 
        - 0.5 * einsum('kila,jkbc,jlbc->ia', hf.ooov, t2_in, t2_in) 
        + 0.5 * einsum('icad,jkbd,jkbc->ia', hf.ovvv, t2_in, t2_in) 
        - 0.25 * einsum('ilac,kjlb,jkbc->ia', t2_in, hf.ooov, t2_in) 
        + 0.25 * einsum('ikac,jcbd,jkbd->ia', t2_in, hf.ovvv, t2_in) 
        ) / df1

def update_ucc3_t2_itm(t1_in, t2_in, hf, df2, itm3_oooo, itm4_vvvv, itm5_ovov):
    return (        
        hf.oovv
        - einsum('jika,kb->ijab', hf.ooov, t1_in).antisymmetrise(2, 3)
        + einsum('icab,jc->ijab', hf.ovvv, t1_in).antisymmetrise(0, 1)
        + einsum('klij,klab->ijab', itm3_oooo, t2_in)
        + einsum('abcd,ijcd->ijab', itm4_vvvv, t2_in)
        #- asymm(i, j, contract(l, 1.0/3.0 * contract(k|c|d, i_oovv(k|l|c|d), t2(j|k|d|c))
        #        + 1.0/6.0 * contract(k|c|d, i_oovv(k|j|c|d), t2(k|l|c|d)),
        #        t2(i|l|a|b))
        #    + 1.0/6.0 * contract(l, contract(k|c|d, t2(k|l|c|d), t2(j|k|d|c)),
        #        i_oovv(i|l|a|b)))
        - 1.0/3.0 * einsum('klcd,jkcd,ilab->ijab', hf.oovv, t2_in, 
            t2_in).antisymmetrise(0, 1)
        - 1.0/6.0 * einsum('kjcd,klcd,ilab->ijab', hf.oovv, t2_in,
            t2_in).antisymmetrise(0, 1)
        - 1.0/6.0 * einsum('klcd,jkdc,ilab->ijab', t2_in, t2_in,
            hf.oovv).antisymmetrise(0, 1)
        #- asymm(a, b, contract(d, 1.0/3.0 * contract(k|l|c, i_oovv(k|l|c|d), t2(k|l|c|b))
        #            + 1.0/6.0 * contract(k|l|c, i_oovv(k|l|c|b), t2(k|l|c|d)),
        #            t2(i|j|a|d))
        #        + 1.0/6.0 * contract(d, contract(k|l|c, t2(k|l|c|d), t2(k|l|c|b)),
        #            i_oovv(i|j|a|d)))
        - 1.0/3.0 * einsum('klcd,klcb,ijad->ijab', hf.oovv, t2_in, 
            t2_in).antisymmetrise(2, 3)
        - 1.0/6.0 * einsum('klcb,klcd,ijad->ijab', hf.oovv, t2_in,
            t2_in).antisymmetrise(2, 3)
        - 1.0/6.0 * einsum('klcd,klcb,ijad->ijab', t2_in, t2_in,
            hf.oovv).antisymmetrise(2, 3)
        + einsum('kaic,jkbc->ijab', itm5_ovov, 
            t2_in).antisymmetrise(2, 3).antisymmetrise(0, 1)
        ) / df2

class LazyUcc3(adcc.LazyMp):
    def __init__(self, hf):
        super().__init__(hf)
        if self.has_core_occupied_space:
            raise NotImplementedError("UCC2 not implemented for CVS.")

    @cached_member_function
    def update_ampl(self, space):
        if space != b.oovv:
            raise NotImplementedError("UCC2 t-amplitudes only"
                                      " implemented for oovv block.")
        # TODO: hacked... move solver code out of routine
        conv_tol = 1e-6
        maxiter = 100
        hf = self.reference_state
        t2 = -1.0 * super().t2(space).evaluate()
        df2 = -1.0 * direct_sum("-i-j+a+b->ijab",
                                hf.foo.diagonal(), hf.foo.diagonal(),
                                hf.fvv.diagonal(), hf.fvv.diagonal())
        df2.evaluate()

        #ucc3 intermediates
        if use_itm:
            itm = ucc3_intermediates()
            itm1_ooov = itm.i1_ooov(hf, t2).evaluate()
            itm2_ovvv = itm.i2_ovvv(hf, t2).evaluate()
            itm3_oooo = itm.i3_oooo(hf, t2).evaluate()
            itm4_vvvv = itm.i4_vvvv(hf, t2).evaluate()
            itm5_ovov = itm.i5_ovov(hf, t2).evaluate()

        diis_handler = Diis()
        print("Niter", "|res|")
        for i in range(maxiter):
            #t1
            t1_old = t1
            if use_itm:
                t1_new = update_ucc3_t1_itm(t1, t2, hf, df1, 
                    itm1_ooov, itm2_ovvv).evaluate()
            else:
                t1_new = update_ucc3_t1(t1, t2, hf, df1).evaluate()
            res_t1 = t1_new - t1
            rnorm_t1 = np.sqrt(res_t1.dot(res_t1))
            #if rnorm_t1 > 1:
            #    print("rnorm_t1:" + str(rnorm_t1))
            diis_handler_t1.add_vectors(t1_new, res_t1)
            t1 = t1_new
            if len(diis_handler_t1.solutions) > 2 and rnorm_t1 <= 1:
                t1 = diis_handler_t1.get_optimal_linear_combination()
                diff_t1 = t1 - diis_handler_t1.solutions[-1]
                diff_t1.evaluate()
                rnorm_t1 = np.sqrt(diff_t1.dot(diff_t1))
            
            #t2
            if use_itm:
                t2_new = update_ucc3_t2_itm(t1_old, t2, hf, df2,
                    itm3_oooo, itm4_vvvv, itm5_ovov).evaluate()
            else:
                t2_new = update_ucc3_t2(t1_old, t2, hf, df2).evaluate()
            res_t2 = t2_new - t2
            rnorm_t2 = np.sqrt(res_t2.dot(res_t2))
            #if rnorm_t2 > 1:
            #    print("rnorm_t2:" + str(rnorm_t2))
            diis_handler_t2.add_vectors(t2_new, res_t2)
            t2 = t2_new
            if len(diis_handler_t2.solutions) > 2 and rnorm_t2 <= 1:
                t2 = diis_handler_t2.get_optimal_linear_combination()
                diff_t2 = t2 - diis_handler_t2.solutions[-1]
                diff_t2.evaluate()
                rnorm_t2 = np.sqrt(diff_t2.dot(diff_t2))
            
            #convergence requirement
            rnorm = np.sqrt(rnorm_t1*rnorm_t1 + rnorm_t2*rnorm_t2)
            print(i, rnorm)
            if rnorm < conv_tol:
                # switch sign for compatibility
                return -1.0 * t2
        raise ValueError("t2 amplitudes not converged.")
'''
    @cached_member_function
    def t2(self, space):
        if space != b.oovv:
            raise NotImplementedError("UCC2 t-amplitudes only"
                                      " implemented for oovv block.")

        # TODO: hacked... move solver code out of routine
        conv_tol = 1e-6
        maxiter = 100
        hf = self.reference_state
        t2 = -1.0 * super().t2(space).evaluate()
        df1 = -1.0 * mp.df(b.ov)
        df2 = -1.0 * direct_sum("-i-j+a+b->ijab",
                                hf.foo.diagonal(), hf.foo.diagonal(),
                                hf.fvv.diagonal(), hf.fvv.diagonal())
        df2.evaluate()

        #DIIS for t1 and t2
        diis_handler_t1 = Diis()
        diis_handler_t2 = Diis()
        print("Niter", "|res|")
        for i in range(maxiter):
            t2new = update_amplitude_ucc2(t2, hf, df2).evaluate()
            res = t2new - t2
            rnorm = np.sqrt(res.dot(res))
            diis_handler.add_vectors(t2new, res)
            t2 = t2new
            if len(diis_handler.solutions) > 2 and rnorm <= 1.0:
                t2 = diis_handler.get_optimal_linear_combination()
                diff = t2 - diis_handler.solutions[-1]
                diff.evaluate()
                rnorm = np.sqrt(diff.dot(diff))
            print(i, rnorm)
            if rnorm < conv_tol:
                # switch sign for compatibility
                return -1.0 * t2
        raise ValueError("t2 amplitudes not converged.")
'''
    @cached_member_function
    def td2(self, space):
        # td2 does not exist for UCC
        return self.t2(space).zeros_like()









#UCC3 energy
def ucc3_energy(t2, hf):
    return (
        #+ 0.25 * dot_product(i_oovv(i|j|a|b), t2(i|j|a|b))
        0.25 * hf.oovv.dot(t2)
        )       

#FIRST calculation
scfres = scf.RHF(geoms[0])
scfres.conv_tol = 1e-10
scfres.conv_tol_grad = 1e-8
scfres.kernel()
hf = adcc.ReferenceState(scfres)
mp = adcc.LazyMp(hf)

#prereqs
#t1 = -1.0 * (mp.df(b.ov).evaluate())
t1 = zeros_like(mp.df(b.ov)).evaluate() 
t2 = -1.0 * mp.t2(b.oovv).evaluate()
energies = np.empty(0, float)

#ucc3 intermediates
if use_itm:
    itm = ucc3_intermediates()
    itm1_ooov = itm.i1_ooov(hf, t2).evaluate()
    itm2_ovvv = itm.i2_ovvv(hf, t2).evaluate()
    itm3_oooo = itm.i3_oooo(hf, t2).evaluate()
    itm4_vvvv = itm.i4_vvvv(hf, t2).evaluate()
    itm5_ovov = itm.i5_ovov(hf, t2).evaluate()

#ITERATION over the rest
c = 0
for i in geoms:
    c += 1
    print("Geometry " + str(c))
    scfres = scf.RHF(i)
    scfres.conv_tol = 1e-10
    scfres.conv_tol_grad = 1e-8
    scfres.max_cycle = 100
    scfres.kernel()
    hf = adcc.ReferenceState(scfres)
    mp = adcc.LazyMp(hf)

    df1 = -1.0 * mp.df(b.ov)
    df2 = -1.0 * direct_sum("-i-j+a+b->ijab",
                            hf.foo.diagonal(), hf.foo.diagonal(),
                            hf.fvv.diagonal(), hf.fvv.diagonal()).evaluate()

    #DIIS for t1 and t2
    conv_tol = 1e-6
    maxiter = 500
    convd = False
    diis_handler_t1 = Diis()
    diis_handler_t2 = Diis()
    print("Niter", "|res|")
    for i in range(maxiter):
        #t1
        t1_old = t1
        if use_itm:
            t1_new = ucc3_t1_update_itm(t1, t2, hf, df1, itm1_ooov, itm2_ovvv).evaluate()
        else:
            t1_new = ucc3_t1_update(t1, t2, hf, df1).evaluate()
        res_t1 = t1_new - t1
        rnorm_t1 = np.sqrt(res_t1.dot(res_t1))
        if rnorm_t1 > 1:
            print("rnorm_t1:" + str(rnorm_t1))
        diis_handler_t1.add_vectors(t1_new, res_t1)
        t1 = t1_new
        if len(diis_handler_t1.solutions) > 2 and rnorm_t1 <= 1:
            t1 = diis_handler_t1.get_optimal_linear_combination()
            diff_t1 = t1 - diis_handler_t1.solutions[-1]
            diff_t1.evaluate()
            rnorm_t1 = np.sqrt(diff_t1.dot(diff_t1))
        #t2
        if use_itm:
            t2_new = ucc3_t2_update_itm(t1_old, t2, hf, df2, itm3_oooo, itm4_vvvv, itm5_ovov).evaluate()
        else:
            t2_new = ucc3_t2_update(t1_old, t2, hf, df2).evaluate()
        res_t2 = t2_new - t2
        rnorm_t2 = np.sqrt(res_t2.dot(res_t2))
        if rnorm_t2 > 1:
            print("rnorm_t2:" + str(rnorm_t2))
        diis_handler_t2.add_vectors(t2_new, res_t2)
        t2 = t2_new
        if len(diis_handler_t2.solutions) > 2 and rnorm_t2 <= 1:
            t2 = diis_handler_t2.get_optimal_linear_combination()
            diff_t2 = t2 - diis_handler_t2.solutions[-1]
            diff_t2.evaluate()
            rnorm_t2 = np.sqrt(diff_t2.dot(diff_t2))
        rnorm = np.sqrt(rnorm_t1*rnorm_t1 + rnorm_t2*rnorm_t2)
        print(i, rnorm)
        if rnorm < conv_tol:
            convd = True
            # switch sign for compatibility with LazyMp
            t2 = -1.0 * t2
            break
    #raise ValueError("t2 amplitudes not converged.")
       
    #t1 = mp.df(b.ov) 
    #t2 = -1.0 * mp.t2(b.oovv).evaluate()

    if convd == True:
        energy = hf.energy_scf + ucc3_energy(-1.0 * t2, hf)
        energies = np.append(energies, energy)
        print(energies)
    else:
        print("DIIS for t amplitudes not converged after " + str(maxiter) + " iterations!!")
        energies = np. append(energies, 0.0)
        #t1 = -1.0 * mp.df(b.ov) 
        t1 = zeros_like(mp.df(b.ov)).evaluate() 
        t2 = -1.0 * mp.t2(b.oovv).evaluate()
    #print("MP2 energy = ", mp.energy(level=2))
