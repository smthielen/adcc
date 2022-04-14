#
# LazyUcc3 based on LazyMp
#
# This class includes the calculation of t1 and t2 amplitudes
# using the DIIS algorithm, the UCC3 gs energy and densities
#
# SMT
#

import adcc
import numpy as np
from adcc.misc import cached_member_function, cached_property
from adcc.timings import timed_member_call
from adcc.OneParticleOperator import OneParticleOperator
from adcc.AmplitudeVector import AmplitudeVector
from adcc.functions import einsum, direct_sum, zeros_like, evaluate
import adcc.block as b
from diis import Diis
from pyscf import gto, scf

from quccsd_ampl import update_quccsd_t1, update_quccsd_t2
from quccsd_ene import quccsd_energy

'''
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
'''

class LazyUcc3(adcc.LazyMp):
    def __init__(self, hf):
        super().__init__(hf)
        if self.has_core_occupied_space:
            raise NotImplementedError("UCC3 not implemented for CVS.")

    def __getattr__(self, attr):
        # Shortcut some quantities, which are needed most often
        if attr.startswith("t2") and len(attr) == 4:  # t2oo, t2oc, t2cc
            xxvv = b.__getattr__(attr[2:4] + "vv")
            return self.t2(xxvv)
        #TODO: is this how t1 should be initialized?
        if attr.startswith("t1") and len(attr) == 4:
            return self.t1(b.ov)
        else:
            raise AttributeError

    @cached_property
    def t_amplitudes(self):
        if self.has_core_occupied_space:
            raise NotImplementedError("UCC3 not implemented for CVS.")
        #if space != b.oovv:
        #    raise NotImplementedError("UCC3 t-amplitudes only"
        #                              " implemented for oovv block.")
        # TODO: hacked... move solver code out of routine
        conv_tol = 1e-6
        maxiter = 100
        hf = self.reference_state
        # TODO: do not setup another mp object here
        df_ov = super().df(b.ov).evaluate()
        t1 = zeros_like(df_ov) 
        #t2 = -1.0 * super().t2(space).evaluate()
        t2 = -1.0 * super().t2(b.oovv).evaluate()
        df1 = -1.0 * df_ov
        # TODO: do we really need df2? use df1 instead
        df2 = -1.0 * direct_sum("-i-j+a+b->ijab",
                                hf.foo.diagonal(), hf.foo.diagonal(),
                                hf.fvv.diagonal(), hf.fvv.diagonal())
        df2.evaluate()

        diis_handler_t1 = Diis()
        diis_handler_t2 = Diis()
        print("Niter", "|res|")
        for i in range(maxiter):
            #t1
            t1_old = t1
            t1_new = update_ucc3_t1(t1, t2, hf, df1).evaluate()
            res_t1 = t1_new - t1
            rnorm_t1 = np.sqrt(res_t1.dot(res_t1))
            diis_handler_t1.add_vectors(t1_new, res_t1)
            t1 = t1_new
            if len(diis_handler_t1.solutions) > 2 and rnorm_t1 <= 1:
                t1 = diis_handler_t1.get_optimal_linear_combination()
                diff_t1 = t1 - diis_handler_t1.solutions[-1]
                diff_t1.evaluate()
                rnorm_t1 = np.sqrt(diff_t1.dot(diff_t1))
            
            #t2
            t2_new = update_ucc3_t2(t1_old, t2, hf, df2).evaluate()
            res_t2 = t2_new - t2
            rnorm_t2 = np.sqrt(res_t2.dot(res_t2))
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
                rho_ov = t1 - 0.5 * einsum("ijab,jb->ia", t2, t1)
                return AmplitudeVector(t1 = -1.0 * t1, t2 = -1.0 * t2)
        raise ValueError("t amplitudes not converged.")

    def t1(self, space):
        return self.t_amplitudes.t1

    def t2(self, space):
        return self.t_amplitudes.t2

    @cached_member_function
    def td2(self, space):
        # td2 does not exist for UCC
        return self.t2(space).zeros_like()

    @cached_property
    @timed_member_call(timer="timer")
    def mp2_diffdm(self):
        """
        Return the UCC3 difference density in the MO basis.
        """
        hf = self.reference_state
        ret = OneParticleOperator(self.mospaces, is_symmetric=True)
        # NOTE: the following 3 blocks are equivalent to the cvs_p0 intermediates
        # defined at the end of this file
        ret.oo = -0.5 * einsum("ikab,jkab->ij", self.t2oo, self.t2oo)
        #ret.ov = -0.5 * (
        #    + einsum("ijbc,jabc->ia", self.t2oo, hf.ovvv)
        #    + einsum("jkib,jkab->ia", hf.ooov, self.t2oo)
        #) / self.df(b.ov)
        ret.ov = self.t1ov - 0.5 * einsum("ijab,jb->ia", self.t2oo, self.t1ov)
        ret.vv = 0.5 * einsum("ijac,ijbc->ab", self.t2oo, self.t2oo)

        if self.has_core_occupied_space:
            raise NotImplementedError("UCC3 not implemented for CVS.")

        ret.reference_state = self.reference_state
        return evaluate(ret)
