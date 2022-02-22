import adcc
import numpy as np
from adcc.misc import cached_member_function
from adcc.functions import einsum, direct_sum
import adcc.block as b
from diis import Diis


def update_amplitude_ucc2(ampl_in, hf, df2):
    return (
        hf.oovv
        + 0.5 * einsum('klij,klab->ijab', hf.oooo, ampl_in)
        + 0.5 * einsum('abcd,ijcd->ijab', hf.vvvv, ampl_in)
        - adcc.contract('kaic,jkbc->ijab', hf.ovov, ampl_in)
        + adcc.contract('kajc,ikbc->ijab', hf.ovov, ampl_in)
        + adcc.contract('kbic,jkac->ijab', hf.ovov, ampl_in)
        - adcc.contract('kbjc,ikac->ijab', hf.ovov, ampl_in)
    ) / df2


class LazyUcc2(adcc.LazyMp):
    def __init__(self, hf):
        super().__init__(hf)
        if self.has_core_occupied_space:
            raise NotImplementedError("UCC2 not implemented for CVS.")

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
        df2 = -1.0 * direct_sum("-i-j+a+b->ijab",
                                hf.foo.diagonal(), hf.foo.diagonal(),
                                hf.fvv.diagonal(), hf.fvv.diagonal())
        df2.evaluate()
        diis_handler = Diis()
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

    @cached_member_function
    def td2(self, space):
        # td2 does not exist for UCC
        return self.t2(space).zeros_like()
