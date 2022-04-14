#
# Class for the calculation of the five UCC3 intermediates
# analogous to the implementation in libadc/ucc
#
# i1_ooov and i2_ovvv to facilitate the ucc3 t1 calculation
# i3_oooo, i4_vvvv and i5_ovov to facilitate the ucc3 t2 calculation
#
# SMT
#

import adcc
from adcc.functions import einsum

class ucc3_intermediates:
    def __init__(self):
        print("Using UCC3 intermediates")

    def i1_ooov(self, hf, t2):
        return (
            - 0.5 * hf.ooov
            - einsum('ilkc,jlbc->kjib', hf.ooov, t2)
            + 0.25 * einsum('ibcd,jkcd->kjib', hf.ovvv, t2)
        )

    def i2_ovvv(self, hf, t2):
        return (
            0.5 * hf.ovvv
            - einsum('kbad,kjcd->jabc', hf.ovvv, t2)
            + 0.25 * einsum('klja,klbc->jabc', hf.ooov, t2)
        )

    def i3_oooo(self, hf, t2):
        return (
            0.5 * hf.oooo
            + 1.0/6.0 * einsum('klcd,ijcd->ijkl', hf.oovv, t2)
            + 1.0/12.0 * einsum('ijcd,klcd->ijkl', hf.oovv, t2)
        )

    def i4_vvvv(self, hf, t2):
        return (
            0.5 * hf.vvvv
            + 1.0/12.0 * einsum('klab,klcd->abcd', hf.oovv, t2)
        )

    def i5_ovov(self, hf, t2):
        return (
            - hf.ovov
            + 1.0/3.0 * einsum('klcd,ilad->kaic', hf.oovv, t2)
            + 1.0/3.0 * einsum('ilad,klcd->kaic', hf.oovv, t2)
        )
