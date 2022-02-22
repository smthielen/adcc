#
# Class for the calculation of the UCC3 t1 and t2 amplitudes
# analogous to the implementation in libadc/ucc
#
# i1_ooov and i2_ovvv needed for the ucc3 t1 calculation
# i3_oooo, i4_vvvv and i5_ovov needed for the ucc3 t2 calculation
#
# SMT
#

import adcc
from adcc.functions import einsum

def ucc3_t1_update(t1_in, t2_in, hf, df1, itm1_ooov, itm2_ovvv)
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

def ucc3_t2_update(t1_in, t2_in, hf, df2, itm3_oooo, itm4_vvvv, itm5_ovov)
    return (        
        hf.oovv
        - einsum('jika,kb->ijab', hf.ooov, t1_in).antisymmetrise(2, 3)
        + einsum('icab,jc->ijab', hf.ovvv, t1_in).antisymmetrise(0, 1)
        + einsum('klij,klab->ijab', itm3_oooo, t2_in)
        + einsum('abcd,ijcd->ijab', itm4_vvvv, t2_in)
        - 1.0/3.0 * einsum('klcd,jkcd,ilab->ijab', hf.oovv, t2_in, 
            t2_in).antisymmetrise(0, 1)
        - 1.0/6.0 * einsum('kjcd,klcd,ilab->ijab', hf.oovv, t2_in,
            t2_in).antisymmetrise(0, 1)
        - 1.0/6.0 * einsum('klcd,jkdc,ilab->ijab', t2_in, t2_in,
            hf.oovv).antisymmetrise(0, 1)
        - 1.0/3.0 * einsum('klcd,klcb,ijad->ijab', hf.oovv, t2_in, 
            t2_in).antisymmetrise(2, 3)
        - 1.0/6.0 * einsum('klcb,klcd,ijad->ijab', hf.oovv, t2_in,
            t2_in).antisymmetrise(2, 3)
        - 1.0/6.0 * einsum('klcd,klcb,ijad->ijab', t2_in, t2_in,
            hf.oovv).antisymmetrise(2, 3)
        + einsum('kaic,jkbc->ijab', itm5_ovov, 
            t2_in).antisymmetrise(2, 3).antisymmetrise(0, 1)
        ) / df2
