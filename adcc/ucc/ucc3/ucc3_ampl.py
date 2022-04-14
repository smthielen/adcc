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
