#
# Class for the calculation of the qUCCSD energy
# analogous to the implementation in libadc/ucc
#
# SMT
#

import adcc
import adcc.block as b
import numpy as np
from adcc.functions import einsum, direct_sum
from pyscf import gto, scf

#qUCCSD energy
def quccsd_energy(t1, t2, hf):
    return (
        #+ 0.25 * dot_product(i_oovv(i|j|a|b), t2(i|j|a|b))
        0.25 * hf.oovv.dot(t2)       
        #+ 1.0/6.0 * dot_product(contract(j|b, i_oovv(i|j|a|b), t1(j|b)), t1(i|a));
        + 1.0/6.0 * t1.dot(einsum('ijab,jb->ia', hf.oovv, t1)) 
        #
        #t2*t2*t2 terms
        #
        #- 1.0/6.0 * dot_product(i_oovv(i|j|a|b), contract(l|d, t2(j|l|b|d), contract(k|c, t2(k|l|c|d), t2(i|k|a|c))))
        - 1.0/6.0 * hf.oovv.dot(einsum('jlbd,klcd,ikac->ijab', t2,  t2, t2))
	#+ 1.0/12.0 * dot_product(i_oovv(i|j|a|b), contract(c, t2(i|j|a|c), contract(k|l|d, t2(k|l|c|d), t2(k|l|b|d))))
        + 1.0/12.0 * hf.oovv.dot(einsum('ijac,klcd,klbd->ijab', t2, t2, t2))
	#+ 1.0/12.0 * dot_product(i_oovv(i|j|a|b), contract(k, t2(i|k|a|b), contract(l|c|d, t2(k|l|c|d), t2(j|l|c|d))))
        + 1.0/12.0 * hf.oovv.dot(einsum('ikab,klcd,jlcd->ijab', t2, t2, t2))
	#- 1.0/48.0 * dot_product(i_oovv(i|j|a|b), contract(k|l, t2(k|l|a|b), contract(c|d, t2(k|l|c|d), t2(i|j|c|d))))
        - 1.0/48.0 * hf.oovv.dot(einsum('klab,klcd,ijcd->ijab', t2, t2, t2))
        #
        #t2*t2*t1 terms
        # 
	#+ 0.5 * dot_product(t1(i|a), contract(j|k, - i_ooov(i|j|k|a), contract(l|b|c, t2(j|l|b|c), t2(k|l|b|c))))
        + 0.5 * t1.dot(einsum('ijka,jlbc,klbc->ia', -1.0 * hf.ooov, t2, t2))
	#- 0.5 * dot_product(t1(i|a), contract(b|c, i_ovvv(i|c|a|b), contract(j|k|d, t2(j|k|c|d), t2(j|k|b|d))))
        - 0.5 * t1.dot(einsum('icab,jkcd,jkbd->ia', hf.ovvv, t2, t2))
	#+ dot_product(t1(j|b), contract(i|k|a, - i_ooov(k|j|i|a), contract(l|c, t2(i|l|b|c), t2(l|k|c|a))))
        + t1.dot(einsum('kjia,ilbc,lkca->jb', -1.0 * hf.ooov, t2, t2))
	#- dot_product(t1(j|b), contract(i|a|c, i_ovvv(i|c|a|b), contract(k|d, t2(j|k|c|d), t2(i|k|a|d))))
        - t1.dot(einsum('icab,jkcd,ikad->jb', hf.ovvv, t2, t2))
	#+ 1./6. * dot_product(t1(l|c), contract(i|b, t2(i|l|b|c), contract(j|k|a, i_ooov(j|k|i|a), t2(k|j|a|b))))
        + 1.0/6.0 * t1.dot(einsum('ilbc,jkia,kjab->jb', t2, hf.ooov, t2))
	#- 1./6. * dot_product(t1(k|d), contract(j|c, t2(j|k|c|d), contract(i|a|b, i_ovvv(i|c|a|b), t2(i|j|a|b))))
        - 1.0/6.0 * t1.dot(einsum('jkcd,icab,ijab->kd', t2, hf.ovvv, t2))
	#- 0.25 * dot_product(t1(l|a), contract(i|j|k, i_ooov(k|j|i|a), contract(c|b, t2(i|l|c|b), t2(k|j|c|b))))
        - 0.25 * t1.dot(einsum('kjia,ilcb,kjcb->la', hf.ooov, t2, t2))
	#+ 0.25 * dot_product(t1(i|d), contract(a|b|c, - i_ovvv(i|c|a|b), contract(j|k, t2(j|k|d|c), t2(k|j|a|b))))
        + 0.25 * t1.dot(einsum('icab,jkdc,kjab->id', -1.0 * hf.ovvv, t2, t2))
        #
        #t2*t1*t1 terms
        #
	#- 1./6. * dot_product(t1(i|a), contract(j|b, i_oovv(i|j|a|b), contract(k|c, t1(k|c), t2(j|k|b|c))))
        - 1.0/6.0 * t1.dot(einsum('ijab,kc,jkbc->ia', hf.oovv, t1, t2))
        #+ 1./6. * dot_product(t1(i|c), contract(k, t1(k|c), contract(j|a|b, i_oovv(i|j|a|b), t2(j|k|b|a))))
        + 1.0/6.0 * t1.dot(einsum('kc,ijab,jkba->ic', t1, hf.oovv, t2))
	#+ 1./6. * dot_product(t1(k|a), contract(c, t1(k|c), contract(i|j|b, i_oovv(i|j|a|b), t2(i|j|c|b))))
        + 1.0/6.0 * t1.dot(einsum('kc,ijab,ijcb->ka', t1, hf.oovv, t2))
	#+ 2./3. * dot_product(t1(k|a), contract(j|c, t1(j|c), contract(i|b, - i_ovov(j|b|i|a), t2(i|k|b|c))))
        + 2.0/3.0 * t1.dot(einsum('jc,jbia,ikbc->ka', t1, -1.0 * hf.ovov, t2))
	#- 1./6. * dot_product(t1(k|a), contract(l|b, t1(l|b), contract(i|j, i_oooo(k|l|i|j), t2(i|j|a|b))))
        - 1.0/6.0 * t1.dot(einsum('lb,klij,ijab->ka', t1, hf.oooo, t2))
	#- 1./6. * dot_product(t1(i|a), contract(j|b, t1(j|b), contract(c|d, i_vvvv(c|d|a|b), t2(i|j|c|d))))
        - 1.0/6.0 * t1.dot(einsum('jb,cdab,ijcd->ia', t1, hf.vvvv, t2))
        #
        #t1*t1*t1 terms
        #
	#+ 2./3. * dot_product(t1(j|b), contract(k, t1(k|b), contract(i|a, - i_ooov(i|j|k|a), t1(i|a))))
        + 2.0/3.0 * t1.dot(einsum('kb,ijka,ia->jb', t1, -1.0 * hf.ooov, t1))	
	#- 2./3. * dot_product(t1(j|a), contract(b, t1(j|b), contract(i|c, - i_ovvv(i|a|b|c), t1(i|c))));
        - 2.0/3.0 * t1.dot(einsum('jb,iabc,ic->ja', t1, -1.0 * hf.ovvv, t1))	
        )
