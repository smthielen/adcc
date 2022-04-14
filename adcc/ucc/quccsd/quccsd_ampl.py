#
# Class for the calculation of the qUCCSD t1 and t2 amplitudes
# analogous to the implementation in libadc/ucc
#
# SMT
#

import adcc
import adcc.block as b
import numpy as np
from adcc.functions import einsum, direct_sum
from pyscf import gto, scf

#qUCCSD t1 calculation
def update_quccsd_t1(t1_in, t2_in, hf):
    return (
	#contract(j|b, - 1.0 * i_ovov(j|a|i|b), t1(j|b))
        - 1.0 * einsum('jaib,jb->ia', hf.ovov, t1_in)        
        #+ 0.5 * contract(j|b, i_oovv(i|j|a|b), t1(j|b))
        + 0.5 * einsum('ijab,jb->ia', hf.oovv, t1_in)        
        #+ 0.5 * contract(j|b|c, - 1.0 * i_ovvv(j|a|c|b), t2(i|j|c|b))
        + 0.5 * einsum('jacb,ijcb->ia', hf.ovvv, t2_in)        
        #- 0.5 * contract(j|k|b, i_ooov(k|j|i|b), t2(j|k|b|a))
        - 0.5 * einsum('kjib,jkba->ia', hf.ooov, t2_in)        
        #- 0.5 * contract(k|l, i_ooov(k|i|l|a),
        #           contract(j|b|c, t2(j|k|b|c), t2(j|l|b|c)))
        - 0.5 * einsum('kila,jkbc,jlbc->ia', hf.ooov, t2_in, t2_in)        
        #+ 0.5 * contract(c|d, i_ovvv(i|c|a|d),
        #           contract(j|k|b, t2(j|k|b|d), t2(j|k|b|c)))
        + 0.5 * einsum('icad,jkbd,jkbc->ia', hf.ovvv, t2_in, t2_in)        
        #- contract(k|l|c, t2(k|l|c|a),
        #           contract(j|b, i_ooov(i|j|l|b), t2(j|k|b|c)))
        - einsum('klca,ijlb,jkbc->ia', t2_in, hf.ooov, t2_in)        
        #+ contract(k|c|d, t2(k|i|c|d),
        #           contract(j|b, i_ovvv(j|d|b|a), t2(j|k|b|c)))
        + einsum('kicd,jdba,jkbc->ia', t2_in, hf.ovvv, t2_in)        
        #- 0.25 * contract(l|c, t2(i|l|a|c),
        #           contract(j|k|b, i_ooov(k|j|l|b), t2(j|k|b|c)))
        - 0.25 * einsum('ilac,kjlb,jkbc->ia', t2_in, hf.ooov, t2_in)        
        #+ 0.25 * contract(k|c, t2(i|k|a|c), // 0.25
        #           contract(j|b|d, i_ovvv(j|c|b|d), t2(j|k|b|d)))
        + 0.25 * einsum('ikac,jcbd,jkbd->ia', t2_in, hf.ovvv, t2_in)        
        #+ 0.25 * contract(j|k|c, t2(j|k|c|a),   // 0.25
        #           contract(b|d, i_ovvv(i|c|b|d), t2(j|k|b|d)))
        + 0.25 * einsum('jkca,icbd,jkbd->ia', t2_in, hf.ovvv, t2_in)        
        #- 0.25 * contract(l|b|c, t2(i|l|c|b),
        #           contract(j|k, i_ooov(k|j|l|a), t2(j|k|b|c))),
        - 0.25 * einsum('ilcb,kjla,jkbc->ia', t2_in, hf.ooov, t2_in)        
        #
        #qUCCSD s-d products
        #
        #+ 5./12. * contract(j|b, t1(j|b), contract(k|c, t2(i|k|a|c), i_oovv(j|k|b|c)))
        + 5.0/12.0 * einsum('jb,ikac,jkbc->ia', t1_in, t2_in, hf.oovv)
	#- 1./3. * contract(k, t1(k|a), contract(j|b|c, t2(i|j|c|b), i_oovv(j|k|b|c)))
        - 1.0/3.0 * einsum('ka,ijcb,jkbc->ia', t1_in, t2_in, hf.oovv)
	#- 1./3. * contract(c, t1(i|c), contract(j|k|b, t2(j|k|b|a), i_oovv(j|k|b|c)))
        - 1.0/3.0 * einsum('ic,jkba,jkbc->ia', t1_in, t2_in, hf.oovv)
	#+ 0.5 * contract(k|c, t1(k|c), contract(j|b, t2(j|k|b|a), i_ovov(j|c|i|b)))
        + 0.5 * einsum('kc,jkba,jcib->ia', t1_in, t2_in, hf.ovov)
	#+ 0.5 * contract(k|c, t1(k|c), contract(j|b, t2(i|j|c|b), i_ovov(j|a|k|b)))
        + 0.5 * einsum('kc,ijcb,jakb->ia', t1_in, t2_in, hf.ovov)
	#- 1./3. * contract(k|c, t1(k|c), contract(j|b, t2(j|k|c|b), i_oovv(i|j|a|b)))
        - 1.0/3.0 * einsum('kc,jkcb,ijab->ia', t1_in, t2_in, hf.oovv)
	#- 1./6. * contract(k, t1(k|a), contract(j|b|c, t2(j|k|b|c), i_oovv(j|i|b|c)))
        - 1.0/6.0 * einsum('ka,jkbc,jibc->ia', t1_in, t2_in, hf.oovv)
	#- 1./6. * contract(c, t1(i|c), contract(j|k|b, t2(j|k|b|c), i_oovv(k|j|a|b)))
        - 1.0/6.0 * einsum('ic,jkbc,kjab->ia', t1_in, t2_in, hf.oovv)
	#+ 0.25 * contract(j|c, t1(j|c), contract(b|d, t2(i|j|b|d), i_vvvv(a|c|b|d)))
        + 0.25 * einsum('jc,ijbd,acbd->ia', t1_in, t2_in, hf.vvvv)
	#+ 0.25 * contract(k|b, t1(k|b), contract(j|l, t2(j|l|a|b), i_oooo(j|l|i|k)))
        + 0.25 * einsum('kb,jlab,jlik->ia', t1_in, t2_in, hf.oooo)
        #
        #qUCCSD s-s products
        #
	#- contract(j|b, t1(j|b), contract(c, t1(i|c), i_ovvv(j|a|c|b)))
        - einsum('jb,ic,jacb->ia', t1_in, t1_in, hf.ovvv)	
	#- contract(j|b, t1(j|b), contract(k, t1(k|a), i_ooov(k|j|i|b)))
        - einsum('jb,ka,kjib->ia', t1_in, t1_in, hf.ooov)	
	#+ 0.5 * contract(j|b, t1(j|b), contract(c, t1(i|c), i_ovvv(j|c|b|a)))
        + 0.5 * einsum('jb,ic,jcba->ia', t1_in, t1_in, hf.ovvv)	
	#- 0.5 * contract(j|b, t1(j|b), contract(k, t1(k|a), i_ooov(i|j|k|b)))
        - 0.5 * einsum('jb,ka,ijkb->ia', t1_in, t1_in, hf.ooov)	
	#+ 0.5 * contract(j|b, t1(j|b), contract(c, t1(j|c), i_ovvv(i|b|a|c)))
        + 0.5 * einsum('jb,jc,ibac->ia', t1_in, t1_in, hf.ovvv)	
	#- 0.5 * contract(j|b, t1(j|b), contract(k, t1(k|b), i_ooov(j|i|k|a))),
        - 0.5 * einsum('jb,kb,jika->ia', t1_in, t1_in, hf.ooov)
        #- df_ov(i|a));
        ) / -1.0 * hf.df_ov

#qUCCSD t2 calculation
def update_quccsd_t2(t1_in, t2_in, hf, df2):
    return (
        i_oovv(i|j|a|b)
        hf.oovv        
        #- asymm(a, b, contract(k, i_ooov(j|i|k|a), t1(k|b)))
        - einsum('jika,kb->ijab', hf.ooov, t1_in).antisymmetrise(2, 3)
        #+ asymm(i, j, contract(c, i_ovvv(i|c|a|b), t1(j|c)))
        + einsum('icab,jc->ijab', hf.ovvv, t1_in).antisymmetrise(0, 1)
        #+ 0.5 * contract(k|l, i_oooo(k|l|i|j), t2(k|l|a|b))
        + 0.5 * einsum('klij,klab->ijab', hf.oooo, t2_in)
        #+ 0.5 * contract(c|d, i_vvvv(a|b|c|d), t2(i|j|c|d))
        + 0.5 * einsum('abcd,ijcd->ijab', hf.vvvv, t2_in)
        #+ asymm(i, j, asymm(a, b, contract(k|c, - i_ovov(k|a|i|c), t2(j|k|b|c))))
        - 1.0 * einsum('kaic,jkbc->ijab', hf.ovov, t2_in).antisymmetrise(2, 3).antisymmetrise(0, 1)
        #+ asymm(i, j, asymm(a, b, contract(l|d, t2(j|l|b|d),
        #                   contract(k|c, 1.0/3.0 * i_oovv(k|l|c|d), t2(i|k|a|c)))))
        + 1.0/3.0 * einsum('jlbd,klcd,ikac->ijab', t2_in, hf.oovv, t2_in).antisymmetrise(2, 3).antisymmetrise(0, 1)
        #+ contract(k|l, t2(k|l|a|b),
        #           contract(c|d, 1.0/6.0 * i_oovv(k|l|c|d), t2(i|j|c|d)))
        + 1.0/6.0 * einsum('klab,klcd,ijcd->ijab', t2_in, hf.oovv, t2_in)
	#- asymm(a, b, contract(k|l|c, t2(k|l|c|b),
        #               contract(d, 1.0/3.0 * i_oovv(k|l|c|d), t2(i|j|a|d))))
        - 1.0/3.0 * einsum('klcb,klcd,ijad->ijab', t2_in, hf.oovv, t2_in).antisymmetrise(2, 3)
	#- asymm(i, j, contract(k|c|d, t2(j|k|d|c),
        #               contract(l, 1.0/3.0 * i_oovv(k|l|c|d), t2(i|l|a|b))))
        - 1.0/3.0 * einsum('jkdc,klcd,ilab->ijab', t2_in, hf.oovv, t2_in).antisymmetrise(0, 1)
	#+ asymm(i, j, asymm(a, b, contract(k|c, t2(j|k|b|c),
        #                   contract(l|d, 1.0/3.0 * i_oovv(i|l|a|d), t2(k|l|c|d)))))
        + 1.0/3.0 * einsum('jkbc,ilad,klcd->ijab', t2_in, hf.oovv, t2_in).antisymmetrise(2, 3).antisymmetrise(0, 1)
	#+ contract(k|l, t2(k|l|a|b),
        #           contract(c|d, 1.0/12.0 * i_oovv(i|j|c|d), t2(k|l|c|d)))
        + 1.0/12.0 * einsum('klab,ijcd,klcd->ijab', t2_in, hf.oovv, t2_in)
	#+ contract(c|d, t2(i|j|c|d),
        #           contract(k|l, 1.0/12.0 * i_oovv(k|l|a|b), t2(k|l|c|d)))
        + 1.0/12.0 * einsum('ijcd,klab,klcd->ijab', t2_in, hf.oovv, t2_in)
	#- asymm(a, b, contract(k|l|c, t2(k|l|c|b),
        #               contract(d, 1.0/6.0 * i_oovv(i|j|a|d), t2(k|l|c|d))))
        - 1.0/6.0 * einsum('klcb,ijad,klcd->ijab', t2_in, hf.oovv, t2_in).antisymmetrise(2, 3)
	#- asymm(i, j, contract(k|c|d, t2(j|k|d|c),
        #                   contract(l, 1.0/6.0 * i_oovv(i|l|a|b), t2(k|l|c|d))))
        - 1.0/6.0 * einsum('jkdc,ilab,klcd->ijab', t2_in, hf.oovv, t2_in).antisymmetrise(0, 1)
        #- asymm(i, j, contract(l, t2(i|l|a|b),
        #                   contract(k|c|d, 1.0/6.0 * i_oovv(k|j|c|d), t2(k|l|c|d))))
        - 1.0/6.0 * einsum('ilab,kjcd,klcd->ijab', t2_in, hf.oovv, t2_in).antisymmetrise(0, 1)
	#- asymm(a, b, contract(d, t2(i|j|a|d),
        #                   contract(k|l|c, 1.0/6.0 * i_oovv(l|k|b|c), t2(k|l|c|d)))),
        - 1.0/6.0 * einsum('ijad,lkbc,klcd->ijab', t2_in, hf.oovv, t2_in).antisymmetrise(2, 3) 
        #
        #qUCCSD s-d products
        #
        #- asymm(i, j, contract(l|c, t1(l|c), contract(k, t2(i|k|a|b), i_ooov(j|l|k|c))))	
        - einsum('lc,ikab,jlkc->ijab', t1_in, t2_in, hf.ooov).antisymmetrise(0, 1)	
	#+ asymm(a, b, contract(l|c, t1(l|c), contract(d, t2(i|j|a|d), i_ovvv(l|d|c|b))))	
        + einsum('lc,ijad,ldcb->ijab', t1_in, t2_in, hf.ovvv).antisymmetrise(2, 3)	
	#+ 0.5 * asymm(i, j, contract(l|c, t1(l|c), contract(d, t2(j|l|d|c), i_ovvv(i|d|a|b))))	
        + 0.5 * einsum('lc,jldc,idab->ijab', t1_in, t2_in, hf.ovvv).antisymmetrise(0, 1)	
	#- 0.5 * asymm(a, b, contract(l|c, t1(l|c), contract(k, t2(k|l|b|c), i_ooov(j|i|k|a))))	
        - 0.5 * einsum('lc,klbc,jika->ijab', t1_in, t2_in, hf.ooov).antisymmetrise(2, 3)	
	#+ contract(l|c, t1(l|c), contract(k, t2(k|l|a|b), i_ooov(i|j|k|c)))	
        + einsum('lc,klab,ijkc->ijab', t1_in, t2_in, hf.ooov)
	#+ asymm(i, j, asymm(a, b, contract(l|c, t1(l|c), contract(k, t2(j|k|c|a), i_ooov(i|l|k|b)))))	
        + einsum('lc,jkca,ilkb->ijab', t1_in, t2_in, hf.ooov).antisymmetrise(2, 3).antisymmetrise(0, 1)	
	#- asymm(i, j, asymm(a, b, contract(l|c, t1(l|c), contract(d, t2(i|l|d|b), i_ovvv(j|d|c|a)))))	
        - einsum('lc,ildb,jdca->ijab', t1_in, t2_in, hf.ovvv).antisymmetrise(2, 3).antisymmetrise(0, 1)	
	#- contract(l|c, t1(l|c), contract(d, t2(i|j|d|c), i_ovvv(l|d|b|a)))	
        - einsum('lc,ijdc,ldba->ijab', t1_in, t2_in, hf.ovvv)
	#+ asymm(i, j, contract(k|c, t1(k|c), contract(l, t2(i|l|a|b), i_ooov(k|l|j|c))))	
        + einsum('kc,ilab,kljc->ijab', t1_in, t2_in, hf.ooov).antisymmetrise(0, 1)
	#+ asymm(a, b, contract(k|c, t1(k|c), contract(d, t2(i|j|a|d), i_ovvv(k|b|c|d))))	
        + einsum('kc,ijad,kbcd->ijab', t1_in, t2_in, hf.ovvv).antisymmetrise(2, 3)
        #+ asymm(i, j, asymm(a, b, contract(l, t1(l|b), contract(k|c, t2(i|k|a|c), i_ooov(k|l|j|c)))))	
        + einsum('lb,ikac,kljc->ijab', t1_in, t2_in, hf.ooov).antisymmetrise(2, 3).antisymmetrise(0, 1)
	#+ asymm(i, j, asymm(a, b, contract(d, t1(j|d), contract(k|c, t2(i|k|a|c), i_ovvv(k|b|c|d)))))	
        + einsum('jd,ikac,kbcd->ijab', t1_in, t2_in, hf.ovvv).antisymmetrise(2, 3).antisymmetrise(0, 1)
	#- 0.5 * asymm(i, j, contract(c, t1(j|c), contract(k|l, t2(k|l|b|a), i_ooov(k|l|i|c))))	
        - 0.5 * einsum('jc,klba,klic->ijab', t1_in, t2_in, hf.ooov).antisymmetrise(0, 1)
	#- 0.5 * asymm(a, b, contract(k, t1(k|b), contract(c|d, t2(i|j|d|c), i_ovvv(k|a|c|d))))
        - 0.5 * einsum('kb,ijdc,kacd->ijab', t1_in, t2_in, hf.ovvv).antisymmetrise(2, 3)
        #
        #qUCCSD s-s products
        #
	#+ 0.5 * asymm(a, b, contract(k, t1(k|a), contract(l, t1(l|b), i_oooo(k|l|i|j))))
        + 0.5 * einsum('ka,lb,klij->ijab', t1_in, t1_in, hf.oooo).antisymmetrise(2, 3)	 
        #+ 0.5 * asymm(i, j, contract(c, t1(i|c), contract(d, t1(j|d), i_vvvv(a|b|c|d))))
        + 0.5 * einsum('ic,jd,abcd->ijab', t1_in, t1_in, hf.vvvv).antisymmetrise(0, 1)	 
	#- asymm(a,b, asymm(i, j, contract(k, t1(k|b), contract(c, t1(i|c), i_ovov(k|a|j|c)))))
        - einsum('kb,ic,kajc->ijab', t1_in, t1_in, hf.ovov).antisymmetrise(0, 1).antisymmetrise(2, 3)	 
	#- 1./3. * asymm(a, b, contract(k, t1(k|b), contract(c, t1(k|c), i_oovv(i|j|a|c))))
        - 1.0 / 3.0 * einsum('kb,kc,ijac->ijab', t1_in, t1_in, hf.oovv).antisymmetrise(2, 3)	 
	#- 1./3. * asymm(i, j, contract(c, t1(j|c), contract(k, t1(k|c), i_oovv(i|k|a|b)))),
        - 1.0 / 3.0 * einsum('jc,kc,ikab->ijab', t1_in, t1_in, hf.oovv).antisymmetrise(0, 1)	 
        #- df_oovv(i|j|a|b));
        ) / df2
