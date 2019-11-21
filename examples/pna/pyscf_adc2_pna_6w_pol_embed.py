#!/usr/bin/env python3
## vi: tabstop=4 shiftwidth=4 softtabstop=4 expandtab

import adcc
import cppe
import numpy as np
from pyscf import gto, scf
from pyscf.solvent import PE

from scipy import constants
eV = constants.value("Hartree energy in eV")  # Hartree to eV

mol = gto.M(
    atom="""
    C          8.64800        1.07500       -1.71100
    C          9.48200        0.43000       -0.80800
    C          9.39600        0.75000        0.53800
    C          8.48200        1.71200        0.99500
    C          7.65300        2.34500        0.05500
    C          7.73200        2.03100       -1.29200
    H         10.18300       -0.30900       -1.16400
    H         10.04400        0.25200        1.24700
    H          6.94200        3.08900        0.38900
    H          7.09700        2.51500       -2.01800
    N          8.40100        2.02500        2.32500
    N          8.73400        0.74100       -3.12900
    O          7.98000        1.33100       -3.90100
    O          9.55600       -0.11000       -3.46600
    H          7.74900        2.71100        2.65200
    H          8.99100        1.57500        2.99500
    """,
    basis='sto-3g',
)
pe_options = cppe.PeOptions()
pe_options.potfile = "pna_6w.pot"

scfres = PE(scf.RHF(mol), pe_options)
scfres.conv_tol = 1e-10
scfres.conv_tol_grad = 1e-10
scfres.max_cycle = 250
scfres.kernel()

state = adcc.adc2(scfres, n_singlets=5, conv_tol=1e-8)

n_exc = 3
ptlr = state.pe_ptlr_correction[n_exc] * eV
ptss = state.pe_ptss_correction[n_exc] * eV

# hard-coded Q-Chem values
np.testing.assert_allclose(
    np.array([ptss, ptlr]), np.array([-0.009186, -0.007242]),
    atol=1e-5
)