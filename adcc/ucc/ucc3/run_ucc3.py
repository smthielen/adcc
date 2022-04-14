import adcc
from ucc3 import LazyUcc3

from pyscf import gto, scf

# Run SCF in pyscf
mol = gto.M(
    atom='O 0 0 0;'
         'H 0 0 1.795239827225189;'
         'H 1.693194615993441 0 -0.599043184453037',
    basis='6-311++G**',
    unit="Bohr"
)
scfres = scf.RHF(mol)
scfres.conv_tol = 1e-10
scfres.conv_tol_grad = 1e-8
scfres.kernel()

refstate = adcc.ReferenceState(scfres)
mp2 = adcc.LazyMp(refstate)
ucc3 = LazyUcc3(refstate)

print("MP2 energy = ", mp2.energy(level=2))
print("UCC3 energy = ", ucc3.energy(level=2))

state = adcc.adc3(ucc3, n_singlets=3, conv_tol = 1e-8)
print(state.describe())
