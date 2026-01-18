#!/usr/bin/env python
# coding: utf-8

# In[2]:ƒ

import at
import numpy as np
#import numpy.linalg
from optics_correction import *
print(f"Load the lattice - radiation off")
ring = at.load_mat('betamodel.mat')
ring.radiation_off()
elements_ind = at.get_refpts(ring, "*")
cor_indices = at.get_refpts(ring, 'S[FIJ]2A*')
Corords = [cor_indices,cor_indices]
used_bpm = at.get_refpts(ring, at.elements.Monitor)
quad_indices = at.get_refpts(ring, at.elements.Quadrupole)
QD3 = at.get_refpts(ring, 'QD3[AE]*')
QF4 = at.get_refpts(ring, 'QF4[ABDE]*')
QD5 = at.get_refpts(ring, 'QD5[BD]*')
combined = np.concatenate((QD3, QF4, QD5))
quad_indices = np.sort(combined)
CAVords = at.get_refpts(ring, at.elements.RFCavity)
sext_indices = at.get_refpts(ring, at.elements.Sextupole)
skew_ord = at.get_refpts(ring, 'S[HFDIJ]*')


dkick = 1e-6



Px, Py, Etax, Qx, Qy, Bx, By = ORM_mu(dkick, ring, quad_indices, used_bpm)

np.save('Px_ebs', Px)
np.save('Py_ebs', Py)
np.save('Etax_ebs', Etax)
np.save('Qx_ebs', Qx)
np.save('Qy_ebs', Qy)
np.save('Bx_ebs', Bx)
np.save('By_ebs', By)
import re



#ORM
print("Calculate ORM before errors (Model)")

[elemdata0, beamdata, elemdata] = at.get_optics(ring, used_bpm)
twiss = elemdata

rdt1, rdt2, Etay = ORM_rdts_new(1e-6, ring, skew_ord, used_bpm)



np.save('difference_ebs', rdt1)
np.save('addit_ebs', rdt2)
np.save('Etay_ebs', Etay)


