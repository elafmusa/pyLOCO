""""" 
Computation of the resonant driving terms at the location of the BPM
"""""
import numpy as np
import at
from collections import namedtuple
from at.physics import find_orbit4, find_orbit6
from at.plot.generic import baseplot
import copy


def dphase(phib, phik, mtune):
	nb=phib.shape[1]
	nk=phik.shape[1]
	dph=phik*np.ones((nb, 1))-np.transpose(phib)*np.ones((1, nk))
	neg=dph<0
	dph[neg]=dph[neg]+mtune
	return dph


def getT1(a, ind):
	t1=0
	if hasattr(a, 'T1'):
		t1=-a.T1[ind]
	return t1


def get_PolynomB(ring, ind):
	temp=-1
	a=ring[ind]
	if hasattr(a, 'PolynomB'):
		if len(a.PolynomB)>1:
			if not (np.isnan(a.PolynomB[1]) or np.isnan(a.PolynomA[1])):
				temp = ind
	# else:
		# print(f'{ring[ind].FamName} has no PolynomB')
	return temp


def excludeelements(array, ind):
	a = copy.deepcopy(array)
	for i in ind:
		temp = np.where(a == i)
		a = np.delete(a, temp)
	return a


def getR1(a):
	r1=0
	if hasattr(a, 'R1'):
   		r1=a.R1[1, 3]
	return r1


def normal_quad_rdts_response(ring, bpmidx, qcoridx):
	"""
	ring    : AT lattice
	bpmindx : BPM indexes
	qcoridx : normal quadrupole indexes
	fx    : f2000 RDT 
	fx    : f0020 RDT
	qcor  : qcor.beta, qcor.phase beta and phase at the quad index (averaged) 
	to obtain rdt for a given set of skewidx strengths (KL)
	f2000=f1.*k1.*Lquad
	f0020=f2.*k1.*Lquad
	this function is an exact copy of semrdtresp by L.Farvacque
    see also: atavedata_mod
	"""

	nbpm = bpmidx.shape[0]
	nqcor= qcoridx.shape[0]
	fx=[]
	fz=[]
	radflag = ring.radiation
	if radflag:
		ring.radiation_off()
		print('Warning, this function calls avlinopt (4D). Consequently, the radiation has been turned off.')
	#get the optics
	allvalues = np.append(bpmidx, ring.i_range[-1]-1)
	allvalues= np.concatenate((qcoridx, allvalues))
	output=np.unique(allvalues, return_index=True, return_inverse=True)
	kl = output[2]
	refpts=output[0]
	jcor=kl[0:nqcor]
	jbpm=kl[nqcor:(nqcor+nbpm)]
	jend=kl[-1]

	a=at.avlinopt(ring, 0, refpts)
	vdata=a[0]
	avebeta=a[1]
	avemu=a[2]
	mtx=vdata[jend].mu[0]
	mtz=vdata[jend].mu[1]

	# Extract parameters
	qcor = namedtuple('qcor', ['beta', 'phase'])
	qcor.beta=avebeta[jcor, :]
	qcor.phase=avemu[jcor, :]

	# Compute terms
	dphix=dphase(np.atleast_2d(avemu[jbpm, 0]), np.atleast_2d(avemu[jcor, 0]), mtx)
	dphiz=dphase(np.atleast_2d(avemu[jbpm, 1]), np.atleast_2d(avemu[jcor, 1]), mtz)

	Df2000_DQuad = np.multiply(-np.atleast_2d(avebeta[jcor, 0])*np.ones((nbpm, 1)),
							   (np.cos(2*dphix)+1j*np.sin(2*dphix)))/(1-(np.cos(2*mtx)+1j*np.sin(2*mtx)))/8

	Df0020_DQuad = np.multiply(np.atleast_2d(avebeta[jcor, 1])*np.ones((nbpm, 1)),
							   (np.cos(2*dphiz)+1j*np.sin(2*dphiz)))/(1-(np.cos(2*mtz)+1j*np.sin(2*mtz)))/8
	
	if radflag:
		ring.radiation_on()

	return Df2000_DQuad, Df0020_DQuad, qcor


def skew_quad_rdts_response(ring, bpmidx, skewidx):
	"""
	SEMRDT compute resonance driving terms at BPM locations
	[f1,f2,skew]=semrdtresp_mod(mach,bpmidx,skewidx)
	ring    : AT lattice
	bpmindx : BPM indexes
	skewidx : skew quadrupole indexes
	f1    : f1001 RDT 
	f2    : f1010 RDT
	skew  : skew.beta skew.phase beta and phase at the skew index (averaged) 
	to obtain rdt for a given set of skewidx strengths (KL)
	f1001=f1.*k1s.*Lskew
	f1010=f2.*k1s.*Lskew
	this function is an exact copy of semrdtresp by L.Farvacque
%see also: atavedata_mod
	"""
	
	nbpm = bpmidx.shape[0]
	nskew= skewidx.shape[0]
	f1=[]
	f2=[]
	radflag = ring.radiation
	if radflag:
		ring.radiation_off()
		print('Warning, this function calls avlinopt (4D). Consequently, the radiation has been turned off.')
	# Get the optics

	allvalues = np.append(bpmidx,ring.i_range[-1]-1)
	allvalues= np.concatenate((skewidx,allvalues))
	output=np.unique(allvalues, return_index=True,return_inverse=True )
	kl = output[2]
	refpts=output[0]
	jsk=kl[0:nskew]
	jbpm=kl[nskew:(nskew+nbpm)]
	jend=kl[-1];

	a=at.avlinopt(ring,0,refpts)
	vdata=a[0]
	avebeta=a[1]
	avemu=a[2]
	mtunes=vdata[jend].mu
	mtx=vdata[jend].mu[0]
	mtz=vdata[jend].mu[1]

	""" to be translated
		if avebeta(avebeta<0).size!=0
			bx=arrayfun(@(a)a.beta(1),vdata)
			by=arrayfun(@(a)a.beta(2),vdata)
			avebeta=[bx,by];
			warning('on','all')
			warning('negative data in AVEBETA! using beta at entrance!')
			save('failingavebetalattice.mat','mach','bpmidx','skewidx')
			warning('off','all')
	"""

	# Extract parameters
	skew = namedtuple('skew', ['beta', 'phase'])
	skew.beta=avebeta[jsk, :]
	skew.phase=avemu[jsk, :]
	bpm = namedtuple('bpm', ['beta', 'phase'])
	bpm.beta=avebeta[jbpm, :]
	bpm.phase=vdata[jbpm].mu

	# Compute terms
	
	jsqb=np.real(np.sqrt(np.multiply(skew.beta[:, 0], skew.beta[:, 1])))
	dphix = dphase(np.atleast_2d(bpm.phase[:, 0]), np.atleast_2d(skew.phase[:, 0]), mtx)
	dphiz = dphase(np.atleast_2d(bpm.phase[:, 1]), np.atleast_2d(skew.phase[:, 1]), mtz)

	re1 = np.multiply(jsqb*np.ones((nbpm, 1)), np.cos(dphix-dphiz))
	im1 = np.multiply(jsqb*np.ones((nbpm, 1)), np.sin(dphix-dphiz))
	t1=mtx-mtz

	denom1 = 4*(1-complex(np.cos(t1), np.sin(t1)))
	Df1001_Dskew=(re1+1j*im1)/denom1

	re2 = np.multiply(jsqb*np.ones((nbpm, 1)), np.cos(dphix+dphiz))
	im2 = np.multiply(jsqb*np.ones((nbpm, 1)), np.sin(dphix+dphiz))
	t2=mtx+mtz

	denom2 = 4*(1-complex(np.cos(t2), np.sin(t2)))
	Df1010_Dskew=(re2+1j*im2)/denom2
	
	if radflag:
		ring.radiation_on()
		
	return Df1001_Dskew, Df1010_Dskew, skew
	
	
def EquivalentGradientsFromAlignments6D(ring, inds=None, inCOD=None):
	"""
	EQUIVALENTGRADIENTSFROMALIGNMENTS6D Estimated normal quad gradients from sext offsets
	[kn,    1) estimated normal quad gradients from sext offsets, quad
		   errors in quadrupoles and sextupoles.
	ks,    2) estimated skew quad gradients from sext offsets, quad
		   errors in quadrupoles, quadrupole rotation.
	ind    3) indexes of locations at wich kn and ks are found
	]=EquivalentGradientsFromAlignments6D(
	r,     1) AT lattice structure with errors
	inCOD  2) closed orbit
	)

	the function finds the closed orbit at sextupoles and converts it into
	equivalent quadrupole and skew quadrupole gradients for the computation
	of skew and normal quadrupole
	RDT quadrupole rotations are also converted in skew quadrupole gradients.
	It returns the complete list of normal (kn) and skew (ks) quadrupole
	gradients at the given indexes (ind) ( not integrated, PolynomB)

	quadrupole and skew quadrupole errors are introduced via COD in
	sextupoles
	"""
	indsext=ring.get_refpts(at.Sextupole)
	#print(f'there are {len(indsext)} sextupoles')
	b3=np.array([ring[ind].PolynomB[2] for ind in indsext])

	_, oin = ring.find_orbit(refpts=indsext,orbit=inCOD)    # orbit at entrance of sextupole DO NOT USE HERE findorbit6Err!
	_, oout = ring.find_orbit(refpts=indsext+1,orbit=inCOD) # orbit at exit of sextupole

	xmisal=np.array([getT1(ring[ind],0) for ind in indsext])
	ymisal=np.array([getT1(ring[ind],2) for ind in indsext])

	Dx=(oout[:,0]+oin[:,0])/2 # orbit average in sextupole
	Dy=(oout[:,2]+oin[:,2])/2 # orbit average in sextupole

	# quadrupole errors in sextupoles

	Qsext = []

	for ind in indsext:

		Qsext.append(ring[ind].PolynomB[1])
		ring[ind].PolynomB[1] = 0 ## this is added

	kn_sext_err=np.array([ring[ind].PolynomB[1] for ind in indsext])
	ks_sext_err=np.array([ring[ind].PolynomA[1] for ind in indsext])
	kn_sext=-2*b3*(-Dx+xmisal)+kn_sext_err
	ks_sext=-2*b3*(-Dy+ymisal)+ks_sext_err

	# quadrupole rotations
	indquad=ring.get_refpts(at.Quadrupole)
	#print(f'there are {len(indquad)} quadrupoles')

	kn2=np.array([ring[ind].PolynomB[1] for ind in indquad])
	ks2=np.array([ring[ind].PolynomA[1] for ind in indquad])
	srot=np.array([getR1(ring[ind]) for ind in indquad])
	kn_quad=(1-srot)*kn2
	ks_quad=-srot*kn2+ks2 #ks_quad=-srot*kn2+ks2    #############

	# all elements with PolynomB, not sextupoles or quadrupoles
	indPolB=np.array([get_PolynomB(ring,ind) for ind in range(len(ring))])
	#print(f'there are {len(indPolB)} PolynomBs')
	indPolB=indPolB[indPolB!=-1]
	#print(f'there are {len(indPolB)} PolynomBs')
	indPolB=excludeelements(indPolB, indquad)
	#print(f'there are {len(indPolB)} PolynomBs')
	indPolB=excludeelements(indPolB, indsext)

	#print(f'there are {len(indPolB)} PolynomBs')

	NpolB=np.array([len(ring[ind].PolynomB) for ind in indPolB])
	NpolA=np.array([len(ring[ind].PolynomA) for ind in indPolB])

	indPolB=indPolB[NpolB+NpolA>=4]

	kn_all=np.array([ring[ind].PolynomB[1] for ind in indPolB])
	ks_all=np.array([ring[ind].PolynomA[1] for ind in indPolB])


	kn = np.zeros(len(ring))
	ks = np.zeros(len(ring))

	kn[indsext] = kn_sext
	kn[indquad] = kn[indquad] + kn_quad
	kn[indPolB] = kn[indPolB] + kn_all

	ks[indsext] = ks_sext
	ks[indquad] = ks[indquad]  + ks_quad
	ks[indPolB] = ks[indPolB]  + ks_all

	[print(f'quad at sext {c} is nan') for c, k in enumerate(kn_sext) if np.isnan(k)]
	[print(f'quad at quad {c} is nan') for c, k in enumerate(kn_quad) if np.isnan(k)]
	[print(f'quad at all {c} is nan') for c, k in enumerate(kn_all) if np.isnan(k)]

	[print(f'skew at sext {c} is nan') for c, k in enumerate(ks_sext) if np.isnan(k)]
	[print(f'skew at quad {c} is nan') for c, k in enumerate(ks_quad) if np.isnan(k)]
	[print(f'skew at all {c} is nan') for c, k in enumerate(ks_all) if np.isnan(k)]

	indices = np.unique(np.concatenate((indsext, indquad, indPolB)))

	# integrated strengths
	#L=np.array([ring[ind] for inf in indices])

	# restrict to a subset of indexes
	if inds is not None:
		inds1_=[False]*len(ring)
		inds2_=[False]*len(ring)
		for ii in inds:
			inds1_[ii] = True
		for ii in indices:
			inds2_[ii] = True

		indices = [c for c,p in enumerate(np.logical_and(inds1_, inds2_)) if p]

	kn=kn[indices]#*L
	ks=ks[indices]#*L


	for ind in range(len(indsext)):
		ring[indsext[ind]].PolynomB[1] = Qsext[ind]

	return kn, ks, indices


def EquivalentGradientsFromAlignments6D_2(ring, inds=None, inCOD=None):
	"""
	EQUIVALENTGRADIENTSFROMALIGNMENTS6D Estimated normal quad gradients from sext offsets
	[kn,    1) estimated normal quad gradients from sext offsets, quad
		   errors in quadrupoles and sextupoles.
	ks,    2) estimated skew quad gradients from sext offsets, quad
		   errors in quadrupoles, quadrupole rotation.
	ind    3) indexes of locations at wich kn and ks are found
	]=EquivalentGradientsFromAlignments6D(
	r,     1) AT lattice structure with errors
	inCOD  2) closed orbit
	)

	the function finds the closed orbit at sextupoles and converts it into
	equivalent quadrupole and skew quadrupole gradients for the computation
	of skew and normal quadrupole
	RDT quadrupole rotations are also converted in skew quadrupole gradients.
	It returns the complete list of normal (kn) and skew (ks) quadrupole
	gradients at the given indexes (ind) ( not integrated, PolynomB)

	quadrupole and skew quadrupole errors are introduced via COD in
	sextupoles
	"""
	indsext=ring.get_refpts(at.Sextupole)
	#print(f'there are {len(indsext)} sextupoles')
	b3=np.array([ring[ind].PolynomB[2] for ind in indsext])

	_, oin = ring.find_orbit(refpts=indsext,orbit=inCOD)    # orbit at entrance of sextupole DO NOT USE HERE findorbit6Err!
	_, oout = ring.find_orbit(refpts=indsext+1,orbit=inCOD) # orbit at exit of sextupole

	xmisal=np.array([getT1(ring[ind],0) for ind in indsext])
	ymisal=np.array([getT1(ring[ind],2) for ind in indsext])

	Dx=(oout[:,0]+oin[:,0])/2 # orbit average in sextupole
	Dy=(oout[:,2]+oin[:,2])/2 # orbit average in sextupole

	# quadrupole errors in sextupoles
	kn_sext_err=np.array([ring[ind].PolynomB[1] for ind in indsext])
	ks_sext_err=np.array([ring[ind].PolynomA[1] for ind in indsext])
	kn_sext=-2*b3*(-Dx+xmisal)+kn_sext_err
	ks_sext=-2*b3*(-Dy+ymisal)+ks_sext_err

	# quadrupole rotations
	indquad=ring.get_refpts(at.Quadrupole)
	#print(f'there are {len(indquad)} quadrupoles')

	kn2=np.array([ring[ind].PolynomB[1] for ind in indquad])
	ks2=np.array([ring[ind].PolynomA[1] for ind in indquad])
	srot=np.array([getT1(ring[ind],1) for ind in indquad])
	kn_quad=(1-srot)*kn2
	ks_quad=-srot*kn2+ks2

	# all elements with PolynomB, not sextupoles or quadrupoles
	indPolB=np.array([get_PolynomB(ring,ind) for ind in range(len(ring))])
	#print(f'there are {len(indPolB)} PolynomBs')
	indPolB=indPolB[indPolB!=-1]
	#print(f'there are {len(indPolB)} PolynomBs')
	indPolB=excludeelements(indPolB, indquad)
	#print(f'there are {len(indPolB)} PolynomBs')
	indPolB=excludeelements(indPolB, indsext)

	#print(f'there are {len(indPolB)} PolynomBs')

	NpolB=np.array([len(ring[ind].PolynomB) for ind in indPolB])
	NpolA=np.array([len(ring[ind].PolynomA) for ind in indPolB])

	indPolB=indPolB[NpolB+NpolA>=4]

	kn_all=np.array([ring[ind].PolynomB[1] for ind in indPolB])
	ks_all=np.array([ring[ind].PolynomA[1] for ind in indPolB])


	kn = np.zeros(len(ring))
	ks = np.zeros(len(ring))

	kn[indsext] = kn_sext
	kn[indquad] = kn[indquad] + kn_quad
	kn[indPolB] = kn[indPolB] + kn_all

	ks[indsext] = ks_sext
	ks[indquad] = ks[indquad]  + ks_quad
	ks[indPolB] = ks[indPolB]  + ks_all

	[print(f'quad at sext {c} is nan') for c, k in enumerate(kn_sext) if np.isnan(k)]
	[print(f'quad at quad {c} is nan') for c, k in enumerate(kn_quad) if np.isnan(k)]
	[print(f'quad at all {c} is nan') for c, k in enumerate(kn_all) if np.isnan(k)]

	[print(f'skew at sext {c} is nan') for c, k in enumerate(ks_sext) if np.isnan(k)]
	[print(f'skew at quad {c} is nan') for c, k in enumerate(ks_quad) if np.isnan(k)]
	[print(f'skew at all {c} is nan') for c, k in enumerate(ks_all) if np.isnan(k)]

	indices = np.unique(np.concatenate((indsext, indquad, indPolB)))

	# integrated strengths
	#L=np.array([ring[ind] for inf in indices])

	# restrict to a subset of indexes
	if inds is not None:
		inds1_=[False]*len(ring)
		inds2_=[False]*len(ring)
		for ii in inds:
			inds1_[ii] = True
		for ii in indices:
			inds2_[ii] = True

		indices = [c for c,p in enumerate(np.logical_and(inds1_, inds2_)) if p]

	kn=kn[indices]#*L
	ks=ks[indices]#*L


	return kn, ks, indices


def plot_rdts(ring: at.Lattice):
	"""Plot a particle's closed orbit
	Parameters:
	ring:           Lattice object
	Keyword Args:
	refpts (list):   (1,n) array: locations to compute closed orbit
	keep_lattice (bool):   Assume no lattice change since the
	previous tracking. Defaults to :py:obj:`False`
	"""

	def pldata_rdts(ring, ii):

		refpts = np.array(range(len(ring) - 1))

		f1001, f1010, f2000, f0020 = get_rdts(ring, refpts)
		s_pos = ring.get_s_pos(refpts)

		xx = [f1001.real, f1010.real, f2000.real, f0020.real]
		zz = [f1001.imag, f1010.imag, f2000.imag, f0020.imag]

		xlabels = ['f1001.real', 'f1010.real', 'f2000.real', 'f0020.real']
		zlabels = ['f1001.imag', 'f1010.imag', 'f2000.imag', 'f0020.imag']

		left = ('position [m]', s_pos, xx + zz, xlabels + zlabels)

		return 'RDTs', left

	return baseplot(ring, pldata_rdts)




def get_rdts(ring, ind_bpms):
    # get Kn and Ks along ring with errors
    kn, ks, inds = EquivalentGradientsFromAlignments6D(ring)

    # get RDT responses
    #f2000_rdt_resp, f0020_rdt_resp, _ = normal_quad_rdts_response(ring, ind_bpms, inds)
    f1001_rdt_resp, f1010_rdt_resp, _ = skew_quad_rdts_response(ring, ind_bpms, inds)

    # magnets lengths
    Ln = ring.get_value_refpts(inds, 'Length')
    Ls = ring.get_value_refpts(inds, 'Length')
    Ln[Ln == 0] = 1  # thin magnets
    Ls[Ls == 0] = 1

    # integrated gradients
    KnL = np.array(kn * Ln)
    KsL = np.array(ks * Ls)  

    #f1001_rdt_resp = np.delete(f1001_rdt_resp, 15, axis=1)
    #f1010_rdt_resp = np.delete(f1010_rdt_resp, 15, axis=1)
    #f2000_rdt_resp = np.delete(f2000_rdt_resp, 15, axis=1)
    #f0020_rdt_resp = np.delete(f0020_rdt_resp, 15, axis=1)

    #KsL = np.delete(KsL, 15)
    #KnL = np.delete(KnL, 15)

    # compute RDTs
    f1001 = np.array([fr + 1j * fi for fr, fi in zip(f1001_rdt_resp.real @ KsL, f1001_rdt_resp.imag @ KsL)])
    f1010 = np.array([fr + 1j * fi for fr, fi in zip(f1010_rdt_resp.real @ KsL, f1010_rdt_resp.imag @ KsL)])
    #f2000 = np.array([fr + 1j * fi for fr, fi in zip(f2000_rdt_resp.real @ KnL, f2000_rdt_resp.imag @ KnL)])
    #f0020 = np.array([fr + 1j * fi for fr, fi in zip(f0020_rdt_resp.real @ KnL, f0020_rdt_resp.imag @ KnL)])

    return f1001, f1010, 1,1 #f2000, f0020

def _test_rdt():

	import commissioningsimulations.config as config
	import time
	import matplotlib.pyplot as plt


	ring = at.load_mat(config.lattice_file_for_test, mat_key=config.lattice_variable_name)

	start_time = time.time()
	ind_bpms = ring.get_refpts('BPM*')

	f1001, f1010, f2000, f0020 = get_rdts(ring, ind_bpms)

	print(f'rdt computation took: {time.time()-start_time} seconds')

	# plot_rdts(ring)
	# plt.show()

	pass


if __name__ == '__main__':

	_test_rdt()

	pass