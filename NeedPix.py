import numpy as np
from scipy.integrate import simps
import matplotlib.pyplot as plt
import healpy as hp
import argparse, os, sys, warnings, glob
import cython_mylibc as mylibc # Needlet library
import cPickle as pickle

class NeedTheory(object):
	""" 
	Class to compute theoretical quantities related to needlets, such as
	the filter function, needlet power spectra, their variance and so on. 
	"""
	def __init__(self, B, npoints=1000):
		""" 
		Parameters
		----------
		B : float
			needlet width parameter

		npoints : float 
			# of points to sample the integrals
		"""
		self.B = B
		self.npoints = npoints
		self.norm = self.get_normalization()

	def f_need(self, t):
		"""
		Standard needlets f function
		@see arXiv:0707.0844

		Notes
		----- 
		* this *vectorized* version works only for *arrays*
		"""
		good_idx = np.logical_and(-1. < t, t < 1.)
		f1 = np.zeros(len(t))
		f1[good_idx] = np.exp(-1./(1.-(t[good_idx]*t[good_idx])))
		return f1

	def get_normalization(self):
		"""
		Evaluates the normalization of the standard needlets function
		@see arXiv:0707.0844
		"""
		t = np.linspace(-1,1,self.npoints)
		return simps(self.f_need(t), t)

	def psi_need(self, u):
		"""
		Standard needlets Psi function
		@see arXiv:0707.0844
		"""
		# u_ = np.linspace(-1.,u,self.npoints)
		return [simps(self.f_need(np.linspace(-1.,u_,self.npoints)), np.linspace(-1.,u_,self.npoints))/self.norm for u_ in u]

	def phi_need(self, t):
		"""
		Standard needlets Phi function
		@see arXiv:0707.0844
		"""
		left_idx = np.logical_and(0 <= t, t <= 1./self.B)
		cent_idx = np.logical_and(1./self.B < t, t < 1.)
		rite_idx = t > 1.

		phi = np.zeros(len(t))
		phi[left_idx] = 1.
		phi[cent_idx] = self.psi_need(1.-2.*self.B/(self.B-1.)*(t[cent_idx]-1./self.B))
		phi[rite_idx] = 0.

		return phi

	def b_need(self, xi):
		"""
		Standard needlets windows function
		@see arXiv:0707.0844
		"""
		# print np.sqrt(np.abs(self.phi_need(xi/self.B)-self.phi_need(xi)))

		return np.sqrt(np.abs(self.phi_need(xi/self.B)-self.phi_need(xi)))

	def cl2betaj(self, jmax, cl):
		"""
		Returns needlet power spectrum \beta_j given an angular power spectrum Cl.

		Parameters
		----------
		jmax : float
			maximum needlet frequency

		cl   : array-like
			angular power spectrum to convert to needlet power spectrum \beta_j
		"""
		assert(np.floor(self.B**(jmax+1)) <= cl.size-1) 
		print np.floor(self.B**(jmax+1)), cl.size-1
 
		betaj = np.zeros(jmax+1)
		for j in xrange(jmax+1):
			lmin = np.floor(self.B**(j-1))
			lmax = np.floor(self.B**(j+1))
			ell  = np.arange(lmin, lmax, dtype=np.int)
			b2   = self.b_need(ell/self.B**j)*self.b_need(ell/self.B**j)
			betaj[j] = np.sum(np.nan_to_num(b2)*(2.*ell+1.)/4./np.pi*cl[ell])
		return np.nan_to_num(betaj)

	def norm_spectrum(self, jmax, lmax=3000):
		"""
		Returns the normalization of the auto/cross needlet power spectrum,
		@see Eq. (2.18) of Bianchini+16, astro-ph:1607.05223

		Parameters
		----------
		jmax : float
			maximum needlet frequency
		"""
		return self.cl2betaj(jmax, np.ones(lmax+1))

	def varj(self, jmax, cl):
		"""
		Returns the variance of the auto/cross needlet power spectrum,
		@see Eq. (2.19) of Bianchini+16, astro-ph:1607.05223

		Parameters
		----------
		jmax : float
			maximum needlet frequency

		cl : array-like ([lmax+1] or [3, lmax+1])
			input angular power spectrum

		Notes
		-----
		* if 1d array -> cl = C_{\ell}^{XX}
		* if 2d array -> cl = [C_{\ell}^{XY}, C_{\ell}^{XX}, C_{\ell}^{YY}]
		"""
		lmax = len(np.atleast_2d(cl)[0]) - 1
		assert(np.floor(self.B**(jmax+1)) <= lmax) 
		assert((len(np.atleast_2d(cl)) == 1) or (len(np.atleast_2d(cl)) == 3))

		varj = np.zeros(jmax+1)

		for j in xrange(jmax+1):
			lmin = np.floor(self.B**(j-1))
			lmax = np.floor(self.B**(j+1))
			ell  = np.arange(lmin, lmax, dtype=np.int)
			# print ell
			b2   = self.b_need(ell/self.B**j)*self.b_need(ell/self.B**j)
			b4   = b2*b2
			if len(np.atleast_2d(cl)) == 1:
				varj[j] = np.sum(2*np.nan_to_num(b4)*(2.*ell+1.)/16./np.pi**2*cl[ell])
			else:
				varj[j] = np.sum(np.nan_to_num(b4)*(2.*ell+1.)/16./np.pi**2*(cl[0][ell]**2 + cl[1][ell]*cl[2][ell]))
		return np.nan_to_num(varj)

	def get_Mll(self, wl, lmax=None):
		"""
		Returns the Coupling Matrix M_ll from l = 0 (Hivon et al. 2002)

		Parameters
		----------
		wl : array-like
			angular power spectrum of the mask

		Notes
		-----
		* M_ll.shape = (lmax+1, lmax+1)
		"""
		try:
			from mll import mll
		except:
			print("...mll.so not found...")
			print("!!!CANNOT COMPUTE M_ll!!!")
			return None

		if lmax == None:
			lmax = wl.size-1
		assert(lmax <= wl.size-1)
		return np.float64(mll.get_mll(wl[:lmax+1], lmax))

	def gammaJ(self, cl, wl, jmax, lmax):
		"""
		Returns the theoretical needlet-pseudo spectrum \Gamma_j vector, 
		@see Eq. (3.3) of Bianchini+16, astro-ph:1607.05223

		Parameters
		----------
		cl : array-like
			input angular power spectrum

		wl : array-like
			angular power spectrum of the mask

		jmax : float
			maximum needlet frequency

		lmax : float
			maximum multipole

		Notes
		-----
		gamma_lj.shape = (lmax+1, jmax+1)
		"""
		Mll = self.get_Mll(wl, lmax=lmax)
		if Mll is None:
			print("!!!CANNOT COMPUTE \Gamma_j!!!")
		ell = np.arange(0, lmax+1, dtype=np.int)
		bjl = np.zeros((jmax+1,lmax+1))

		for j in xrange(jmax+1):
			b2 = self.b_need(ell/self.B**j)**2
			b2[np.isnan(b2)] = 0.
			bjl[j,:] = b2*(2*ell+1.) 

		return np.dot(bjl, np.dot(Mll, cl[:lmax+1]))/4./np.pi

class NeedAnalysis(object):
	"""
	Class to perform needlet analysis (based on Alessandro Renzi's needlet library). 
	@see arXiv:0707.0844	
	"""	
	def __init__(self, jmax, lmax, norm=True):
		"""
		Parameters
		----------
		jmax : int
		    Maximum needlet frequency

		lmax : int
			Maximum multipole value

		norm : boolean
			If True, returns the normalized needlet power spectra 
		"""

		self.jmax  = jmax
		self.lmax  = lmax

		# Initialize Needlet library
		print("...Initializing Needlet library...")
		self.B = mylibc.jmax_lmax2B(self.jmax, self.lmax)
		
		print("==>lmax={:d}, jmax={:d}, B={:e}".format(self.lmax, self.jmax, self.B))
		self.b_values = mylibc.needlets_std_init_b_values(self.B, self.jmax, self.lmax)
		mylibc.needlets_check_windows(self.jmax, self.lmax, self.b_values)
		self.jvec = np.arange(self.jmax+1)

		if norm:
			needth = NeedletTheory(self.B)
			self.norm = needth.norm_spectrum(self.jmax)
		else:
			self.norm = np.ones(self.jvec.size)

		print("...done...")

	def Betajk2Betaj(self, betajk1, betajk2=None):#, mask=None):
		"""
		Returns the needlet (auto- or cross-) power spectrum \beta_j given the needlet coefficients.

		Parameters
		----------
		betajk1 : array [jmax, npix]
			Map 1 needlet coefficients 

		betajk2 : array [jmax, npix]
			Map 2 needlet coefficients 

		mask : array
			(Healpix) Mask to be applied to each needlet frequency map
		"""
		if betajk2 is None: # auto-spectrum
			betajk2 = betajk1.copy()
		else:
			betaj = np.mean( betajk1[:self.jmax+1,:] * betajk2[:self.jmax+1,:], axis=1 )

		return np.nan_to_num(betaj/self.norm)
	
	def Map2Betaj(self, map1, map2=None, mask=None, noise=0., pseudo=False):
		"""
		Returns the needlet (auto- or cross-) power spectrum \beta_j given Healpix maps.

		Parameters
		----------
		map1 : Healpix map
			Map 1

		map2 : Healpix map
			Map 2 

		mask : Healpix map
			(binary) Mask applied to each map 

		noise : float or array-like
			noise needlet power spectrum

		pseudo : boolean
			if False, extracted spectrum is rescaled for f_sky factor

		Returns
		-------
		betaj : array
			Array w/ shape (jmax+1) containing auto(cross) needlet power spectrum

		Notes
		-----
		* in case of cross-spectrum the mask is applied to *both* maps
		* in case of masked sky, noise bias is *not* divided by f_sky factor
		"""
		assert ( (np.atleast_1d(noise).size == 1) or (np.atleast_1d(noise).size == self.jmax+1) ) 

		fsky = 1.
		
		if mask is not None:
			map1 *= mask 
			if not pseudo:
				fsky = np.mean(mask**2) 

		
		map1 = hp.remove_dipole(map1, verbose=False)#.compressed()
		
		betajk1 = mylibc.needlets_f2betajk_healpix_harmonic(map1, self.B, self.jmax, self.lmax)

		if map2 is None: # Auto-
			return self.Betajk2Betaj(betajk1)/fsky - noise
		else: # Cross-
			if mask is not None:
				map2 *= mask 
			map2 = hp.remove_dipole(map2, verbose=False)#.compressed()
			betajk2 = mylibc.needlets_f2betajk_healpix_harmonic(map2, self.B, self.jmax, self.lmax)
			return self.Betajk2Betaj(betajk1, betajk2=betajk2)/fsky

	def GetBetajPlanckLensSims(self, nsim, fix_field=None, mask=None, fname=None, noise=0., pseudo=False, 
									path_planck_sims='/Volumes/SAMSUNG/Work/SimMap_Lensing/Planck_data_release_2/sims/obs_klms/',
									nside=512, pixwin=True):
		"""
		Evaluates needlet (auto- or cross-) power spectrum for simulated Planck lensing maps. 

		Parameters
		----------
		nsim  : int
			Number of simulations to be analyzed

		fix_field : str or Healpix map
			Map to cross-correlate with nsim simulated Planck lensing maps. This is for *null-tests*.

		mask  : Healpix map
			(binary) Mask applied to each map

		fname : str
			pkl filename where to store the output

		noise : float or array-like
			noise needlet power spectrum

		path_planck_sims : str
			path where the Planck sims are stored

		nside : int
			Healpix resolution at which Planck sims maps are created

		pixwin : boolean
			if True the Planck sims are convolved for the pixel window function

		Returns
		-------
		betaj_sims : array
			array with ((nsim, jmax+1)) shape containing needlet coefficients 
		"""
		try:
			from mpi4py import MPI
			comm = MPI.COMM_WORLD
			myid, nproc = comm.Get_rank(), comm.Get_size()
		except ImportError:
			myid, nproc = 0, 1

		try:
			results = pickle.load(open(fname, 'rb'))
			if myid == 0: print("...Beta_J Sims " + fname + " found...")
		except:
			if myid == 0: print("...evaluating Beta_j Sim...")
			if myid == 0: print("...fix_field-->", fix_field)

			if fix_field is not None:
				if type(fix_field) == str:
					fix_field = hp.read_map(fix_field)
				assert( hp.isnpixok(len(fix_field)) )				

			dim = (nsim) / nproc
			if nproc > 1:
				if myid < (nsim) % nproc:
					dim += 1

			betaj_sims = np.zeros((dim, self.jmax+1))

			# Cycle over simulations 
			k = 0
			if fix_field is None: # Auto-spectra of Planck lensing maps
				for n in xrange(myid, nsim, nproc): 
					kappa_sim_lm 	 = hp.read_alm(path_planck_sims + 'sim_%04d_klm.fits' %n)
					kappa_sim    	 = hp.alm2map(kappa_sim_lm, nside=nside, pixwin=pixwin)
					betaj        	 = self.Map2Betaj(kappa_sim, mask=mask, noise=noise, pseudo=pseudo)
					betaj_sims[k, :] = betaj
					k += 1
			else: # (Null-test)
				for n in xrange(myid, nsim, nproc): 
					kappa_sim_lm     = hp.read_alm(path_planck_sims + 'sim_%04d_klm.fits' %n)
					kappa_sim        = hp.alm2map(kappa_sim_lm, nside=nside, pixwin=pixwin)
					betaj            = self.Map2Betaj(kappa_sim, map2=fix_field, mask=mask, pseudo=pseudo)
					betaj_sims[k, :] = betaj
					k += 1

			assert (k == dim)

			if nproc > 1:
				betaj_sims_tot = comm.gather(betaj_sims, root=0)
				if myid == 0:
				    betaj_sims_tot = np.vstack((_sims for _sims in betaj_sims_tot)) 
			else:
				betaj_sims_tot = betaj_sims

			if myid == 0: 
				results = {}
				results['betaj_sims'] = np.nan_to_num(betaj_sims_tot)
				results['j']          = self.jvec
				results['B']          = self.B
				results['lmax']       = self.lmax
				results['betaj_mean'] = np.mean(betaj_sims_tot, axis=0) 
				results['betaj_cov']  = np.cov(betaj_sims_tot.T) 
				results['betaj_corr'] = np.corrcoef(betaj_sims_tot.T) 
				results['betaj_err']  = np.sqrt(np.diag(results['betaj_cov'])) 

				if fname is not None:
					print("...evaluation terminated...")
					print("...saving to output " + fname + "...")
					pickle.dump(results, open(fname, 'wb'))
					# if nproc > 1:
					#     comm.Barrier()
		
		return results

	# def GetDjkFromMaps(self, field, mask, nsim, fname=None):
	# 	"""
	# 	Returns MC averaged difference between needlet coefficients w/ and w/o masking applied.
	# 	@see arXiv:0707.0844 (Eq. 7)

	# 	Parameters
	# 	----------
	# 	field : str
	# 		Cosmic field to be analyzed => 'kappaT, kappaN, kappaS, deltaT, deltaS'
	# 	mask  : Healpix map
	# 		(binary) Mask applied to each map
	# 	nsim  : int
	# 		Number of simulations to be analyzed
	# 	fname : str
	# 		File name to store the array. 

	# 	Returns
	# 	-------
	# 	Djk : array
	# 		array with ((jmax+1, npix)) shape containing Djk quantity.
	# 	"""
		
	# 	try:
	# 		Djk = np.load(self.OutDir+fname)
	# 		print("...Djk array " + self.OutDir + fname + " found...")
	# 	except:
	# 		# print("...simulations beta_jk's " + fname + " not found...")
	# 		print("...evaluating Djk array...")

	# 		delta2_betajk_mean = np.zeros((self.jmax+1, hp.nside2npix(self.Sims.SimPars['nside']))) 
	# 		betajk2_mean       = np.zeros((self.jmax+1, hp.nside2npix(self.Sims.SimPars['nside'])))

	# 		for n in xrange(0,nsim):
	# 			m = self.Sims.GetSimField(field, n)
	# 			betajk      = mylibc.needlets_f2betajk_healpix_harmonic(m, self.B, self.jmax, self.lmax)
	# 			betajk_mask = mylibc.needlets_f2betajk_healpix_harmonic(m*mask, self.B, self.jmax, self.lmax)
	# 			delta2_betajk_mean += (betajk_mask - betajk)**2
	# 			betajk2_mean       += betajk**2
	# 			del betajk, betajk_mask

	# 		delta2_betajk_mean /= nsim
	# 		betajk2_mean       /= nsim

	# 		Djk = delta2_betajk_mean/betajk2_mean

	# 		if fname is not None:
	# 			print("...evaluation terminated...")
	# 			np.save(self.OutDir + fname, Djk)
	# 			print("...saved to output " + self.OutDir + fname + "...")

	# 	return Djk

	# def GetDjkThresholded(Djk, j, mask=None, threshold=0.1):
	# 	"""
	# 	Evaluates *mean* needlet (auto- or cross-) power spectrum for simulated maps of a given field. 

	# 	Parameters
	# 	----------
	# 	Djk : array
	# 		Array with ((jmax+1, npix)) shape containing Djk quantity.

	# 	j : int
	# 		Needlet frequency

	# 	mask  : Healpix map
	# 		(binary) Mask applied to each map

	# 	threshold : float
	# 		Cut-off to select pixels with Djk > threshold

	# 	Returns
	# 	-------
	# 	m_cut : Healpix map
	# 		Binary map (i.e. pixels = 0 or 1) with 1 where pixel > threshold
	# 	"""
	# 	m = Djk[j,:]
		
	# 	if mask is not None:
	# 		m *= mask
		
	# 	m_cut = np.zeros(m.size)
	# 	m_cut[m > threshold] = 1.
		
	# 	return m_cut
