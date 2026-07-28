################################ SubEvo #################################

# Program that evolves the subhaloes intialized by TreeGen_Sub.py
# This version of the code is meant to work with the Green model of
# stripped subhalo density profiles.

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016-2017 Hebrew University
# Arthur Fangzhou Jiang 2020 Caltech
# Sheridan Beckwith Green 2020 Yale University
# -- Changed loop order so that redshift is the outermost loop,
#    which enables mass of ejected subhaloes to be removed from
#    the corresponding host; necessary for mass conservation

######################## set up the environment #########################

import config as cfg
import cosmo as co
import evolve as ev
import profiles as pf
from orbit import orbit
import galhalo as gh
import aux

import numpy as np
import sys
import os 
import time 
import glob
from multiprocessing import Pool, cpu_count
from scipy import optimize as opt, stats

# <<< for clean on-screen prints, use with caution, make sure that 
# the warning is not prevalent or essential for the result
import warnings
#warnings.simplefilter('always', UserWarning)
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) 
import logging
logging.basicConfig(format='%(asctime)s | %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('init')

########################### user control ################################

disk_mass_over_mvir = 0.05  # <<< play with, disk mass fraction: no disk potential if 0
disk_radius_over_height = 12.5  # disk scale radius / disk scale height

datadir = os.environ.get('SATGEN_TREES')
# will share parent directory with datadir: same name scheme but replaces the first
# instance of "TREES" with "SAT" (e.g. TREES_NIHAO/ becomes SAT_NIHAO/)
outfolder = os.environ.get('SATGEN_OUTFOLDER', os.path.basename(os.path.normpath(datadir)).replace("TREES", "SAT", 1))



#---stripping efficiency type
alpha_type = 'conc' # 'fixed' or 'conc'

#---dynamical friction strength
cfg.lnL_pref = 0.75 # Fiducial, but can also use 1.0

#---evolution mode (resolution limit in m/m_{acc} or m/M_0)
cfg.evo_mode = 'arbres' # or 'withering'
cfg.phi_res = 10**-4.5 # when cfg.evo_mode == 'arbres',
#                        cfg.phi_res sets the lower limit in m/m_{acc}
#                        that subhaloes evolve down until

########################### evolve satellites ###########################

#---get the list of data files
files = glob.glob(os.path.join(datadir, 'tree*.npz'))
files.sort()
log.warning(f'{len(files)} trees found in {os.path.basename(os.path.normpath(datadir))}')
log.warning(f'Writing to {outfolder}')

# creating output directory
outdir = os.path.join(os.path.dirname(os.path.normpath(datadir)), outfolder)
if not os.path.exists(outdir):
    os.mkdir(outdir)

log.warning('>>> Evolving subhaloes ...')
conc_scatter = float(os.environ.get('SATGEN_CONC_SCATTER',0.12))
log.warning(f'adding {conc_scatter} dex of scatter to concentrations!')

#---
time_start = time.time()
#for file in files: # <<< serial run, only for testing
def loop(file): 
    """
    Replaces the loop "for file in files:", for parallelization.
    """
    log = logging.getLogger(f'{os.path.splitext(os.path.basename(file))[0]}')
    log.setLevel("INFO")

    # skip if we already ran this one and are re-running
    # uncompleted trees on a second pass-through
    outfile = os.path.join(outdir, os.path.basename(file))
    if os.path.exists(outfile) or os.path.exists(outfile + '.part'):
        # if os.path.exists(outfile + '.part'):
        #     log.info('reserved, skipping')
        return
    os.mknod(outfile + '.part')
    log.info('not yet reserved, beginning to process')

    time_start_tmp = time.time()  
    
    #---load trees
    try:
        with np.load(file) as f:
            redshift = f['redshift']
            CosmicTime = f['CosmicTime']
            mass = f['mass']
            StellarMass = f['StellarMass']
            order = f['order']
            ParentID = f['ParentID']
            VirialRadius = f['VirialRadius']
            concentration = f['concentration']
            coordinates = f['coordinates']
    except:
        log.error('this tree had a loading error')
        return

    # add scatter here (2025/06/27)
    concentration_scatter = 10**stats.norm.rvs(size=len(concentration), scale=conc_scatter)
    concentration = np.where(concentration < 0, concentration, concentration * concentration_scatter[:, np.newaxis])

    # compute the virial overdensities for all redshifts
    VirialOverdensity = co.DeltaBN(redshift, cfg.Om, cfg.OL) # same as Dvsample
    GreenRte = np.zeros(VirialRadius.shape) - 99. # contains r_{te} values
    StellarSize = np.zeros(VirialRadius.shape) - 99. # contains r_{1/2} values
    alphas = np.zeros(VirialRadius.shape) - 99.
    tdyns  = np.zeros(VirialRadius.shape) - 99.

    #---identify the roots of the branches
    izroot = mass.argmax(axis=1) # root-redshift ids of all the branches
    idx = np.arange(mass.shape[0]) # branch ids of all the branches
    levels = np.unique(order[order>=0]) # all >0 levels in the tree
    izmax = mass.shape[1] - 1 # highest redshift index

    #---list of potentials and orbits for each branch
    #   additional, mass of ejected subhaloes stored in ejected_mass
    #   to be removed from corresponding host at next timestep
    potentials = [0] * mass.shape[0]
    orbits = [0] * mass.shape[0]
    trelease = np.zeros(mass.shape[0])
    ejected_mass = np.zeros(mass.shape[0])
    rHalf_over_rMax_init = np.zeros(mass.shape[0]) - 99.
    rHalf_init = np.zeros(mass.shape[0]) - 99.
    stellar_mass_init = np.zeros(mass.shape[0]) - 99.
    mass_within_rMax_init = np.zeros(mass.shape[0]) - 99.

    #---list of minimum masses, below which we stop evolving the halo
    M0 = mass[0,0]
    min_mass = np.zeros(mass.shape[0])

    #---evolve
    for iz in np.arange(izmax, 0, -1): # loop over time to evolve
        iznext = iz - 1                
        z = redshift[iz]
        tcurrent = CosmicTime[iz]
        tnext = CosmicTime[iznext]
        dt = tnext - tcurrent
        Dv = VirialOverdensity[iz]

        for level in levels: #loop from low-order to high-order systems
            for id in idx[order[idx, iz] == level]: # loop over branches at this level
                if(iz <= izroot[id]): # halo needs to be orbit-evolved
                    if(iz == izroot[id]): # accretion happens at this timestep
                        # initialize Green profile and orbit

                        za = z
                        ta = tcurrent
                        Dva = Dv
                        ma = mass[id,iz] # initial mass that we will use for f_b
                        c2a = concentration[id,iz]
                        xva = coordinates[id,iz,:]

                        # some edge case produces nan in velocities in TreeGen
                        # if so, print warning and mass fraction lost
                        if(np.any(np.isnan(xva))):
                            log.error(f'NaNs detected in init xv of id {id}. {ma/mass[0, 0]:0.1e} of tree mass is lost.')
                            mass[id,:] = -99.
                            coordinates[id,:,:] = 0.
                            idx = np.delete(idx, np.argwhere(idx == id)[0])
                            # this is an extremely uncommon event, but should
                            # eventually be fixed
                            continue

                        potentials[id] = pf.Green(ma,c2a,Delta=Dva,z=za)
                        orbits[id] = orbit(xva)
                        trelease[id] = ta
                        rHalf = gh.Reff(VirialRadius[id, iz], c2a)
                        rHalf_over_rMax_init[id] = rHalf/potentials[id].rmax
                        mass_within_rMax_init[id] = potentials[id].M(potentials[id].rmax)
                        rHalf_init[id] = rHalf
                        StellarSize[id, iz] = rHalf
                        stellar_mass_init[id] = StellarMass[id, iz]

                        if cfg.evo_mode == 'arbres':
                            min_mass[id] = cfg.phi_res * ma
                        elif cfg.evo_mode == 'withering':
                            min_mass[id] = cfg.psi_res * M0

                    #---main loop for evolution

                    # the p,s,o objects are updated in-place in their arrays
                    # unless the orbit is replaced with a new object when released
                    ip = ParentID[id,iz]
                    p = potentials[ip]
                    s = potentials[id]

                    # update mass of subhalo object based on mass-loss in previous snapshot
                    # we wait to do it until now so that the pre-stripped subhalo can be used
                    # in the evolution of any higher-order subhaloes
                    # We also strip off the mass of any ejected systems
                    # the update_mass function handles cases where we fall below resolution limit
                    if(s.Mh > min_mass[id]):
                        if(ejected_mass[id] > 0):
                            mass[id,iz] -= ejected_mass[id]
                            ejected_mass[id] = 0
                            mass[id,iz] = max(mass[id,iz], cfg.phi_res*s.Minit)

                        # s.update_mass(mass[id,iz])
                        try:
                            s.update_mass(mass[id,iz])
                        except:
                            log.error(f'unable to update mass of {id = }, {iz = }, {mass[id,iz] = :0.2e}')
                            raise
                        rte = s.rte()

                    o = orbits[id]
                    xv = orbits[id].xv
                    m = s.Mh
                    
                    stellar_mass = StellarMass[id, iz]
                    rHalf = StellarSize[id, iz]
                    
                    m_old = m
                    r = np.sqrt(xv[0]**2+xv[2]**2)

                    #---time since in current host
                    t = tnext - trelease[id]

                    # Order should always be one higher than parent unless 
                    # ejected,in which case it should be the same as parent
                    k = order[ip,iznext] + 1

                    # alpha: stripping efficiency
                    if(alpha_type == 'fixed'):
                        alpha = 0.55
                    elif(alpha_type == 'conc'):
                        if type(p) is list:
                            c2p = mw_conc_w_disk
                        else:
                            c2p = p.ch
                        alpha = ev.alpha_from_c2(c2p, s.ch)

                    #---evolve satellite
                    # as long as the mass is larger than resolution limit
                    if m > min_mass[id]:

                        # evolve subhalo properties
                        m,lt = ev.msub(s,p,xv,dt,choice='King62',
                            alpha=alpha)
                        
                        # evolve baryon properties
                        m_within_rMax = s.M(s.rmax)
                        g_rhalf, g_ms = ev.g_EPW18(m_within_rMax/mass_within_rMax_init[id], 1, rHalf_over_rMax_init[id])
                        rHalf = g_rhalf[0] * rHalf_init[id]
                        stellar_mass = g_ms[0] * stellar_mass_init[id]

                    else: # we do nothing about disrupted satellite, s.t.,
                        # its properties right before disruption would be 
                        # stored in the output arrays
                        pass

                    #---evolve orbit
                    if m > min_mass[id]:
                        # NOTE: We previously had an additional check on r>Rres
                        # here, where Rres = 10^-3 Rvir(z), but I removed it
                        # All subhalo orbits are evolved until their mass falls
                        # below the resolution limit.
                        # NOTE: No use integrating orbit any longer once the halo
                        # is disrupted, this just slows it down
                    
                        tdyn = pf.tdyn(p,r)
                        o.integrate(t,p,m_old)
                        xv = o.xv # note that the coordinates are updated 
                        # internally in the orbit instance "o" when calling
                        # the ".integrate" method, here we assign them to 
                        # a new variable "xv" only for bookkeeping
                        
                    else: # i.e., the satellite has merged to its host, so
                        # no need for orbit integration; to avoid potential 
                        # numerical issues, we assign a dummy coordinate that 
                        # is almost zero but not exactly zero
                        tdyn = pf.tdyn(p,cfg.Rres)
                        xv = np.array([cfg.Rres,0.,0.,0.,0.,0.])

                    r = np.sqrt(xv[0]**2+xv[2]**2)
                    m_old = m


                    #---if order>1, determine if releasing this high-order 
                    #   subhalo to its grandparent-host, and if releasing,
                    #   update the orbit instance
                    if k>1:
                    
                        if (r > VirialRadius[ip,iz]) & (iz <= izroot[ip]): 
                            # <<< Release condition:
                            # 1. Host halo is already within a grandparent-host
                            # 2. Instant orbital radius is larger than the host
                            # TIDAL radius (note that VirialRadius also contains
                            # the tidal radii for the host haloes once they fall
                            # into a grandparent-host)
                            # 3. (below) We compute the fraction of:
                            #             dynamical time / alpha
                            # corresponding to this dt, and release with
                            # probability dt / (dynamical time / alpha)

                            # Compute probability of being ejected
                            odds = np.random.rand()
                            dyntime_frac = alphas[ip,iz] * dt / tdyns[ip,iz]
                            if(odds < dyntime_frac):
                                if(ParentID[ip,iz] == ParentID[ip,iznext]):
                                    # host wasn't also released at same time
                                    # New coordinates at next time are the
                                    # updated subhalo coordinates plus the updated
                                    # host coordinates inside of grandparent
                                    xv = aux.add_cyl_vecs(xv,coordinates[ip,iznext,:])
                                else:
                                    xv = aux.add_cyl_vecs(xv,coordinates[ip,iz,:])
                                    # This will be extraordinarily rare, but just
                                    # a check in case so that the released order-k
                                    # subhalo isn't accidentally double-released
                                    # in terms of updated coordinates, but not
                                    # in terms of new host ID.
                                orbits[id] = orbit(xv) # update orbit object
                                k = order[ip,iz] # update instant order to the same as the parent
                                ejected_mass[ip] += m 
                                # add updated subhalo mass to a bucket to be removed from host
                                # at start of next timestep
                                ip = ParentID[ip,iz] # update parent id
                                trelease[id] = tnext # update release time

                    #---update the arrays for output
                    mass[id,iznext] = m
                    StellarMass[id, iznext] = stellar_mass
                    StellarSize[id, iznext] = rHalf

                    order[id,iznext] = k
                    ParentID[id,iznext] = ip
                    try:
                        VirialRadius[id,iznext] = lt # storing tidal radius
                    except UnboundLocalError:
                        # TreeGen gives a few subhaloes with root mass below the
                        # given resolution limit so some subhaloes will never get
                        # an lt assigned if they aren't evolved one step. This can
                        # be fixed by lowering the resolution limit of SubEvo
                        # relative to TreeGen by some tiny epsilon, say 0.05 dex
                        print("No lt for id ", id, "iz ", iz, "masses ",
                              np.log10(mass[id,iz]), np.log10(mass[id,iznext]), file)
                        return

                    # NOTE: We store tidal radius in lieu of virial radius
                    # for haloes after they start getting stripped
                    GreenRte[id,iz] = rte 
                    coordinates[id,iznext,:] = xv

                    # NOTE: the below two are quantities at current timestep
                    # instead, since only used for host release criteria
                    # This won't be output since only used internally
                    alphas[id,iz] = alpha
                    tdyns[id,iz] = tdyn

                else: # iz > izroot; before accretion. Halo is an NFW profile
                    assert (concentration[id,iz] > 0) 
                    if id == 0: # need to add in disk potential for MW
                        # fiducial disk parameters from SatGen II paper, Green (2022)
                        f_a, b_over_a, beta_a, f_M, beta_M = 0.0125, 0.08, 1/3, 0.05, 1
                        a_disk = f_a *(mass[0,iz]/mass[0,0])**beta_a * VirialRadius[0,0]
                        b_disk = b_over_a * a_disk
                        M_disk = f_M * mass[0,iz] # NOT GENERAL: assumes beta_M = 1
                        # save these potentials to the big list
                        mw_halo = pf.NFW(mass[0,iz] - M_disk, concentration[id,iz],
                                         Delta=VirialOverdensity[iz],z=redshift[iz])
                        disk_potential = pf.MN(M=M_disk, a=a_disk, b=b_disk)
                        potentials[0] = [mw_halo, disk_potential]
                        # to account for disk's affect on concentration,
                        # find a new scale radius with the same enclosed mass
                        mass_within_rs = np.log10(mw_halo.M(mw_halo.rs))
                        mass_diff = lambda r: np.log10(pf.M(potentials[0], r)) - mass_within_rs
                        disk_rs = opt.brentq(mass_diff, mw_halo.rs * 1e-3, mw_halo.rs)
                        mw_conc_w_disk = mw_halo.rh/disk_rs
                    else: # just an NFW out in the field
                        potentials[id] = pf.NFW(mass[id,iz],concentration[id,iz],
                                                Delta=VirialOverdensity[iz],z=redshift[iz])

    #---output
    np.savez(outfile, 
        redshift = redshift,
        CosmicTime = CosmicTime,
        mass = mass,
        order = order,
        ParentID = ParentID,
        VirialRadius = VirialRadius,
        GreenRte = GreenRte,
        # this contains values during stripping, -99 prior to stripping and
        # once the halo falls below the resolution limit
        concentration = concentration, # this is unchanged from TreeGen output
        coordinates = coordinates,
        StellarMass = StellarMass,
        StellarSize = StellarSize
        )
    
    #---on-screen prints
    m0 = mass[:,0][1:]
    
    msk = (m0 > cfg.psi_res*M0) & (m0 < M0) & order[1:,0] == 1
    fsub = m0[msk].sum() / M0
    
    MAH = mass[0,:]
    iz50 = aux.FindNearestIndex(MAH,0.5*M0)
    z50 = redshift[iz50]
    
    time_end_tmp = time.time()
    log.info(f'{(time_end_tmp-time_start_tmp)/60:.2f} min, {z50 = :.2f}, {fsub = :.5f}')
    os.remove(outfile + '.part')

#---for parallelization, comment for testing in serial mode
if __name__ == "__main__":
    # read OMP_NUM_THREADS, but default to cpu_count() 
    # if the environment variable is undefined
    Ncores = int(os.getenv('OMP_NUM_THREADS', cpu_count()))
    print(f'>>> {Ncores} cores available for use')
    with Pool(Ncores) as pool:  # use as many as requested
        # pool.map(loop, np.random.permutation(files), 1)
        # pool.map(loop, reversed(files), 1)
        pool.map(loop, files, 1)

time_end = time.time() 
print('    total time: %5.2f hours'%((time_end - time_start)/3600.))
