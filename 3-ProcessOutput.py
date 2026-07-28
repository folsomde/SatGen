from multiprocessing import Pool, cpu_count
import os 
from glob import glob
import numpy as np
from scipy import optimize as opt, signal as sig
import config as cfg, cosmo as co, profiles as pf, evolve as ev
import logging
logging.basicConfig(format='%(asctime)s | %(name)s %(levelname)s: %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger('read')
log.setLevel('INFO')

def each(index_array):
   return np.indices((len(index_array),), sparse=True)[0], index_array

def read_file(mwID, file):
    log.info(f'reading {file}')
    try:
        with np.load(file) as f:
            redshift = f['redshift']
            t = f['CosmicTime']
            mvir = f['mass']
            order = f['order']
            parent = f['ParentID']
            rvir = f['VirialRadius']
            tidal_radius = f['GreenRte']
            conc = f['concentration']
            coords = f['coordinates']
            stellar_mass = f['StellarMass']
            r_half = f['StellarSize']
    except:
        log.error(f'error loading {file}')
        raise
    Nhalos = len(mvir)
    
    # want to get (1) coordinates in the MW's frame rather than relative to the parent halo
    # and (2) the order-one parent of each halo (i.e. the primary satellite with which a halo fell in)
    #-------
    # convert cylindrical coords to cartesian
    xy_direction = np.array((np.cos(coords[...,1]), np.sin(coords[...,1]))) # indices are [xy, branch, snapshot]
    xy = coords[np.newaxis, ..., 0] * xy_direction # scale this by cylindrical r
    xyz = np.moveaxis(np.array((*xy, coords[...,2])), 0,2) # create [xyz, branch, snapshot] array, reorder to be like normal

    O1_parent = np.copy(parent)
    galcen_pos = np.copy(xyz) 
    for o in range(order.max(), 1, -1):
        # for each high-order satellite
        high_order = order >= o
        iz = np.where(high_order)[1]
        # add to its position the position of its higher-order parent
        galcen_pos[high_order] = galcen_pos[high_order] + xyz[O1_parent[high_order], iz, :]
        # update the higher-order parent to be the next level up (until we reach o == 1, because then all parents are the MW)
        if o != 2: O1_parent[high_order] = parent[O1_parent[high_order], iz]
    # since order-one sats have not had parent updated, need to say they're part of their own group
    O1_parent = np.where(order == 1, np.tile(np.arange(Nhalos), (len(t),1)).T, O1_parent)
    r_of_t = np.linalg.norm(galcen_pos, axis = 2)
    
    # compute pericenter
    bad_snaps = np.isclose(xyz, [1e-3, 0, 0]).all(axis = 2)
    r_msk = np.where(bad_snaps, np.nan, r_of_t)
    branch, peri = sig.argrelmin(r_msk, axis=1)
    peri_snap = np.where(np.isin(np.arange(Nhalos), branch), 
                         peri[np.array([(branch == i).argmax() for i in range(Nhalos)])], -1)
    r_peri = np.where(peri_snap > 0, r_of_t[np.arange(Nhalos), peri_snap], np.nan)
    t_peri = np.where(peri_snap > 0, t[0] - t[peri_snap], np.nan)
    #------ done processing orbits

    host = np.array([((mvir[0,0], conc[0,0], co.DeltaBN(0, cfg.Om, cfg.OL), 0), rvir[0,0])], 
                    dtype = [('Green_params', np.float64, (4,)), ('virial_radius', np.float64)])

    ## other output arrays
    # z = 0 quantities
    z0_mass, z0_mstar, z0_pos, z0_order, z0_parent, z0_rhalf = mvir[:,0],      stellar_mass[:,0],      galcen_pos[:,0],      order[:,0],      O1_parent[:,0],      r_half[:,0]
    # tpeak quantities
    # tp = mvir.argmax(axis = 1)
    tp = np.argmax(conc > 0, axis = 1)
    tp_mass, tp_mstar, tp_pos, tp_order, tp_parent, tp_rhalf = mvir[each(tp)], stellar_mass[each(tp)], galcen_pos[each(tp)], order[each(tp)], O1_parent[each(tp)], r_half[each(tp)]
    tp_time, tp_conc = t[0] - t[tp], conc[each(tp)]
    green_params = list(zip(tp_mass, tp_conc, co.DeltaBN(redshift, cfg.Om, cfg.OL)[tp], redshift[tp]))
    surviving = np.isclose(xyz[:,0], [1e-3, 0, 0]).all(axis = 1)

    #----- update Mstar, rhalf properly
    tp_rmax = 2.163 * rvir[each(tp)]/tp_conc # proportional to rs
    l_eff0_over_l_max0 = tp_rhalf/tp_rmax
    m_within_rmax0, m_within_rmax = [], []
    tp_rmax, tp_vmax = [], []
    z0_rmax, z0_vmax = [], []
    for M_now, MCDz in zip(z0_mass, green_params):
        p = pf.Green(*MCDz)
        tp_rmax.append(p.rmax)
        tp_vmax.append(p.Vcirc(p.rmax))
        m_within_rmax0.append(p.M(p.rmax))
        
        # apply tidal truncation
        p.update_mass(M_now)
        rmax_now = opt.minimize_scalar(lambda r: -p.Vcirc(r), bounds=(0, p.rh), method='bounded').x
        z0_rmax.append(rmax_now)
        z0_vmax.append(p.Vcirc(rmax_now))
        m_within_rmax.append(p.M(rmax_now))
    m_max_ratio = np.array(m_within_rmax)/m_within_rmax0
    l_eff_ratio, m_star_ratio = np.array([ev.g_EPW18(mm, lefflmax = ll) 
                                          for mm, ll in zip(m_max_ratio, l_eff0_over_l_max0)]).T[0]
    z0_mstar, z0_rhalf = m_star_ratio * tp_mstar, l_eff_ratio * tp_rhalf
    
    sat_props = list(zip(np.full(t_peri.shape, mwID), z0_mass[1:], z0_mstar[1:], z0_pos[1:], z0_order[1:], z0_parent[1:]-1, z0_rhalf[1:], z0_rmax[1:], z0_vmax[1:],
                         tp_mass[1:], tp_mstar[1:], tp_pos[1:], tp_order[1:], tp_parent[1:]-1, tp_rhalf[1:], tp_rmax[1:], tp_vmax[1:],
                         tp_time[1:], tp_conc[1:], green_params[1:], 
                         t_peri, r_peri, surviving))
    sats = np.array(sat_props, dtype = [ ('mwID', type(mwID)),
        ('virial_mass', np.float64), ('stellar_mass', np.float64), ('position', np.float64, (3,)), ('order', np.int16), ('groupID', np.float64), ('r_half', np.float64), ('r_max', np.float64), ('v_max', np.float64),
        ('peak_mass', np.float64), ('tpeak_stellar_mass', np.float64), ('tpeak_position', np.float64, (3,)), ('tpeak_order', np.int16), ('tpeak_groupID', np.float64), ('tpeak_r_half', np.float64), ('tpeak_r_max', np.float64), ('tpeak_v_max', np.float64), 
        ('tpeak', np.float64), ('tpeak_conc', np.float64), ('Green_params', np.float64, (4,)), 
        ('tperi', np.float64), ('rperi', np.float64), ('surviving', bool)])
    
    return host, sats

def read_folder(directory, savename, isats = None):
    log.warning(f'searching {directory}')
    all_files = list(sorted(glob(f'{directory}/*.npz')))
    if isats is None:
        isats = np.arange(len(all_files))
    log.warning(f'{len(all_files)} files found, saving {len(isats)}')
    log.warning(f'will save to {savename}')
    Ncores = int(os.getenv('OMP_NUM_THREADS', cpu_count()))
    log.warning(f'{Ncores} cores available for use')
    with Pool(Ncores) as p:  # use as many as requested
        map_output = p.starmap(read_file, [(ifile, all_files[ifile]) for ifile in isats])
    ahosts, asats = zip(*map_output)
    ahosts = np.concatenate(ahosts)
    asats = np.concatenate(asats)
    log.warning(f'saving now')
    np.savez_compressed(savename, hosts = ahosts, sats = asats)
    return ahosts, asats

