# Get exposure-weighted off-axis angle 
# for multiple observations

import numpy as np
import sys

def sph_distance(ra1s, dec1s, ra2s, dec2s):
    '''
    Input: radec (decimal deg) of the two sources \
           (or two lists of soruces)
    See the figure above to understand
        the algorithm
    '''
    # Convert variables to arrays
    ra1s = np.array(ra1s)
    dec1s = np.array(dec1s)
    ra2s = np.array(ra2s)
    dec2s = np.array(dec2s)
    
    # Get b, c, A and convert to rad from deg
    rad_to_deg = np.pi / 180.
    cs = (90. - dec1s) * rad_to_deg
    bs = (90. - dec2s) * rad_to_deg
    As = np.abs(ra1s-ra2s) * rad_to_deg
    # Calculate a, and convert to arcsec
    cos_as = np.cos(bs)*np.cos(cs) + np.sin(bs)*np.sin(cs)*np.cos(As)
    # To prevent floating errors
    cos_as[np.where(cos_as>1)]=1
    cos_as[np.where(cos_as<-1)]=-1
    As = np.arccos(cos_as) / rad_to_deg * 3600
    # a is in arcsec
    return As

def get_off_ang(src_ras, src_decs, pt_ras, pt_decs, exps, verbose=False):
    '''
    Input:
        src_ras, src_decs, 
            source positions in deg
        pt_ras, pt_decs, 
            pointing posigions in deg
        exps, 
            exposure time the pointings
    Output:
        avg_angs, 
            exposure-weighted off-axis angle
    '''    
    # Array to store results
    avg_angs = []
    # Iterate over each sources
    for src_idx in range(len(src_ras)):
        # Compute distance to the pointings
        # convert arcsec to arcmin
        ds = sph_distance(src_ras[src_idx], src_decs[src_idx], pt_ras, pt_decs)/60.
        # Apply the angular cut (8')
        use_idxs = np.where( ds<8. )[0]
        # Check if no coverage at all
        if verbose:
            if len(use_idxs)==0: 
                print("warning: src=%d is not covered by any observations; set angle=min_angle=%.1f" \
                      %(src_idx, np.min(ds)) )
                avg_angs.append(np.min(ds))
                continue
            # Get exp-weighted off-axis angle
            avg_ang = np.average(ds[use_idxs], weights=exps[use_idxs])
            avg_angs.append(avg_ang)
        else:
            if len(use_idxs)==0: 
                avg_angs.append(np.min(ds))
                continue
            # Get exp-weighted off-axis angle
            avg_ang = np.average(ds[use_idxs], weights=exps[use_idxs])
            avg_angs.append(avg_ang)            
    return np.array(avg_angs)

def get_chandra_eef(thetas, R0=1.32, R10=10.1, alpha=2.42):
    '''
    Inputs:
        thetas: the off-axis angle (') array
    Keywords:
        R0, EEF radius (") for off-axis angle=0
            default=1.07 (90% EEF, from Table A1 of Vito+16)
        R10, EEF radius (") for off-axis angle=0
            default=9.65 (90% EEF, from Table A1 of Vito+16)
    Output:
        Rs, EEF radia from thetas
    '''
    # Creat the EEF array
    Rs = np.zeros(len(thetas))-99.
    # Get sources with positive off-axis angle
    use_idxs = np.where(thetas>=0)[0]
    # Derive EEF
    Rs[use_idxs] = R0 + R10*(thetas[use_idxs]/10.)**alpha
    # Check if all sources have positive off-aixs angle
    num_bad = len(thetas)-len(use_idxs)
    if num_bad>0:
        print("warning: %d sources are not calculated due to negative off-axis angle" %num_bad)
    return Rs

def get_pix_in_annulus(cent_x, cent_y, r, R):
    '''
    Get pixel coordinates in a given annulus
    Inputs:
        cent_x, cent_y, the center of the annulus
        r, R, inner and outer radius
    Outputs:
        the x,y coordinate of pixels
                that are within the annulus
    '''
    # Get the pixels whtin the big square
    # Set the x-y grids
    x_grids, y_grids = np.arange( np.floor(cent_x-R), np.ceil(cent_x+R)+0.1 ), \
                       np.arange( np.floor(cent_y-R), np.ceil(cent_y+R)+0.1 )
    # mesh grids
    xs_all, ys_all = np.meshgrid(x_grids, y_grids)
    xs_all, ys_all = xs_all.flatten().astype(int), ys_all.flatten().astype(int)
    # Calculate distances to the center
    Rs_all = ( (xs_all-cent_x)**2 + (ys_all-cent_y)**2  )**0.5
    # Filter out outside and inside pixels
    use_idxs = np.where( (Rs_all<=R) & (Rs_all>r) )[0]    
    return (xs_all[use_idxs], ys_all[use_idxs])

def get_pix_in_circle(cent_x, cent_y, R):
    '''
    Get pixel coordinates in a given cirvle
    Input:
        cent_x, cent_y, the center of the circle
        R, the radius of the circle
    Output:
        (xs,ys), the x,y coordinate of pixels
                that are within the circle
    '''
    # Use annulus but with negative inner radius
    pix_idxs = get_pix_in_annulus(cent_x, cent_y, -1, R)
    return pix_idxs
