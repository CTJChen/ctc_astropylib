import numpy as np
from astropy.io import fits
from astropy import wcs
from astropy.table import Table as tab
import argparse
import os, sys
class Parser(argparse.ArgumentParser):
    '''An interactive error handler.
    '''
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        sys.stderr.write('use the -h option\" for helps\n')
        sys.exit(2)

def add_column(hdu, coldata, colname, formatstr, unit, overwrite,
    coord_type=None, coord_unit=None, coord_ref_point=None,
    coord_ref_value=None, coord_inc=None,
    verbose=False):
    """A wrapper of astropy fits operations that add a column to the given hdu

    Mandatory inputs:
    hdu - an astroy HDUList object.
    coldata - an array of data to be added to HDU, should be a numpy array.
    colname - name of the new column to be added.
    formatstr - fits column formatter string.
    unit - fits column unit string.
    overwrite - set to true to overwrite the existing column.

    Optional inputs:
    See https://docs.astropy.org/en/stable/io/fits/api/tables.html#astropy.io.fits.Column
    for the details of the following optional inputs:
    coord_type, coord_unit, coord_ref_point, coord_ref_value, coord_inc.
    verbose - a verbosity switch. Default:False

    Output:
    A fits hdu list object with the new column.

    Unit test: A column of all ones is added to an event list, and we verify the presence of the column, that its minimum/maximum values are both equal to 1, and that it has the name "ONE"
    """
    newhdu = fits.BinTableHDU(data=hdu.data.copy())
    table = newhdu.data
    if colname in table.columns.names and overwrite:
        if verbose:print('deleting existing column')
        table.columns.del_col(colname)
    elif colname in table.columns.names:
        sys.exit('Column already exisits. Please set overwrite=True')

    col = fits.Column(name=colname, format=formatstr, unit=unit,
            coord_type=coord_type, coord_unit=coord_unit, coord_ref_point=coord_ref_point,
            coord_ref_value=coord_ref_value, coord_inc=coord_inc,
        array=coldata)
    # NOTE: append the new time column to the *last*!
    # Otherwise the TLMIN??/TLMAX?? keyword pairs, which record the
    # minimum/maximum values of corresponding columns, will have
    # a different order. Therefore the output FITS file can cause weird problems
    # with DS9 and DM tools.
    newtable = fits.BinTableHDU.from_columns(
            table.columns + fits.ColDefs([col]), header=hdu.header)
    newhdu.data = newtable.data
    # update header
    newhdu.header.update(newtable.header)
    return newhdu



def main(args):
    filename = args.file
    if len(args.out) == 0:
        outname = filename.replace('.evt','.fits')
    hdu = fits.open(filename)
    evttab = tab(hdu[1].data)
    header = hdu[1].header
    headerkeys = [k for k in header.keys()]
    if 'RA_PNT' in headerkeys:
        racen = headerkeys['RA_PNT']
    elif args.ra > 0:
        racen = args.ra
    else:
        print('No RA_PNT, nor -ra argument, taking the median RA in events as RA_PNT')
        racen = np.median(hdu[1].data['RA'])
    if 'DEC_PNT' in headerkeys:
        deccen = headerkeys['DEC_PNT']
    elif np.abs(args.dec) <= 90.:
        deccen = args.dec
    else:
        print('No DEC_PNT, nor -dec argument, taking the median DEC in events as RA_PNT')
        deccen = np.median(hdu[1].data['DEC'])
    xsize = args.xsize
    ysize = args.ysize
    projection = args.projection
    # 0.5 arcmin padding next to min/max of all events
    ramin = np.min(hdu[1].data['RA'] - 0.5/60.)
    ramax = np.max(hdu[1].data['RA'] + 0.5/60.)
    decmin = np.min(hdu[1].data['DEC'] - 0.5/60.)
    decmax = np.max(hdu[1].data['DEC'] + 0.5/60.)

    npixx = np.ceil((ramax - ramin) / (xsize / 3600.)).astype(int)
    npixy = np.ceil((decmax - decmin) / (ysize / 3600.)).astype(int)
    this_wcs = wcs.WCS(naxis=2)
    this_wcs.wcs.crpix = [npixx / 2., npixy / 2.]
    this_wcs.wcs.cdelt = np.array([-1 * xsize / 3600., ysize / 3600.])
    this_wcs.wcs.crval = [racen, deccen]
    this_wcs.wcs.ctype = ["RA---" + projection, "DEC--" + projection]
    this_wcs.wcs.cunit = ["deg", "deg"]    

    skyxy = this_wcs.world_to_array_index_values(evttab['RA'],evttab['DEC'])
    skyx = skyxy[1]
    skyy = skyxy[0]
    newhdu = add_column(hdu[1], skyx,'X','D','',True)
    newhdu2 = add_column(newhdu, skyy,'Y','D','',True)
    newevents = fits.HDUList()
    newevents.append(fits.PrimaryHDU())
    newevents.append(newhdu2)
    # if there's a GTI HDU, adding it
    newevents.append(hdu[2])
    newevents.writeto(outname,overwrite=True)

def parse_args():

    parser = Parser(description=__doc__,
        epilog="""Chien-Ting Chen <ct.chen@nasa.gov>""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-file',type=str,default="",
        required=True, help="Filename")
    parser.add_argument('-out',type=str,default="",
        required=False, help="Output filename. If the input file name has .evt extension, the output file name will be the input file name with a .fits extention (unless specified otherwise).")
    parser.add_argument('-xsize',type=float,default=1.5,
        required=False, help="X pixel size in arcsec, default is 1.5")    
    parser.add_argument('-ysize',type=float,default=1.5,
        required=False, help="Y pixel size in arcsec, default is 1.5")    
    parser.add_argument('-ra',type=float,default=-1.,
        required=False, help="Pointing RA position of the event file, default action is to read the value off the file header. Non-negative values will supercede that.")    
    parser.add_argument('-dec',type=float,default=100.,
        required=False, help="Pointing DEC position of the event file, default action is to read the value off the file header. abs(DEC) < 90 will supercede that.")    
    parser.add_argument('-projection',type=str,default='TAN',
        required=False, help="Fits projection type, default is TAN (tangential)")    
    args, unknown = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)
