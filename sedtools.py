import numpy as np
import pandas as pd

def int_over_resp(filters,temp_wav,temp_flux,redshift=None):
    '''
    Takes in: filters -- a pandas DF, including a "WAV" column
    templates -- two columns DF/array
    redshift -- set if redshift is required
    '''
    #Check if WAV exist in the filter DF
    filter=pd.DataFrame.copy(filters)
    if not 'WAV' in filters:raise NameError('filters should have a column named "WAV"')
    f_wav=filter.WAV.values
    temp_freq=3e14/temp_wav
    filter.drop('WAV',axis = 1, inplace = True)
    flux_out=[]
    colname=[]
    if not np.array_equal(f_wav,temp_wav):
        for column in filter.iteritems():
            colname.append(list(column)[0])
            resp=np.asarray(list(column)[1])
            resp = np.interp(temp_wav, f_wav, resp)
            #ignore the values in the resp outside of the filter wavelengths
            resp[np.where((temp_wav < min(f_wav)) | (temp_wav > max(f_wav)))]=0.
            int_flux=np.trapz(temp_flux*resp,x=temp_freq)
            int_norm=np.trapz(resp,x=temp_freq)
            flux_out.append(int_flux/int_norm)
    else:
        for column in filter.iteritems():
            colname.append(list(column)[0])
            resp=np.asarray(list(column)[1])
            #ignore the values in the resp outside of the filter wavelengths
            resp[np.where((temp_wav < min(f_wav)) | (temp_wav > max(f_wav)))]=0.
            int_flux=np.trapz(temp_flux*resp,x=temp_freq)
            int_norm=np.trapz(resp,x=temp_freq)
            flux_out.append(int_flux/int_norm)
        
    result=pd.DataFrame({'filter':colname, 'flux':flux_out})
    return result
