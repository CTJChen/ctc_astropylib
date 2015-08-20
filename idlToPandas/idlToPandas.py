#!/usr/bin/env python

"""
Created on 2014-06-01T20:15:40
"""

from __future__ import division, print_function
import sys


try:
    import argparse
except ImportError:
    print('You need argparse installed')
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print('You need pandas installed')
    sys.exit(1)

try:
    from scipy.io.idl import readsav
except ImportError:
    print('You need scipy installed')
    sys.exit(1)


__author__ = "Matt Giguere (github: @mattgiguere)"
__maintainer__ = "Matt Giguere"
__email__ = "matthew.giguere@yale.edu"
__status__ = " Production"
__version__ = '0.1.0'


def idlToPandas(fileName, keyValue=0):
    """PURPOSE: To restore an IDL strcture contained
    within an IDL save file and add it to a pandas
    data frame."""
    idlSavedVars = readsav(fileName)

    #check if the keyValue passed in is actually an index
    #rather than the keyValue name:
    if valIsNumber(keyValue):
        keys = idlSavedVars.keys()
        keyValue = keys[keyValue]

    struct = idlSavedVars[keyValue]
    tags = []
    for tag in struct.dtype.descr:
        tags.append(tag[0][0])

    #now take care of potential big-endian/little-endian issues
    dt = struct.dtype
    dt = dt.descr
    for i in range(len(dt)):
        if(dt[i][1][0] == '>' or dt[i][1][0] == '<'):
            dt[i] = (dt[i][0], dt[i][1][1:])
    struct = struct.astype(dt)

    pdf = pd.DataFrame.from_records(struct, columns=tags)
    return pdf


def valIsNumber(instring):
    try:
        float(instring)
        return True
    except:
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='A routine to restore a structure in an IDL save file '
        ' and store the contents in a pandas data frame.')
    parser.add_argument(
        'fileName',
        help='The name of the IDL save structure if there is more than '
        'one.', nargs='?')
    parser.add_argument(
        'keyValue',
        help='The name of the IDL save structure if there is more than '
        'one.', nargs='?')
    if len(sys.argv) > 2:
        print('use the command')
        print('python idlToPandas.py fileName')
        sys.exit(2)

    args = parser.parse_args()

    idlToPandas(args.fileName, keyValue=args.keyValue)
