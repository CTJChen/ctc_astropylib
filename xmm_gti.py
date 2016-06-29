"""
Needs to fix outputfile, it's of no use here.
"""
import sys, getopt
from ctc_xray import xmm_bkgd

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'test.py -i <inputfile> -o <outputfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'test.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
    df = xmm_bkgd(inputfile, df=True)
    print(np.sum(df.RATE < 0.4)/float(np.sum(df)))

if __name__ == "__main__":
   main(sys.argv[1:])
