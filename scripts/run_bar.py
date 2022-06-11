from model import *
import sys

if __name__ == '__main__':
    """Wrapper to run bar models"""
    
    args = sys.argv
    if len(args)<5:
        args += [False]
    
    evolve_bar_stars(mw_label=args[1], Nskip=int(args[2]), iskip=int(args[3]), test=bool(args[4]), m=5e9*u.Msun)
