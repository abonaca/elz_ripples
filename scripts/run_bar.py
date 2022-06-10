from model import *
import sys

if __name__ == '__main__':
    """Wrapper to run bar models"""
    
    args = sys.argv
    
    evolve_bar_stars(mw_label=args[1], Nskip=int(args[2]), iskip=int(args[3]))
