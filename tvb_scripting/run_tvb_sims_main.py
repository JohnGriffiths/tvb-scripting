"""
====================
run_tvb_sims_main.py
====================

Wrapper for Sim class in Runner.py.


Usage:
-----

python run_tvb_sims_main.py <parameter_file>

"""
import numpy
import sys
from parameters import ParameterSet
from runner import Sim

#improt os
#tvbsd = os.path.abspath(os.getcwd())
#'/media/sf_WINDOWS_D_DRIVE/Neurodebian/code/git_repos/TheVirtualBrain/'\
#           '/tvb-pack/library/contrib/simulator/tools'
#sys.path.append(tools_dir)    
#from tvb_scripting.Runner import Sim


def main(pset):

 Ps = pset.as_dict() 
 S = Sim(Ps)
 S.run()

# Read parameter file
parameter_file = sys.argv[1]
pset  = ParameterSet('file://%s' %parameter_file)

# Run
main(pset)
