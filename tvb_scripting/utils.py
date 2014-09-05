"utils.py"


from parameters import ParameterSet
from itertools import product
import numpy as np

def create_param_files(default_pset_file, iterlist, new_pfx, writefiles=False):
  """
  Usage:
  ======

  outdir =  '/tmp/tvb_scripting_iter_devel'
  outpfx = outdir + '/new_set_'
  default_pset_file = '/media/sf_SharedFolder/Code/git_repos_of_mine/tvb-scripting/examples/tvb_demos/params__region_deterministic_demo.param'

  #                type           params/type        param                                values
  iterlist = [ ['connectivity',     'params',       'speed',                       np.linspace(0,10,15)       ],
               ['model',             'type',                    ['Generic2dOscillator', 'WongWang', 'JansenRit']  ],  ]


  model_files = create_param_files(default_pset_file, iterlist, new_pfx)


  """

  # Read default parameter file
  default_pset  = ParameterSet('file://%s' %default_pset_file)
  
  all_vals = [ [ a[0:-1]+[n] for n in a[-1] ] for a in iterlist]
  
  all_prods = list(product(*all_vals))

  all_new_psets = []
  all_new_names = []
  all_new_fnames = []
  all_new_fs_dict = {}

  model_files_dict = {}
  model_params_dict = {}

  for ap_it, ap in enumerate(all_prods):
    new_pset = default_pset.copy()
    new_name = 'model%s__' %(ap_it)
    for aa in ap:
  
      if len(aa) == 3:
        new_pset[aa[0]][aa[1]] = aa[2]
      elif len(aa) == 4: 
        new_pset[aa[0]][aa[1]][aa[2]] = aa[3]
   
      if 'type' in aa:
        new_name += '__%s_%s' %(aa[0], aa[2])
      else: 
        new_name += '__%s_%s' %(aa[-2],aa[-1])
    
    all_new_psets.append(new_pset)
    all_new_names.append(new_name)

    new_fname = '%s_%s.param' %(new_pfx,  new_name)
    print 'new file: %s' %new_fname
    if writefiles:
      f = open(new_fname, 'w+')
      f.writelines(str(new_pset))
      f.close()
    
    all_new_fnames.append(new_fname)

    all_new_fs_dict[new_name] = new_fname 

    model_files_dict[new_name] = new_fname
    model_params_dict[new_name] = {'num': ap_it,
                                   'varied_params': ap,
                                   'params_dict': new_pset}

  return model_files_dict, model_params_dict

  #return all_new_fs_dict
    


