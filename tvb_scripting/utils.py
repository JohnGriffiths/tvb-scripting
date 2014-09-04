"utils.py"


from parameters import ParameterSet
from itertools import product

def create_param_files(default_pset_file, iterlist, new_pfx):
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
  for ap in all_prods:
    for aa in ap:
      new_pset = default_pset.copy()
      if len(aa) == 3:
        new_pset[aa[0]][aa[1]] = aa[2]
      elif len(aa) == 4: 
        new_pset[aa[0]][aa[1]][aa[2]] = aa[3]
      all_new_psets.append(new_pset)
  #all_new_psets

  new_fs = []        
  for a_it, a in enumerate(all_new_psets): 
    new_fname = '%s_%s.param' %(new_pfx, a_it)
    print 'new file: %s' %new_fname
    f = open(new_fname, 'w+')
    f.writelines(str(a))
    f.close()
    new_fs.append(new_fname)


  return new_fs
  




