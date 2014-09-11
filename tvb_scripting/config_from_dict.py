# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-


def getsimattr(_class,_params):
  _type = _params['type']
  _simattr = getattr(_class,_type)
  _trait_params = [p for p in _params if p in _simattr.trait.keys()]
  return _simattr(**_trait_params)


def config_from_dict(Ps):

  import numpy as np
  import os
  from tvb.simulator.lab import connectivity,coupling,models,integrators,simulator,monitors,equations,patterns
  from tvb.datatypes import surfaces, projections
  import tvb.analyzers.fmri_balloon as bold
  from tvb.simulator.plot.tools import plot_local_connectivity
  from tvb.simulator import noise
  #from tvb.simulator.plot.tools import surface_timeseries
  from scipy.io.matlab import loadmat

   
  

  # Simulator
  # ---------
  _sim = getsimattr(simulator, Ps['sim_params'])()



  # 1. Mandatory simulator components:


  """
  Connectivity
  ------------
  """

  # Defaults
  _params = Ps['connectivity'])
  _connectivity = getsimattr(connectivity,_params)

  # Non-defaults
  if 'folder_path' in _params:
    _connectivity.default.reload(connectivity, _params['folder_path'])
    # (from the demo)

  # (to add here: 
  #  - load from other data structures, e.g. .cff file
  #  - load weights, lengths, etc. directly from data matrices etc

  _sim.connectivity = _connectivity

  """
  Coupling
  --------
  """

  # Defaults
  _coupling = getsimattr(coupling, Ps['coupling')

  # Non-defaults

  _sim.coupling = _coupling


  """
  Model
  -----
  """

  # Defaults
   _model = getsimattr(models,Ps['model'])

  # Non-defaults
 
  _sim.model = _model

  """
  Integrator
  ----------
  """

  # Defaults
  # - 'dt' is the only one that needs to be set in the config dict. 
  # - ('noise' is configured in the non-defaults section below )
  
  _params = Ps['integrator']
  _integrator = getsimattr(integrators,_params)

  
  # Non-defaults
 
  # format: 
  # Ps['integrator']['noise_type'] = 'Additive'
  # Ps['integrator']['Additive__nsig'] = [0.0] 
  if _params['noise_type'] is 'Additive': 
    _noise = noise.Additive()
    _noise.nsig = _params['Additive__nsig']
    _noise.ntau = _params['Additive__ntau']
    
  # TO ADD: STUFF FOR 'random_stream' configuration ('init_seed' arg? ) ...
  # if 'Additive__random_stream__'

  
  if _params['noise_type'] is 'Multiplicative': 
    #  TO SORT THIS; c.f. stuff below...
    #eqn = equations.TemporalApplicableEquation()
    #b._params)
    # format:  ( double underscores '__' attempt to indicate sub-traits...)
    # Ps['integrator']['Multiplicative__equation'] = 'Linear'
    # Ps['integrator']['Multiplicative__Linear__a'] = 1.0 
    # Ps['integrator']['Multiplicative__Linear__b'] = 0.0 
     
  _sim.integrator = _integrator

  """
  Monitors
  --------
  """

  _monitors_list = []
  _monitor_keys = [k for k in Ps if 'monitor_' in k]
  for _m in _monitor_keys:

    _params = Ps[_m] 

    # Default params
    _monitor = getsimattr(monitors,_params) 

    # EEG projections    
    if _params['type'] is 'EEG' and 'proj_mat_path' in m:
      proj_mat = loadmat(_params['proj_mat_path'])['ProjectionMatrix']
      pr = projections.ProjectionRegionEEG(projection_data=proj_mat)
      _.monitors.projection_matrix_data=pr

    # Non-default params
    _monitors_list.append(_monitor)

  _monitors = tuple(_monitors_list)

  _sim.monitors = _monitors




  # 2. Non-mandatory components:


  """
  Speed
  -----
  """


  
  """
  Surface
  -------
  """
  if 'surface' in Ps:
    _surface = getsimattr(surface, Ps['surface')
    # note: might need to use the default() call if this isn't working...
    #  surf = getattr(surfaces,Ps['surface']['surface_type']).default()
    
    #
    #if 'local_connectivity_params' in Ps['surface']:
    #    localsurfconn = getattr(surfaces,'LocalConnectivity')(**Ps['surface']['local_connectivity_params'])
    #    for ep in Ps['surface']['local_connectivity_equation_params'].items():
    #      localsurfconn.equation.parameters[ep[0]] = ep[1]
    #    surf.local_connectivity = localsurfconn
    #  localcoupling = np.array( Ps['surface']['local_coupling_strength'] )
    #  surf.coupling_strength = localcoupling
    #  sim.surface = surf

  _sim.surface = _surface

  """
  Stimulus
  --------
  """

  
  if 'stimulus' in Ps:
    _params = Ps['stimulus']
    _stimulus = getsimattr(stimulus, _params)
    # ...now do non-default params individually
    
    
    #stim = getattr(patterns,Ps['stimulus']['type'])()
    #  if 'equation' in Ps['stimulus']: # looks like need to do this to keep the other params as default; slightly different to above 
    #    stim_eqn_params = Ps['stimulus']['equation']['params']
    #    # use this if need to evaluate text    
    #    # (stim_eqn_params = {p[0]: eval(p[1]) for p in Ps['stimulus']['equation']['params'].items() } (
    #    stim_eqn_t = getattr(equations,Ps['stimulus']['equation']['type'])()
    #    stim_eqn_t.parameters.update(**stim_eqn_params)
    #    stim.temporal = stim_eqn_t
    #  elif 'equation' not in Ps['stimulus']:
    #    # (still need to do this...)
    #    print 'something to do here'
    #
    #  sim.connectivity.configure()
    #  stim_weighting = np.zeros((sim.connectivity.number_of_regions,))
    #  stim_weighting[Ps['stimulus']['nodes']]  = np.array(Ps['stimulus']['node_weightings'])
    #
    #  stim.connectivity = sim.connectivity
    #  stim.weight = stim_weighting
    #
    #  sim.stimulus = stim
  
  _sim.stimulus = _stimulus    



  """
  Initial conditions
  ------------------
  """


  """
  Simulation_length
  -----------------
  """



  # Others:
  

  """
  Smooth pvar
  -----------
  """
  # (c.f. lots of stuff in other file)
  #_model = getsimattr(models,Ps['model'])


  """
  Additionals
  ----
  """
  # (c.f. lots of stuff in other file)
  # _model = getsimattr(models,Ps['model'])
  


  
  return _sim
  





