# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
runner_long.py


-- motivation = have to do lots of custom things
   ...might make this function more transparent and manageable if 
   there was just a separate bit for each configurable bit, 
   explicitly dealt with...

so would look somehting like:

def __init__:

  model = ... 
  sim.model = models

  integrator = ...
  sim.integrator = integrator

  ...etc etc.
     
  ADVANTAGE OF THIS = CAN PUT *ALL* PARAMS IN THE PARAMS FIELD
  ...and deal with exceptions in each case

  ...so something like:
  
  param_exceps = 'blah'
  params_notexceps = [p for p in Ps['model']['params'] if p not in param_exceps]
  model = getattr(models,Ps['model']['type'], **params_notexceps)
  ...then add in the exceps...



"""



"""
=========
runner.py
=========

Class for running TVB simulations in the TVB-scripting format. 

"""

from IPython import embed

import numpy as np
import sys
import os
import pickle as pkl
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tvb.simulator.lab import connectivity,coupling,models,integrators,simulator,monitors,equations,patterns
from tvb.datatypes import surfaces, projections

import tvb.analyzers.fmri_balloon as bold

from tvb.simulator.plot.tools import plot_local_connectivity
#from tvb.simulator.plot.tools import surface_timeseries

from scipy.io.matlab import loadmat


class Sim(object):

  """
  Generic TVB simulation runner, using the TVB-scripting generic 
  simulation specification syntax. 


  Usage:
  ------

  from parameters import ParameterSet
  from Runner import Sim


  Ps = ParameterSet('file://' + <parameters_file>).as_dict()
  S = Sim()
  S.run()

  """

  def __init__(self,Ps): 
    
    """
    Initialize simulation
    ----------------------
    """

    sim_length = Ps['sim_params']['length']
    outdir = Ps['sim_params']['outdir']
    if not os.path.isdir(outdir): os.mkdir(outdir)

    print '\nConfiguring sim...'
   
    sim = simulator.Simulator()

    _classes = [models,    connectivity,   coupling,   integrators,  monitors ]
    _names =   ['model',  'connectivity', 'coupling', 'integrator', 'monitors'] 
   
    for _class,_name in zip(_classes,_names):
      if _name is 'monitors': 
        thisattr = tuple([getattr(_class,m['type'])(**m['params']) for m in Ps['monitors'] ])
      else:
        if 'type' in Ps[_name]:
          thisattr = getattr(_class,Ps[_name]['type'])(**Ps[_name]['params']) 
      setattr(sim,_name,thisattr)
      
 
    # Additionals - parameters that are functions of other classes
    # (example = larter_breakdspear demo)
    if 'additionals' in Ps:
      for a in Ps['additionals']: 
        setattr(eval(a[0]), a[1],eval(a[2]))
        #sim,eval(a[0]),eval(a[1]))

    # Stochastic integrator
    if 'HeunStochastic' in Ps['integrator']:
      from tvb.simulator.lab import noise
      hiss = noise.Additive(nsig=np.array(Ps['integrator']['stochastic_nsig']))  # nsigm 0.015
      sim.integrator.noise = hiss
 
    # Non-default connectivity     
    # (to add here: 
    #  - load from other data structures, e.g. .cff file
    #  - load weights, lengths, etc. directly from data matrices etc
    if 'connectivity' in Ps:
     if 'folder_path' in Ps['connectivity']: # (this is from the deterministic_stimulus demo)
       sim.connectivity.default.reload(sim.connectivity, Ps['connectivity']['folder_path'])
       sim.connectivity.configure()


    # EEG projections 
    # (need to do this separately because don't seem to be able to do EEG(projection_matrix='<file>')
    for m_it, m in enumerate(Ps['monitors']): # (yes I know enumerate isn't necessary here; but it's more transparent imho)
      # assumption here is that the sim object doesn't re-order the list of monitors for any bizarre reason...
      # (which would almost certainly cause an error anyway...)
      if m['type'] is 'EEG' and 'proj_mat_path' in m:
        proj_mat = loadmat(m['proj_mat_path'])['ProjectionMatrix']
        pr = projections.ProjectionRegionEEG(projection_data=proj_mat)
        sim.monitors[m_it].projection_matrix_data=pr


    # Surface
    if 'surface' in Ps: 
      surf = getattr(surfaces,Ps['surface']['surface_type']).default() 
      if 'local_connectivity_params' in Ps['surface']:
        localsurfconn = getattr(surfaces,'LocalConnectivity')(**Ps['surface']['local_connectivity_params'])
        for ep in Ps['surface']['local_connectivity_equation_params'].items(): 
          localsurfconn.equation.parameters[ep[0]] = ep[1]            
        surf.local_connectivity = localsurfconn
      localcoupling = np.array( Ps['surface']['local_coupling_strength'] )
      surf.coupling_strength = localcoupling
      sim.surface = surf

    # Stimulus    
    if 'stimulus' in Ps:
      stim = getattr(patterns,Ps['stimulus']['type'])()
      if 'equation' in Ps['stimulus']: # looks like need to do this to keep the other params as default; slightly different to above 
        stim_eqn_params = Ps['stimulus']['equation']['params']
        # use this if need to evaluate text    
        # (stim_eqn_params = {p[0]: eval(p[1]) for p in Ps['stimulus']['equation']['params'].items() } (
        stim_eqn_t = getattr(equations,Ps['stimulus']['equation']['type'])()
        stim_eqn_t.parameters.update(**stim_eqn_params)
        stim.temporal = stim_eqn_t
      elif 'equation' not in Ps['stimulus']:
        # (still need to do this...)
        print 'something to do here' 
      
      sim.connectivity.configure()
      stim_weighting = np.zeros((sim.connectivity.number_of_regions,))
      stim_weighting[Ps['stimulus']['nodes']]  = np.array(Ps['stimulus']['node_weightings'])

      stim.connectivity = sim.connectivity    
      stim.weight = stim_weighting
 
      sim.stimulus = stim

    # Configure sim 
    sim.configure()
    
    # Configure smooth parameter variation (if used)
    spv = {}
    if 'smooth_pvar' in Ps:
      par_length = eval(Ps['smooth_pvar']['par_length_str'])
      spv['mon_type'] = Ps['smooth_pvar']['monitor_type']
      spv['mon_num']  = [m_it for m_it, m in enumerate(Ps['monitors']) if m == spv['mon_type'] ] # (yes, a bit clumsy..) 
       
      # a) as an equally spaced range
      if 'equation' not in Ps['smooth_pvar']: 
        spv['a'] = eval(Ps['smooth_pvar']['spv_a_str'])   
      # b) using an Equation datadtype
      else: 
        spv['params'] = {}
        for p in Ps['smooth_pvar']['equation']['params'].items():
          spv['params'][p[0]] = eval(p[1])
        #sim_length = Ps['sim_params']['length'] # temporary fix]
        #spv_a_params = {p[0]: eval(p[1]) for p in Ps['smooth_pvar']['equation']['params'].items() }
        spv['eqn_t'] = getattr(equations,Ps['smooth_pvar']['equation']['type'])()
        spv['eqn_t'].parameters.update(**spv['params'])

        spv['pattern'] =  eval(Ps['smooth_pvar']['equation']['pattern_str'])
        spv['a'] = spv['pattern'] # omit above line? At moment this follows tutorial code

    # recent additions....
    self.sim = sim
    self.Ps = Ps
    self.sim_length = sim_length
    self.spv = spv

  def run(self):  
    
    """
    Run model
    -------------
    """

    # temp bit
    Ps = self.Ps
    sim = self.sim
    sim_length = self.sim_length

    print '\n\nRunning sim...'

    monitor_types = [m['type'] for m in Ps['monitors']]

    sim_time_lists = {}; sim_data_lists = {}
    for m in monitor_types: 
      sim_time_lists[m] = []
      sim_data_lists[m] = []
 
    for sim_it in sim(simulation_length = sim_length): 
      for m_it, m in enumerate(monitor_types):
        if sim_it[m_it] is not None:
          sim_time_lists[m].append(sim_it[m_it][0])
          sim_data_lists[m].append(sim_it[m_it][1]) 
          # Change a model parameter at runtime (smooth parameter variation)
          if self.spv: # 'smooth_pvar' in Ps:
            if m == self.spv['mon_type']: 
             sim.model.a = self.spv['a'][len(sim_time_lists[m])-1]

    sim_data_arrs = {}; sim_time_arrs = {}
    for m in monitor_types:
      sim_data_arrs[m]  =  np.array(sim_data_lists[m])
      sim_time_arrs[m]  =  np.array(sim_time_lists[m])

    # make the arrays 2d and put in a pandas dataframe
    df_sims = {}
    if 'make_dataframe' in Ps['sim_params']: # (have this in because this doesn't work for surfaces yet)
      if Ps['sim_params']['make_dataframe']:
        for m in monitor_types:
          thesedfs = []

          #embed()

          # temp solution; haven't yet found how to get the EEG sensor labels...this needs to be added here
          if 'EEG' in m: labels = ['sensor_%s' %(s+1) for s in np.arange(0,sim_data_arrs[m].shape[2]) ]
          else: labels = sim.connectivity.region_labels
          
          for l_it, l in enumerate(labels): #(sim.connectivity.region_labels):
            #for s_it, s in enumerate(sim.model.state_variables)
            for s_it, s in enumerate(np.arange(0,sim_data_arrs[m].shape[1])): # NEED TO CHANGE THIS
              thesedfs.append(pd.DataFrame(sim_data_arrs[m][:,s_it,l_it,0], 
                                           index = sim_time_arrs[m],
                                           columns=['%s_%s_sv%s' %(m[0:4],l,s)]))   
          df_sims[m] = pd.concat(thesedfs, axis=1)


    """
    Save data to file
    -----------------
    """

    outpfx = Ps['sim_params']['outpfx']
    outdir = Ps['sim_params']['outdir']
    if not os.path.isdir(outdir): os.mkdir(outdir)

    print '\nFinished. Saving outputs to %s/%s_' %(outdir,outpfx) 
    
    pkl.dump({'time_lists': sim_time_lists, 
              'data_lists': sim_data_lists,
              'time_arrs' : sim_time_arrs,
              'data_arrs':  sim_data_arrs,
              'df_sims'  : df_sims},
               open('%s/%s__dat.pkl' %(outdir,outpfx), "wb"))
    
    if 'make_dataframe' in Ps['sim_params']:
      if Ps['sim_params']['make_dataframe']: 
        for d in df_sims.items(): d[1].to_pickle('%s/%s__df_%s.pkl' %(outdir,outpfx,d[0]))
   
 
    """
    Make plots
    -----------
    """

    if 'plots' in Ps:
      for p in Ps['plots']:

        fig = plt.figure()
        T = sim_time_arrs[p['monitor']]
        D = sim_data_arrs[p['monitor']]
        for pc in p['plot_cmds']: 
          #eval(pc)
          exec(pc) in locals(), globals() 
        #exec p['plot_cmd']
        plt.title(p['title'])
        plt.savefig('%s/%s__%s' %(outdir,outpfx, p['plot_fname']))
        plt.close()



    def analyze():
       """
       Analyze simulated time series with TVB tools

       (Note: this function is preliminary and may be subject to change, 
              depending on how well the code below captures the majority of 
              use cases)


       If time series and analyzers are hierarchical (i.e. 
       some time series defined from analyzers that are in turn defined 
       from the original simulated time series), they should be specified 
       in that order in the 'analyzers' list. 
       They will then be added to the the 'my_analyzers' dict in order and 
       can be accessed as one would in a mult-line script. 
  
       """

       
       outpfx = self.Ps['sim_params']['outpfx']
       outdir = self.Ps['sim_params']['outdir']

       sim = self.sim

       from tvb.datatypes.time_series import TimeSeriesRegion # (others?)
       from tvb.analyzers import fmri_balloon 

       
       if 'analyzers' in self.Ps:
         my_analyzers = {}
         analyzers_out = {}
         # put the simulated data in the analyzers dict (simplifies syntax)
         for t in self.sim_time_arrs: 
           my_analyzers[t] = {'obj': self.sim, 
                              'time': self.sim_time_arrs[t], 
                              'data': self.sim_data_arrs[t]}
             
         for a in self.Ps['analyzers']:
           evalparams = {}
           for ep in a['params']: 
             if 'eval' in a['params'][ep]: 
               evalparams[ep] = eval(a['params'][ep].split('eval')[1]) 
           if 'time_series' in  a['analyzer_type']: 
               thisattr = getattr(a['type'])(data = my_analyzers['data'][a['source']],
                                             time= my_analyzers['time'][a['source']],
                                             **evalparams)
               thisattr.configure()
               my_analyzers[a['name']] = {'obj': thisattr}


           elif 'analyzer' in a['analyzer_type']: 
             
               thisattr_model = getattr(a['type'])(time_series = my_analyzers[a['source']])
               thisattr_data = thisattr_model.evaluate()
               my_analyzers[a['name']] = {'obj': thisattr_model,
                                          'data': thisattr_data.data,
                                          'time': thisattr_data.time}

       if a['save']: 
            
         analyzers_out[a['name']] = {'input_params': a,
                                     'result': my_analyzers[a['name']] }


       analyzers_outfname = '%s/%s__analyzers_dat.pkl' %(outdir, outpfx)
       print '\nSaving analyzer outputs to ' + analyzers_outfname 
       pkl.dump(analyzers_out, open(analyzers_outfname, "wb"))



    #def plot():
    #   """
    #   Plot
    #   """

    ##Prutty puctures...
    #tsi = timeseries_interactive.TimeSeriesInteractive(time_series = bold_tsr)
    #tsi.configure()
    #tsi.show()

 

