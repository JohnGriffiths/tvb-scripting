{
 'sim_params':                   {'length':  10, #2**12,
                                  'outdir': 'Data',
                                  'outpfx': 'mse_analyse_region_demo',
                                  'make_dataframe': True},
 'model':                        {'type':   'Generic2dOscillator' ,
                                  'params':  {'a':-0.5, 'b':-10., 'c':0.0,'d':0.02}  },
 'coupling':                     {'type':   'Linear',
                                  'params': {'a'   :   0.042}           },
 'integrator':                   {'type':   'HeunStochastic',
                                  'params': {'dt'  :   2**-6},
                                  'stochastic_nsig': 0.00                },
 'monitors':                     [ {'type':   'TemporalAverage', 
                                    'params': {'period': 1e3/4096.}      },
                                   {'type': 'EEG', 
                                    'params': {'period': 1e3/4096.},
                                    'proj_mat_path': '/tmp/region_conn_74_eeg_1020_62.mat'}  ] ,
 'connectivity':                 {'type':   'Connectivity',
                                  'params': {'speed' :  [4.0], 
                                             'load_default': True }         },
 'stimulus':                     {'type':   'StimuliRegion', 
                                  'equation': {'type': 'PulseTrain',
                                               'params':  {'onset': 1000.0,
                                                           'tau':   5.0,
                                                           'T':     750.   }},
                                  'nodes':   [35, 36],
                                  'node_weightings': [[3.5],[0.]]          }
}
