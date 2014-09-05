
# Notebook Runner
# ===============

#Using runipy

"""
from runipy.notebook_runner import NotebookRunner
from IPython.nbformat.current import read

notebook = read(open("MyNotebook.ipynb"), 'json')
r = NotebookRunner(notebook)
r.run_notebook()

r = NotebookRunner(notebook, pylab=True)

from IPython.nbformat.current import write
write(r.nb, open("MyOtherNotebook.ipynb", 'w'), 'json')
"""

# Command line version:

# Run with, e.g. SPEED=8.0 runipy Tutorial_Exploring_A_Model__tvb_scripting.ipynb new_nb5.ipynb


# nipype interface for runipy
# ===========================


#$ myvar=value runipy MyNotebook.ipynb

import os

def call_runipy(var_list, orig_notebook, new_notebook):

  # var_list looks something like
  # var_list = [ ['SPEED', 4.0], 
  #               ['DT', 2**1 ]  ]


  cmd = ''
  for v in var_list: 
    cmd += '%s=%s ' %(v[0], v[1]) # v[0] = name, v[1] = value

  cmd += '  runipy ' + orig_notebook + ' ' + new_notebook

  print 'calling runipy: \n ' + cmd

  os.system(cmd)


