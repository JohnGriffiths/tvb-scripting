# tvb-scripting

generic scripting syntax for running tvb simulations



### *What's this then?*

tvb-scripting is a simple set of tools that allow a range of tvb simulation scenarios to be run by simply supplying a short text file detailing the parameter values to be used. 



### *Why?*

The rationale is to try to separate code specifying how a simulation is run from code specifying the parameter values used. This is generally regarded as 'good practice' or 'desirable' in both simulation and data analysis research, and has a number of advantages.  

*(Most of these and more are summarized better by Andrew Davison, author of the Sumatra and PyNEURON libraries).*



***Less code needed to specify a simulation***

As rules of thumb go, 'the less code the better' isn't a bad one. 

My personal experience of scripting, especially sharing of scripts between multiple users and over long periods of time, is that a certain amount of 'drift' in cosmetic things like layout, variable names, etc. is inevitable. At best this is often unecessary; at worse this makes understanding something you didn't write a serious headfuck requiring a lot more time and brainpower than it should. It also makes 'real' version control (therefore long-term management, bux-fixing, reproducibility) exponentially harder than it needs to be. 

Other reasons why less code is better include minimizing redundancy; less chance of silly-mistake bugs; being more easily human readable, etc. etc. 



***Plays well with databases***

The json-type nested python dictionaries used by tvb-scripting lend themselves very well to databasing, as they effectively define database entries. In fact this is precisely what Sumatra does: log the input parameters in a database, and take a snapshot of the execution code using a version control system such as git. Whilst tvb-scripting is intended primarily for use with tvb-sumatra, it doesn't need to be, and the functionality was considered useful enough on its own to separate the two. The nested dictionary format isn't arbitrary - it follows the recommended format of the Parameters python library, which provides some other useful tools for dealing with data structures of this kind. 


***Plays well with Sumatra***

ibid...




### *Usage*

The command line usage is simple: 

        python run_tvb_sims.py <PARAMETER FILE>


***Examples***

Parameter files matching several of the demo examples from the tvb-library repo are given in /examples/tvb_demos. 














