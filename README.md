# A Flow Shop Problem Exercise

The production line has 8 machines, and 28 jobs are ready to be processed. The process time of each job to the 
machine is given in [Processing Time.csv](00_Data/Processing Time.csv).  

We suppose: 
- each machine can process one job at one time. 
- the process of each job at each machine cannot be spitted. 
- Each job has to be processed along machines based on sequence 1,2,3,, ,8  (for example, any job cannot be processed on machine 2 before machine 1, it cannot be processed on machine 3 before machines 1 or 2).
- For Jobs 3,5,7,9,11,14 you can choose only 3 job to process, and the others can be ignored.

You need give a process sequence of jobs, such that can minimize the final complete time (at last machine) of last job. 
This process sequence cannot be changed along all the machines. For example, once the sequence is 1-2-3-…-27-28, then, for each machine, the job 1 is processed first, then job 2,…, and job28 is the last process job on each machine. 

