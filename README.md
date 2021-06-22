# A Flow Shop Problem Exercise

### Problem Description
The production line has 8 machines, and 28 jobs are ready to be processed. The process time of each job to the machine is given in [Processing Time.csv](00_Data/Processing Time.csv).  

We suppose: 
1. each machine can process one job at one time. 
2. the process of each job at each machine cannot be spitted. 
3. Each job has to be processed along machines based on sequence 1,2,3,, ,8  (for example, any job cannot be processed on machine 2 before machine 1, it cannot be processed on machine 3 before machines 1 or 2).
4. For Jobs 3,5,7,9,11,14 you can choose only 3 job to process, and the others can be ignored.

You need give a process sequence of jobs, such that can minimize the final complete time (at last machine) of last job. 
This process sequence cannot be changed along all the machines. For example, once the sequence is 1-2-3-…-27-28, then, for each machine, the job 1 is processed first, then job 2,…, and job28 is the last process job on each machine. 

### Mixed Integer Programming

A MIP model is built first to test the possibility of finding the optimal solution by MIP model. The model does not take the forth constrain into consideration for simplicity.

The variables are given as follows:

| Variable                                         | Meaning                                                      |
| ------------------------------------------------ | ------------------------------------------------------------ |
| ![](http://latex.codecogs.com/svg.latex?M)       | Set of  machines                                             |
| ![](http://latex.codecogs.com/svg.latex?J)       | Set of Jobs                                                  |
| ![](http://latex.codecogs.com/svg.latex?s_{ij})  | Starting time of job ![](http://latex.codecogs.com/svg.latex?i \in J) at machine ![](http://latex.codecogs.com/svg.latex?j \in M) |
| ![](http://latex.codecogs.com/svg.latex?x_{ij})  | 1 if job ![](http://latex.codecogs.com/svg.latex?i \in J) precedes job ![](http://latex.codecogs.com/svg.latex?j \in J), 0 otherwise |
| ![](http://latex.codecogs.com/svg.latex?C_{max}) | Maximum completion time (makespan)                           |
| ![](http://latex.codecogs.com/svg.latex?p_{ij})  | Processing time of job ![](http://latex.codecogs.com/svg.latex?i \in J) at machine ![](http://latex.codecogs.com/svg.latex?j \in M) |

The MIP model is given as follows:

![](http://latex.codecogs.com/svg.latex?Minimize \ C_{max})

subject to

![](http://latex.codecogs.com/svg.latex?s_{im}+p_{im}\leq C_{max} \qquad \forall j \in J)

![](http://latex.codecogs.com/svg.latex?s_{im}+p_{im}\leq C_{max} \qquad \forall j \in J)

