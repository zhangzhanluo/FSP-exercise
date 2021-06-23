# A Flow Shop Problem Exercise

> Full project:  https://github.com/zhangzhanluo/FSP-exercise

### Problem Description

The production line has 8 machines, and 28 jobs are ready to be processed. The process time of each job to the machine is given in [Processing Time.csv](00_Data/Processing Time.csv).  

We suppose: 
1. each machine can process one job at one time. 
2. the process of each job at each machine cannot be spitted. 
3. Each job has to be processed along machines based on sequence 1,2,3,, ,8  (for example, any job cannot be processed on machine 2 before machine 1, it cannot be processed on machine 3 before machines 1 or 2).
4. For Jobs 3,5,7,9,11,14 you can choose only 3 job to process, and the others can be ignored.

You need give a process sequence of jobs, such that can minimize the final complete time (at last machine) of last job. 
This process sequence cannot be changed along all the machines. For example, once the sequence is 1-2-3-…-27-28, then, for each machine, the job 1 is processed first, then job 2,…, and job28 is the last process job on each machine. 

> See [code_10_fsp.py](code_10_fsp.py)

### Mixed Integer Programming

A MIP model is built first to test the possibility of finding the optimal solution by MIP model. The model does not take the forth constrain into consideration for simplicity.

The variables are given as follows:

| Variable                                         | Meaning                                                      |
| ------------------------------------------------ | ------------------------------------------------------------ |
| ![](http://latex.codecogs.com/svg.latex?M)       | Set of  machines                                             |
| ![](http://latex.codecogs.com/svg.latex?J)       | Set of Jobs                                                  |
| ![](http://latex.codecogs.com/svg.latex?s_{ik})  | Starting time of job ![](http://latex.codecogs.com/svg.latex?i\in%20J) at machine ![](http://latex.codecogs.com/svg.latex?k%20\in%20M) |
| ![](http://latex.codecogs.com/svg.latex?x_{ij})  | 1 if job ![](http://latex.codecogs.com/svg.latex?i%20\in%20J) precedes job ![](http://latex.codecogs.com/svg.latex?j%20\in%20J), 0 otherwise |
| ![](http://latex.codecogs.com/svg.latex?C_{max}) | Maximum completion time (makespan)                           |
| ![](http://latex.codecogs.com/svg.latex?p_{ik})  | Processing time of job ![](http://latex.codecogs.com/svg.latex?i%20\in%20J) at machine ![](http://latex.codecogs.com/svg.latex?k%20\in%20M) |
| ![](http://latex.codecogs.com/svg.latex?Q)       | A very large number                                          |

The MIP model is given as follows:

![](http://latex.codecogs.com/svg.latex?Minimize%20\quad%20C_{max})

subject to

![](http://latex.codecogs.com/svg.image?s_{im}+p_{im}\leq%20C_{max}%20\qquad%20\forall%20i%20\in%20J)

![](http://latex.codecogs.com/svg.image?s_{ik}+p_{ik}\leq%20s_{i,k+1}%20\qquad%20\forall%20i%20\in%20J,(k,k+1)\in%20M)

![](http://latex.codecogs.com/svg.image?s_{jk}-(s_{ik}&plus;p_{ik})&plus;Q(1-x_{ij})\geqslant&space;0\qquad&space;\forall&space;i,j\in&space;J:i<j,k\in&space;M)

![](http://latex.codecogs.com/svg.image?s_{ik}-(s_{jk}&plus;p_{jk})&plus;Q(1-x_{ij})\geqslant&space;0\qquad&space;\forall&space;i,j\in&space;J:i<j,k\in&space;M)

![](http://latex.codecogs.com/svg.latex?s_{ik}\geqslant0\qquad&space;\forall&space;i\in&space;J)

![](http://latex.codecogs.com/svg.latex?x_{ij}\in\{0,1\}\qquad&space;\forall&space;i,j\in&space;J,k\in&space;M)

#### Validation of the MIP model

A flow shop scheduling instance with 12 jobs and 8 machines are solved by the above MIP model. The Gantt Chart for the solved result is shown in the below figure.

<div align=center><img width="1200" src="01_Gantt_Chart\MIP\Method%20MIP%20-%20makespan%20315%20-%20n_jobs%2012%20-%20n_machines%208%20-%20solving%20time%2014.8s.png"/></div>

> See [code_20_mip.py](code_20_mip.py)

#### Solving time of MIP models

The solving time of the MIP models with different number of jobs and different machines is given in the below figure.

<div align=center><img width="600" src="02_Results\MIP\Solving%20Time%20of%20MIP%20Models.png"/></div>

As can be seen in the above figure, the solving time increased exponentially with the increase of the number of jobs. The number of machines has little effect on the growth of the time.

Take the number of machines as 8 and fit the solving time curve. The results are showed below.

<div align=center><img width="600" src=".\02_Results\MIP\Solving%20Time%20of%20MIP%20Models%20(8%20Machines).png"/></div>

As can be seen in the picture, the time complex of the MIP model is at least ![](http://latex.codecogs.com/svg.image?O(2.842^n)). For the original problem, ![](http://latex.codecogs.com/svg.latex?n\geqslant25). Thus the solving time for the original flow shop problem will be larger than 156 days!

> See [code_21_mip_experiments.py](code_21_mip_experiments.py) and [code_22_mip_analysis.py](code_22_mip_analysis.py)

### Benchmark Model

10 random sequences are given and their makespans are calculated. The results are [597, 614, 627, 630, 573, 606, 589, 628, 602, 628]. The average is **609.4s**, which will be used as the benchmark for this problem. A boxplot for the results is given as follows:

<div align=center><img width="250" src=".\02_Results\Random\10 random solutions.png"/></div>

The Gantt Chart for one of the random solutions is given as follows:

![](01_Gantt_Chart\random\Method%20random%20-%20makespan%20597%20-%20n_jobs%2025%20-%20n_machines%208%20-%20random_seed%200.png)

> See [code_11_benchmark.py](code_11_benchmark.py)

### Greedy Algorithm

The NEH heuristic is used here to give the greedy solution. The following figure (Öztop, 2019) gives the details of the NEH heuristic:

<div align=center><img width="800" src=".\02_Results\Reference\NEH_Heuristic.jpg"/></div>

The solution given by NEH heuristic is [17, 2, 15, 21, 28, 25, 13, 12, 1, 14, 19, 11, 24, 26, 6, 7, 4, 27, 10, 23, 8, 18, 20, 16, 22]. The makespan for this solution is **477s**. The Gantt Chart for this solution is as follows:

<div align=center><img width="1200" src=".\01_Gantt_Chart\NEH\Method%20NEH%20-%20makespan%20477%20-%20n_jobs%2025%20-%20n_machines%208.png"/></div>

The NEH Heuristic treat the job with longest total processing time first. To see the difference, a reversed NEH heuristic which sorts the jobs in increasing order. The solution given by reversed NEH heuristic is [21, 6, 19, 3, 12, 20, 1, 11, 4, 15, 13, 27, 17, 10, 8, 2, 24, 5, 26, 28, 25, 22, 23, 16, 18]. The makespan for this solution is **502s**. The Gantt Chart for this solution is as follows:

<div align=center><img width="1200" src=".\01_Gantt_Chart\Reversed%20NEH\Method%20Reversed%20NEH%20-%20makespan%20502%20-%20n_jobs%2025%20-%20n_machines%208.png"/></div>

The reversed NEH heuristic works worse than the original NEH algorithm. This indicates that taking the job with longest total processing time firstly does work for this problem!

> See [code_30_greedy_algorithms.py](code_30_greedy_algorithms.py)

### Genetic Algorithm

#### Basic Algorithm

Genetic algorithm is employed to solve this FSP. Some general setting are:

| Parameter            | Value                                                  |
| -------------------- | ------------------------------------------------------ |
| Number of Generation | 100                                                    |
| Population Size      | 50                                                     |
| Crossover Rate       | 0.8                                                    |
| Mutation Rate        | 0.003                                                  |
| Selection Method     | Championship or Roulette Wheel                         |
| Crossover Method     | Partial-Mapped Crossover (PMX) or Order Crossover (OX) |

More details about the mentioned selection method and crossover method can be referenced in [this article](https://blog.csdn.net/u012750702/article/details/54563515) (in Chinese). Elite strategy is employed to make sure the the population will at least not be worse than the former ones.

The difference between the selection methods and crossover methods is analyzed. Using Championship selection method and PMX crossover method, the revolution goes like this:

<div align=center><img width="1200" src="02_Results\GA\GA_Revolution_with_Championship_Selection_PMX_Crossover_False_Good_Start.png"/></div>

Championship selection method and OX crossover method:

<div align=center><img width="1200" src="02_Results\GA\GA_Revolution_with_Championship_Selection_OX_Crossover_False_Good_Start.png"/></div>

Roulette Wheel selection method and PMX crossover method:

<div align=center><img width="1200" src="02_Results\GA\GA_Revolution_with_Roulette_Wheel_Selection_PMX_Crossover_False_Good_Start.png"/></div>

Roulette Wheel selection method and OX crossover method:

<div align=center><img width="1200" src="02_Results\GA\GA_Revolution_with_Roulette_Wheel_Selection_OX_Crossover_False_Good_Start.png"/></div>

In summary:

|                | PMX                                     | OX                                                        |
| -------------- | --------------------------------------- | --------------------------------------------------------- |
| Champion       | Best makespan: 503<br />Quick decrease  | Best makespan: 503<br />Fluctuation and unsteady decrease |
| Roulette Wheel | Best makespan: 484<br />Smooth decrease | Best makespan: 508<br />Fluctuation and unsteady decrease |

As we can see from the above table, the PMX method outperforms the OX method in terms of both the final best makespan and the revolution process for this FSP problem. 

#### Good Start

Since we already have a good solution by greedy algorithm. A very straight forward idea is to add the solution given by the greedy algorithm to the initial population. After adding the solution found by greedy algorithm, the results are:

|                | PMX                                     | OX                                                        |
| -------------- | --------------------------------------- | --------------------------------------------------------- |
| Champion       | Best makespan: 477<br />Quick decrease  | Best makespan: 477<br />Fluctuation and unsteady decrease |
| Roulette Wheel | Best makespan: 474<br />Smooth decrease | Best makespan: 477<br />Fluctuation and unsteady decrease |

As can be seen in the above table, only the combination of Roulette Wheel selection and PMX produces a slightly better result. The revolution process is:

<div align=center><img width="1200" src="02_Results\GA\GA_Revolution_with_Roulette_Wheel_Selection_PMX_Crossover_True_Good_Start.png"/></div>

 The new best solution is [17, 21, 15, 2, 28, 25, 13, 12, 1, 14, 19, 8, 24, 26, 6, 7, 5, 4, 27, 10, 23, 18, 20, 16, 22]. The corresponding makespan is **474s**. The corresponding Gantt Chart is:

<div align=center><img width="1200" src="01_Gantt_chart\GA\Method_GA_+_makespan_474+PMX_Roulette_Wheel_Good_Start.png"/></div>

#### Education

Another import concept for genetic algorithm is Education. In this problem, a very simple education strategy is to swap all the pairs in a solution and see whether a new solution can be get. To avoid too much calculation, the ones whose makespan is no 10s larger than the best solution can get the chance to be educated. A list called school is created to record the history of educated solution. If a solution is in school already, it will not be educated again. With education strategy, the results now are:

|                | PMX                                                      | OX                                                       |
| -------------- | -------------------------------------------------------- | -------------------------------------------------------- |
| Champion       | Best makespan: 459<br />Best result from 28th generation | Best makespan: 459<br />Best result from 58th generation |
| Roulette Wheel | Best makespan: 459<br />Best result from 20th generation | Best makespan: 461<br />Best result from 28th generation |

New best solutions are get from the combinations of Champion + PMX, Champion + OX, and Roulette Wheel + PMX. Among all the combinations, Roulette Wheel + PMX has the fastest decreasing rate, which is showed in the following figure:

<div align=center><img width="1200" src="02_Results\GA\GA_Revolution_with_Roulette_Wheel_Selection_PMX_Crossover_True_Good_Start_True_Education.png"/></div>

Although three combinations give the same best makespan, their final solutions are different. The solutions are:

| Combination          | Solution                                                     |
| -------------------- | ------------------------------------------------------------ |
| Champion + PMX       | [21, 24, 1, 17, 28, 6, 13, 4, 15, 14, 19, 8, 7, 26, 25, 2, 5, 12, 27, 3, 9, 10, 23, 11, 18, 20, 16, 22] |
| Champion + OX        | [21, 24, 17, 5, 15, 28, 13, 12, 1, 14, 19, 8, 7, 26, 25, 2, 6, 4, 27, 3, 9, 10, 23, 11, 18, 20, 16, 22] |
| Roulette Wheel + PMX | [21, 24, 5, 1, 28, 15, 13, 12, 17, 14, 19, 8, 7, 26, 25, 2, 6, 4, 27, 3, 9, 10, 23, 11, 18, 20, 16, 22] |

The Gantt Chart for the Roulette Wheel + PMX solution is:

<div align=center><img width="1200" src="01_Gantt_Chart\GA\Method_GA_+_makespan_459+PMX_Roulette_Wheel_Good_Start_Education.png"/></div>

> see [code_40_ga.py](code_40_ga.py)

### Conclusion

| Combination                                     | Makespan | Solution                                                     |
| ----------------------------------------------- | -------- | ------------------------------------------------------------ |
| Random                                          | 609.4    | None                                                         |
| Greedy Algorithm                                | 477      | [17, 2, 15, 21, 28, 25, 13, 12, 1, 14, 19, 11, 24, 26, 6, 7, 4, 27, 10, 23, 8, 18, 20, 16, 22] |
| Genetic Algorithm                               | 484      | [21, 6, 2, 1, 17, 28, 24, 19, 16, 10, 8, 26, 13, 5, 7, 15, 4, 25, 3, 18, 12, 23, 9, 11, 14, 22, 27, 20] |
| Genetic Algorithm +Greedy Algorithm             | 474      | [17, 21, 15, 2, 28, 25, 13, 12, 1, 14, 19, 8, 24, 26, 6, 7, 5, 4, 27, 10, 23, 18, 20, 16, 22] |
| Genetic Algorithm +Greedy Algorithm + Education | 459      | [21, 24, 1, 17, 28, 6, 13, 4, 15, 14, 19, 8, 7, 26, 25, 2, 5, 12, 27, 3, 9, 10, 23, 11, 18, 20, 16, 22]<br />[21, 24, 17, 5, 15, 28, 13, 12, 1, 14, 19, 8, 7, 26, 25, 2, 6, 4, 27, 3, 9, 10, 23, 11, 18, 20, 16, 22]<br />[21, 24, 5, 1, 28, 15, 13, 12, 17, 14, 19, 8, 7, 26, 25, 2, 6, 4, 27, 3, 9, 10, 23, 11, 18, 20, 16, 22] |

The best solution is [21, 24, 5, 1, 28, 15, 13, 12, 17, 14, 19, 8, 7, 26, 25, 2, 6, 4, 27, 3, 9, 10, 23, 11, 18, 20, 16, 22]. The makespan for the best solution is **459s**. The Gantt Chart for the best solution is as follows:

<div align=center><img width="1200" src="01_Gantt_Chart\GA\Method_GA_+_makespan_459+PMX_Roulette_Wheel_Good_Start.png"/></div>

### Acknowledge

The author thanks [Assoc. Prof. Liu](https://me.sjtu.edu.cn/teacher_directory1/liuran.html) for his teaching in the class.

### Reference

Öztop H, Tasgetiren M F, Eliiyi D T, et al. Metaheuristic algorithms for the hybrid flowshop scheduling problem[J]. Computers & Operations Research, 2019, 111: 177-196.
