"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210621
    Description: Some Greedy Algorithms
"""
from code_10_fsp import FSP


class GreedyAlgorithms:
    def __init__(self, instance: FSP):
        self.instance = instance

    def sort_the_jobs(self):
        """
        stage 1 of NEH heuristic.
        the sum of the processing times on all stages (TPj) is calculated for each job j ∈ J.
        Then, jobs are sorted in decreasing order of TPj.

        :return: sorted jobs
        """
        TP = [sum(self.instance.processing_time[j]) for j in range(self.instance.n_total_jobs)]
        tp_jobs = [(TP[i], i + 1) for i in range(self.instance.n_total_jobs)]  # 这里给出了每个任务的名字，就是他们对应的顺序
        tp_jobs.sort(key=lambda x: x[0], reverse=True)
        sorted_jobs = [tp_job[1] for tp_job in tp_jobs]
        return sorted_jobs

    def NEH_insertion(self, sequence, job):
        """
        Try all possible position and insert the job into the best position.

        :param sequence: a not completed solution
        :param job: the inserted job
        :return: pi+job for the best position
        """
        assert job not in sequence
        fitness_ls = []
        for j in range(len(sequence) + 1):
            current_sequence = sequence.copy()
            current_sequence.insert(j, job)
            total_processing_time, _ = self.instance.forward_schedule(current_sequence)
            fitness_ls.append(total_processing_time)
        sequence.insert(fitness_ls.index(min(fitness_ls)), job)
        return sequence

    def NEH_heuristic(self, guiding_solution=None):
        """
        Nawaz, Enscore, Ham (NEH) Algorithm, 1983. Sort first and insert the job one by one.

        :param guiding_solution: a guiding solution
        :return: a NEH solution and its forward scheduling result
        """
        if guiding_solution is None:
            guiding_solution = self.sort_the_jobs()
        current_sequence = [guiding_solution[0]]
        for i in range(1, len(guiding_solution)):
            current_sequence = self.NEH_insertion(current_sequence, guiding_solution[i])
        return current_sequence

    def reversed_NEH_heuristic(self):
        decrease_solution = self.sort_the_jobs()
        increase_solution = decrease_solution[::-1]
        return self.NEH_heuristic(increase_solution)


if __name__ == '__main__':
    fsp = FSP()
    greedy_solver = GreedyAlgorithms(fsp)
    neh_sequence = greedy_solver.NEH_heuristic()
    makespan, job_info = fsp.forward_schedule(neh_sequence)
    fsp.draw_gant_chart(job_machine_infos=job_info, method='NEH', C_max=makespan,
                        description='n_jobs {} - n_machines {}'.format(
                            len(set([x[0] for x in job_info])),
                            len(set([x[1] for x in job_info]))))
    print('Makespan:', makespan, ';\tSolution:', fsp.drop_free_jobs(neh_sequence))
    reversed_neh_sequence = greedy_solver.reversed_NEH_heuristic()
    makespan, job_info = fsp.forward_schedule(reversed_neh_sequence)
    fsp.draw_gant_chart(job_machine_infos=job_info, method='Reversed NEH', C_max=makespan,
                        description='n_jobs {} - n_machines {}'.format(
                            len(set([x[0] for x in job_info])),
                            len(set([x[1] for x in job_info]))))
    print('Makespan:', makespan, ';\tSolution:', fsp.drop_free_jobs(reversed_neh_sequence))
