"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210623
    Description: Genetic algorithm
"""
import os
import time
import random
from matplotlib import pyplot as plt
from code_10_fsp import FSP
from code_30_greedy_algorithms import GreedyAlgorithms


class GA:
    """
    遗传算法求解工具，可以对染色体长度、种群数量，交叉概率，遗传概率进行设定。
    一个简单的使用例子为：

    from ga import GA

    ga_solver = GA()
    records = ga_solver.revolution()
    best_x, best_y = ga_solver.get_best_result(records[-1])
    """

    def __init__(self, instance: FSP, population_size=100, crossover_rate=0.8, mutation_rate=0.003, n_generations=100,
                 selection_method='championship', crossover_method='PMX', random_seed=None, good_start=False,
                 education=False):
        self.fsp = instance
        self.dna_size = instance.n_total_jobs
        self.pop_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.best_individual = None
        self.best_makespan = None
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.random_seed = random_seed
        self.school = []
        self.good_start = good_start
        self.education = education
        self.pic_path = '02_Results/GA/'
        for path in [self.pic_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def get_fitness(self, pop):
        """
        计算适应度。

        :param pop: 种群所有个体的DNA编码。list of lst
        :return: 种群中所有个体的适应度。np.array: (POP_SIZE, 1)
        """
        makespans = []
        for individual in pop:
            makespan, _ = self.fsp.forward_schedule(individual.copy())
            makespans.append(makespan)
        return makespans

    def initial_pop(self):
        """
        种群初始化。

        :return: 种群所有个体的DNA编码。np.array: (POP_SIZE, DNA_SIZE)
        """
        solution = list(range(1, self.fsp.n_total_jobs + 1))
        pop = []
        for _ in range(self.pop_size):
            tmp = solution.copy()
            random.shuffle(tmp)
            pop.append(tmp)
        return pop

    def select(self, pop, fit):
        if self.selection_method == 'Championship':
            new_pop = []
            for _ in range(self.pop_size):
                i_1 = random.randint(0, self.fsp.n_total_jobs - 1)
                i_2 = random.randint(0, self.fsp.n_total_jobs - 1)
                selected_ind = pop[i_1] if fit[i_1] < fit[i_2] else pop[i_2]
                new_pop.append(selected_ind)
        elif self.selection_method == 'Roulette Wheel':
            fit_probability = [max(fit) - x for x in fit]
            new_pop = random.choices(pop, weights=fit_probability, k=self.pop_size)
        else:
            raise NameError('No {} selection method defined!'.format(self.selection_method))
        return new_pop

    def mutation(self, child):
        if random.random() < self.mutation_rate:  # 以MUTATION_RATE的概率进行变异
            mutate_point_1 = random.randint(0, self.dna_size - 1)  # 随机产生一个实数，代表要变异基因的位置
            mutate_point_2 = random.randint(0, self.dna_size - 1)
            child[mutate_point_1], child[mutate_point_2] = child[mutate_point_2], child[mutate_point_1]
        return child

    @staticmethod
    def pmx(mother_middle_part, father_middle_part, x):
        while x in mother_middle_part:
            x = father_middle_part[mother_middle_part.index(x)]
        return x

    def educate(self, child):
        child_makespan, _ = self.fsp.forward_schedule(child.copy())
        if child_makespan - self.best_makespan < 10 and child not in self.school:
            self.school.append(child.copy())
            for i in range(self.dna_size):
                for j in range(i, self.dna_size):
                    educated_child = child.copy()
                    educated_child[i], educated_child[j] = child[j], child[i]
                    educated_makespan, _ = self.fsp.forward_schedule(educated_child.copy())
                    if educated_makespan < child_makespan:
                        return educated_child
        return child

    def crossover_and_mutation(self, pop):
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if random.random() < self.crossover_rate:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[random.randint(0, self.pop_size - 1)]  # 在种群中选择另一个个体，并将该个体作为母亲
                cross_points = [random.randint(0, self.dna_size - 1), random.randint(0, self.dna_size - 1)]
                cross_points = [min(cross_points), max(cross_points)]
                mother_middle_part = mother[cross_points[0]:cross_points[1]]
                father_middle_part = father[cross_points[0]:cross_points[1]]
                if self.crossover_method == 'PMX':
                    child_part_1 = [
                        x if x not in mother_middle_part else self.pmx(mother_middle_part, father_middle_part, x)
                        for x in father[:cross_points[0]]]
                    child_part_3 = [
                        x if x not in mother_middle_part else self.pmx(mother_middle_part, father_middle_part, x)
                        for x in father[cross_points[1]:]]
                    child = child_part_1 + mother_middle_part + child_part_3
                elif self.crossover_method == 'OX':
                    mother_reorder = mother[cross_points[1]:] + mother[:cross_points[1]]
                    for x in father_middle_part:
                        mother_reorder.remove(x)
                    tail_len = self.dna_size - cross_points[1]
                    child = mother_reorder[tail_len:] + father_middle_part + mother_reorder[:tail_len]
                else:
                    raise NameError('No {} crossover method defined!'.format(self.crossover_method))
            child = self.mutation(child)  # 每个后代有一定的机率发生变异
            if self.education:
                child = self.educate(child)
            new_pop.append(child)
        return new_pop

    def revolution(self, time_limit=1e10):
        random.seed(self.random_seed)
        pop_records = []
        # 初始化种群
        pop = self.initial_pop()
        if self.good_start:
            greedy_algorithm_solver = GreedyAlgorithms(self.fsp)
            pop[0] = greedy_algorithm_solver.NEH_heuristic().copy()
        fit = self.get_fitness(pop)
        self.best_individual = pop[fit.index(min(fit))].copy()
        self.best_makespan = min(fit)
        pop_records.append(pop)
        # 演化
        start_time = time.time()
        for _ in range(self.n_generations):
            if time.time() - start_time > time_limit:
                break
            # 选择
            pop = self.select(pop, fit)
            # 交叉和变异
            pop = self.crossover_and_mutation(pop)
            fit = self.get_fitness(pop)
            if min(fit) < self.best_makespan:
                self.best_individual = pop[fit.index(min(fit))].copy()
                self.best_makespan = min(fit)
            else:
                pop[0] = self.best_individual.copy()
            pop_records.append(pop.copy())
        random.seed(None)
        return pop_records

    def plot_evolution(self, pops, generation_range=None, fig_size=(8, 3)):
        if generation_range is None:
            generation_range = (0, len(pops))
        plt.figure(figsize=fig_size)
        fitness_records = [self.get_fitness(x) for x in
                           pops[generation_range[0]: generation_range[1]]]
        plt.boxplot(fitness_records, labels=range(generation_range[0], generation_range[1]))
        plt.xlabel('Generation')
        plt.ylabel('Makespans')
        title = 'GA Revolution with {} Selection {} Crossover {} Good Start {} Education'.format(self.selection_method,
                                                                                                 self.crossover_method,
                                                                                                 str(self.good_start),
                                                                                                 str(self.education))
        plt.title(title)
        plt.xticks(range(0, len(fitness_records), 5), range(0, len(fitness_records), 5))
        plt.text(plt.gca().get_xlim()[-1] * 0.99, plt.gca().get_ylim()[-1] * 0.99,
                 'DNA Size: {}\n'
                 'Population Size: {}\n'
                 'Crossover Rate: {}\n'
                 'Mutation Rate: {}\n'
                 'Random Seed: {}\n\n'
                 'Best Makespan: {}'.format(
                     self.dna_size,
                     self.pop_size,
                     self.crossover_rate,
                     self.mutation_rate,
                     self.random_seed,
                     self.best_makespan
                 ), ha='right', va='top')
        plt.tight_layout()
        title = title.replace(' ', '_')
        plt.savefig(self.pic_path + '{}.png'.format(title), dpi=300)


if __name__ == '__main__':
    fsp = FSP()
    ga_solver = GA(instance=fsp, population_size=50, crossover_method='PMX', selection_method='Roulette Wheel',
                   random_seed=1, good_start=False, education=False)
    _ = ga_solver.revolution()
    print(ga_solver.best_makespan, ga_solver.best_individual)
    for crossover in ['OX', 'PMX']:
        for selection in ['Championship', 'Roulette Wheel']:
            for good_start in [True]:
                for edu in [True]:
                    ga_solver = GA(instance=fsp, population_size=50, crossover_method=crossover,
                                   selection_method=selection, random_seed=1,
                                   good_start=good_start, education=edu)
                    pop_history = ga_solver.revolution()
                    ga_solver.plot_evolution(pop_history)
                    print(crossover, selection)
                    print(ga_solver.best_makespan, ga_solver.best_individual)
    ga_solver = GA(instance=fsp, population_size=50, crossover_method='PMX', selection_method='Roulette Wheel',
                   random_seed=1, good_start=True, education=True)
    _ = ga_solver.revolution()
    _, job_info = fsp.forward_schedule(ga_solver.best_individual)
    fsp.draw_gant_chart(job_info, method='GA', C_max=ga_solver.best_makespan,
                        description='PMX_Roulette_Wheel_Good_Start_Education')
