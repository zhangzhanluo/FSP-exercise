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


class GA:
    """
    遗传算法求解工具，可以对染色体长度、种群数量，交叉概率，遗传概率进行设定。
    一个简单的使用例子为：

    from ga import GA

    ga_solver = GA()
    records = ga_solver.revolution()
    best_x, best_y = ga_solver.get_best_result(records[-1])
    """

    def __init__(self, instance: FSP, population_size=100, crossover_rate=0.8, mutation_rate=0.003, n_generations=100):
        self.fsp = instance
        self.dna_size = instance.n_total_jobs
        self.pop_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_generations = n_generations
        self.best_individual = None
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

    def select(self, pop, fit, method='championship'):
        if method == 'championship':
            new_pop = []
            for _ in range(self.pop_size):
                i_1 = random.randint(0, self.fsp.n_total_jobs-1)
                i_2 = random.randint(0, self.fsp.n_total_jobs-1)
                selected_ind = pop[i_1] if fit[i_1] < fit[i_2] else pop[i_2]
                new_pop.append(selected_ind)
        elif method == 'roulette_wheel':
            fit_probability = [max(fit) - x for x in fit]
            new_pop = random.choices(pop, weights=fit_probability, k=self.pop_size)
        else:
            raise NameError('No {} selection method defined!'.format(method))
        return new_pop

    def mutation(self, child):
        if random.random() < self.mutation_rate:  # 以MUTATION_RATE的概率进行变异
            mutate_point_1 = random.randint(0, self.dna_size-1)  # 随机产生一个实数，代表要变异基因的位置
            mutate_point_2 = random.randint(0, self.dna_size-1)
            child[mutate_point_1], child[mutate_point_2] = child[mutate_point_2], child[mutate_point_1]
        return child

    def crossover_and_mutation(self, pop):
        new_pop = []
        for father in pop:  # 遍历种群中的每一个个体，将该个体作为父亲
            child = father  # 孩子先得到父亲的全部基因（这里我把一串二进制串的那些0，1称为基因）
            if random.random() < self.crossover_rate:  # 产生子代时不是必然发生交叉，而是以一定的概率发生交叉
                mother = pop[random.randint(0, self.pop_size-1)]  # 在种群中选择另一个个体，并将该个体作为母亲
                cross_points = random.randint(0, self.dna_size-1)  # 随机产生交叉的点
                child[cross_points:] = mother[cross_points:]  # 孩子得到位于交叉点后的母亲的基因
            child = self.mutation(child)  # 每个后代有一定的机率发生变异
            new_pop.append(child)
        return new_pop

    def revolution(self, random_seed=None, time_limit=1e10):
        random.seed(random_seed)
        pop_records = []
        # 初始化种群
        pop = self.initial_pop()
        pop_records.append(pop)
        # 演化
        start_time = time.time()
        for _ in range(self.n_generations):
            if time.time() - start_time > time_limit:
                break
            # 评估群体中个体的适应度
            fit = self.get_fitness(pop)
            self.best_individual = pop[fit.index(min(fit))].copy()
            print('Makespan:', min(fit), ';\tSolution:', self.best_individual)
            # 选择
            pop = self.select(pop, fit)
            # 交叉和变异
            pop = self.crossover_and_mutation(pop)
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
        plt.ylabel('Target Function Values')
        plt.xticks(range(0, len(fitness_records), 5), range(0, len(fitness_records), 5))
        plt.text(plt.gca().get_xlim()[-1], plt.gca().get_ylim()[0],
                 'DNA Size: {}\nPopulation Size: {}\nCrossover Rate: {}\nMutation Rate: {}'.format(
                     self.dna_size, self.pop_size, self.crossover_rate, self.mutation_rate
                 ), ha='right', va='bottom')
        plt.tight_layout()
        plt.savefig(self.pic_path+'GA_Revolution.png', dpi=150)


if __name__ == '__main__':
    fsp = FSP()
    ga_solver = GA(instance=fsp, population_size=50)
    pop_history = ga_solver.revolution(random_seed=1)
    ga_solver.plot_evolution(pop_history)
