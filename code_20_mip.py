"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210616
    Description: Mixed Integer Programming
"""
import time
import gurobipy as grb
from code_10_fsp import FSP


class MIP:
    def __init__(self, instance: FSP):
        """
        MIP model.

        :param instance: FSP instance
        """
        self.instance = instance
        self.n_jobs = instance.n_jobs
        self.n_machines = instance.n_machines
        self.p = instance.processing_time
        self.obj_val = None
        self.model_build_time = None
        self.model_solve_time = None
        self.job_infos = []

    def solve(self, time_limit=None):
        """
        solve the gurobi model.

        :param time_limit: time limit for Gurobi model solving
        :return: None
        """
        Q = 10000.0  # a very large number
        try:
            start_time = time.time()
            # Create a new model
            m = grb.Model("mip_fsp")

            # Create variables
            s = m.addVars(self.n_jobs, self.n_machines, lb=0, vtype=grb.GRB.CONTINUOUS, name='s')
            x = m.addVars(self.n_jobs, self.n_jobs, vtype=grb.GRB.BINARY, name='y')

            c_max = m.addVar(vtype=grb.GRB.CONTINUOUS, name='c_max', obj=1)

            # Add constrains
            m.addConstrs(
                (s[i, self.n_machines - 1] + self.p[i][self.n_machines - 1] <= c_max for i in range(self.n_jobs)),
                name='(1)')
            m.addConstrs(
                (s[i, k] + self.p[i][k] <= s[i, k + 1] for i in range(self.n_jobs) for k in range(self.n_machines - 1)),
                name='(2)')

            # Here, both solution 1 and solution 2 are right
            # %% solution 1 start
            m.addConstrs(
                (s[j, k] - (s[i, k] + self.p[i][k]) + Q * (1 - x[i, j]) >= 0
                 for i in range(self.n_jobs)
                 for j in range(self.n_jobs)
                 for k in range(self.n_machines)
                 if i < j),
                name='(3)')
            m.addConstrs(
                (s[i, k] - (s[j, k] + self.p[j][k]) + Q * x[i, j] >= 0
                 for i in range(self.n_jobs)
                 for j in range(self.n_jobs)
                 for k in range(self.n_machines)
                 if i < j),
                name='(4)')
            # %% solution 1 end
            # %% solution 2 start
            # m.addConstrs(
            #     (s[j, k] - (s[i, k] + self.p[i][k]) + Q * (1 - x[i, j]) >= 0
            #      for i in range(self.n_jobs)
            #      for j in range(self.n_jobs)
            #      for k in range(self.n_machines)
            #      if j != i),
            #     name='(3)')
            # m.addConstrs((x[i, j] + x[j, i] == 1
            #               for i in range(self.n_jobs)
            #               for j in range(self.n_jobs)
            #               if j != i))
            # %% solution 2 end

            # Set objective
            m.modelSense = grb.GRB.MINIMIZE
            model_building_time = time.time() - start_time

            # solve the model
            start_time = time.time()
            if time_limit is not None:
                m.setParam('TimeLimit', time_limit)
            m.optimize()
            model_solving_time = time.time() - start_time

            # print results
            print('Obj:', m.objVal)

            self.obj_val = m.objVal
            self.model_build_time = model_building_time
            self.model_solve_time = model_solving_time

            for i in range(self.n_jobs):
                for k in range(self.n_machines):
                    self.job_infos.append((i + 1, 'Machine {}'.format(k+1), s[i, k].x, self.p[i][k]))

        except grb.GurobiError:
            print('Error reported')

    def save_gantt_chart(self):
        """
        Save the gantt chart. Must be used after MIP.solve().

        :return: None
        """
        self.instance.draw_gant_chart(self.job_infos, method='MIP', C_max=self.obj_val,
              description='n_jobs {} - n_machines {} - solving time {:.1f}s'.format(self.n_jobs,
                                                                                    self.n_machines,
                                                                                    self.model_solve_time))


if __name__ == '__main__':
    fsp = FSP(n_jobs=13, n_machines=8)
    mip_model = MIP(fsp)
    mip_model.solve(time_limit=600)
    mip_model.save_gantt_chart()
