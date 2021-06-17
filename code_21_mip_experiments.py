"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210616
    Description: some mip experiments
"""
import os
from code_20_mip import MIP, FSP

if not os.path.exists('02_Results/MIP/MIP experiments.csv'):
    with open('02_Results/MIP/MIP experiments.csv', 'w') as f:
        f.write(','.join(['n_jobs', 'n_machines', 'model_building_time', 'model_solving_time']))
        f.write('\n')

# for n_jobs in range(15, 18):
#     for n_machines in range(1, 9):
#         fsp = FSP(n_jobs=n_jobs, n_machines=n_machines)
#         mip = MIP(fsp)
#         mip.solve()
#         with open('02_Results/MIP/MIP experiments.csv', 'a') as f:
#             f.write(','.join([str(n_jobs), str(n_machines), str(mip.model_build_time), str(mip.model_solve_time)]))
#             f.write('\n')

fsp = FSP(n_jobs=17, n_machines=8)
mip = MIP(fsp)
mip.solve()
with open('02_Results/MIP/MIP experiments.csv', 'a') as f:
    f.write(','.join([str(17), str(8), str(mip.model_build_time), str(mip.model_solve_time)]))
    f.write('\n')
