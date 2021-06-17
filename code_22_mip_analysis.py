"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210617
    Description: Analyse results of MIP experiments
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

file_dir = '02_Results/MIP/MIP experiments.csv'
results = pd.read_csv(file_dir, header=0)
results_grouper = results.groupby('n_machines')
machines = []
building_times = []
solving_times = []
for n_machine, group in results_grouper:
    machines.append(n_machine)
    building_times.append(group['model_building_time'].tolist()[:16])
    solving_times.append(group['model_solving_time'].tolist()[:16])

fig_size = (6, 4)
font_size_large = 15
font_size_medium = 12
font_size_small = 10

lines = pd.DataFrame(solving_times).T.plot(figsize=fig_size)
lines = lines.lines
plt.yscale('log')
plt.xlabel('Number of Jobs', fontsize=font_size_medium)
plt.ylabel('Model Solving Time (s)', fontsize=font_size_medium)
plt.xticks(range(len(solving_times[0])), range(1, len(solving_times[0]) + 1), fontsize=font_size_small)
plt.yticks(fontsize=font_size_small)
plt.legend(lines, ['Number of Machine(s): {}'.format(i) for i in machines], fontsize=font_size_small)
title = 'Solving Time of MIP Experiments'
plt.title(title, fontsize=font_size_large)
plt.tight_layout()
plt.savefig('02_Results/MIP/{}.png'.format(title), dpi=300)
plt.show()

machines = []
building_times = []
solving_times = []
for n_machine, group in results_grouper:
    machines.append(n_machine)
    building_times.append(group['model_building_time'].tolist())
    solving_times.append(group['model_solving_time'].tolist())
machine_8_solving_time = solving_times[machines.index(8)]

x = np.arange(1, len(machine_8_solving_time) + 1)
y = np.array(machine_8_solving_time)

f_polyfit = np.polyfit(x, y, 5)
print('f_polyfit:\n', f_polyfit)
y_polyfit_val = np.polyval(f_polyfit, x)
f_polyfit_expression = 'y='
for i in range(len(f_polyfit)-1):
    f_polyfit_expression += '{}x^{}+'.format(round(f_polyfit[i], 3), len(f_polyfit) - i - 1)
f_polyfit_expression += '{:.3f}'.format(f_polyfit[-1])
f_polyfit_expression = f_polyfit_expression.replace('+-', '-')

y_log = np.log(y + 0.0001)
f_logfit = np.polyfit(x, y_log, 1)
print('f_logfit:\n', f_logfit)
y_logfit_val = np.exp(np.polyval(f_logfit, x))
f_logfit_expression = 'y={:.7f}*{:.3f}^x'.format(np.exp(f_logfit[-1]), np.exp(f_logfit[0]))

plt.figure(figsize=fig_size)
plt.scatter(range(len(machine_8_solving_time)), machine_8_solving_time, label='Original Values')
plt.plot(y_polyfit_val, '-', label='${}$'.format(f_polyfit_expression))
plt.plot(y_logfit_val, '--', label='${}$'.format(f_logfit_expression))
plt.xlabel('Number of Jobs', fontsize=font_size_medium)
plt.ylabel('Model Solving Time (s)', fontsize=font_size_medium)
plt.xticks(range(len(solving_times[0])), range(1, len(solving_times[0]) + 1), fontsize=font_size_small)
plt.yticks(fontsize=font_size_small)
plt.legend(fontsize=font_size_small-1)
title = 'Solving Time of MIP Experiments (8 Machines)'
plt.title(title, fontsize=font_size_large)
plt.tight_layout()
plt.savefig('02_Results/MIP/{}.png'.format(title), dpi=300)
plt.show()
