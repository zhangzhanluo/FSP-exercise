"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210622
    Description: Use random as benchmark algorithms and average of 10 random result as benchmark.
"""
from matplotlib import pyplot as plt
from code_10_fsp import FSP

makespans = []
fsp_instance = FSP()
for i in range(10):
    solution = fsp_instance.generate_random_solution(i)
    makespan, _ = fsp_instance.forward_schedule(solution)
    makespans.append(makespan)

plt.figure(figsize=(4, 6))
plt.boxplot(makespans)
plt.xticks([1], ['Random'])
plt.ylabel('Makespan (s)')
plt.tight_layout()
plt.savefig('02_Results/Random/10 random solutions.png', dpi=300)
plt.show()
