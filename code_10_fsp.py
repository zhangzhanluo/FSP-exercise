"""
    Author: Zhanluo Zhang
    Author E-mail: zhangzhanluo@outlook.com
    Version: v1.0
    Created Date: 20210615
    Description: Create Flow Shop Problem Instance
"""
import os
from matplotlib import pyplot as plt


class FSP:
    def __init__(self):
        """
        Flow Shop Problem Instance.

        An usage example:
            fsp_instance = FSP()
            makespan, job_info = fsp_instance.forward_schedule()
            fsp_instance.draw_gant_chart(job_machine_infos=job_info, method='default', C_max=makespan)
        """
        self.n_jobs = None
        self.n_machines = None
        self.processing_time = []
        self.load_data()

    def load_data(self):
        """
        Load processing time and get the number of jobs and machines.

        :return: None
        """
        with open('00_Data/Processing Time.csv', 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                line = line.strip('\n')
                line = line.split(',')[1:]
                self.processing_time.append([int(x) for x in line])
        self.n_jobs = len(self.processing_time)
        self.n_machines = len(self.processing_time[0])

    def forward_schedule(self, sequence=None):
        """
        Unlimited buffers. More details can be get from README.md.

        :param sequence: solution
        :return: makespan and job scheduling information for gantt chart
        """
        if sequence is None:
            sequence = list(range(self.n_jobs))
        machines_finishing_time = [-1 for _ in range(self.n_machines)]
        machines_job = [-1 for _ in range(self.n_machines)]
        buffers = [[] for _ in range(self.n_jobs + 1)]
        buffers[0] = sequence.copy()
        clock = 0
        job_machine_info = []
        while True:
            for i in range(self.n_machines):
                if machines_finishing_time[i] <= clock:  # 机器任务结束
                    if machines_job[i] != -1:  # 如果正在加工任务，机器释放任务
                        buffers[i + 1].append(machines_job[i])
                        machines_job[i] = -1  # 重要，机器结束加工后应及时更正为没有任务在被加工
                        machines_finishing_time[i] = -1
                    if len(buffers[i]) > 0:  # 如果该机器的buffer有任务要加工，就开始加工下一个任务
                        job = buffers[i].pop(0)
                        machines_job[i] = job
                        machines_finishing_time[i] = clock + self.processing_time[job][i]
                        job_machine_info.append((job, 'Machine {}'.format(i + 1), clock, self.processing_time[job][i]))
                    else:  # 如果没有要加工的任务，就把机器完成时间设定为-1，表示空闲，下次遍历时会处理
                        machines_finishing_time[i] = -1
            all_finishing_time = set(machines_finishing_time)
            if -1 in all_finishing_time:
                all_finishing_time.remove(-1)
            if len(all_finishing_time) == 0:  # 任务终止条件是所有机器都空闲了，这也意味着缓冲区一定也没有可加工的内容了
                break
            else:
                clock = min(all_finishing_time)  # 寻找下一个系统改变的时刻
        return clock, job_machine_info

    @staticmethod
    def draw_gant_chart(job_machine_infos=None, ax=None, method=None, C_max=None, dpi=300):
        """
        Draw gant chart.

        :param job_machine_infos:job scheduling information
        :param ax: optional
        :param method: algorithm name
        :param C_max: makespan
        :param dpi: optional, parameter for saving the pic
        :return: None
        """
        machines_name = list(set([job_machine_info[1] for job_machine_info in job_machine_infos]))
        machines_name.sort()
        if ax is None:
            plt.figure(figsize=(20, 5))
            ax = plt.gca()
        for block in job_machine_infos:
            y = machines_name.index(block[1])
            rect = ax.barh(y, left=block[2], width=block[3], height=0.6, color='white', edgecolor='black', alpha=0.8)
            ax.text(rect[0].xy[0] + rect[0]._width * 0.5, rect[0].xy[1] + rect[0]._height * 0.5, block[0], ha='center',
                    va='center')
        plt.yticks(range(len(machines_name)), machines_name)
        ax.text(1, 0, 'https://github.com/zhangzhanluo/a-FSP-exercise', ha='right', va='bottom',
                fontsize=4, transform=ax.transAxes)
        ax.set_xlabel('Time')
        ax.set_ylabel('Machine')

        title = 'Method {} - makespan {}'.format(method, C_max)
        ax.set_title(title)
        plt.tight_layout()
        save_path = '01_Pics/{}/'.format(method)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.savefig(save_path + title + '.png', dpi=dpi)
        plt.close()


if __name__ == '__main__':
    fsp_instance = FSP()
    makespan, job_info = fsp_instance.forward_schedule()
    fsp_instance.draw_gant_chart(job_machine_infos=job_info, method='default', C_max=makespan)
