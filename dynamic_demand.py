import matplotlib.pyplot as plt
import numpy as np
import random
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def simulate_passenger_demand():
    stop_num = 13
    RT = [2, 3, 2, 2, 3, 3, 4, 2, 3, 2, 3, 3]
    for i in range(len(RT)):
        RT[i] = RT[i] - 1
    w_ijt = [[[0 for _ in range(200)] for _ in range(stop_num)] for _ in range(stop_num)]
    for i in range(stop_num - 1):
        for j in range(i + 1, stop_num):
            w_ijt[i][j][0] = random.randint(10, 30)
    for t in range(1, 200):
        for i in range(stop_num - 1):
            time_start = sum(RT[:i]) + i
            for j in range(stop_num):
                if i < j:
                    if time_start <= t <= time_start + 30:
                        w_ijt[i][j][t] = w_ijt[i][j][t-1] + random.choices([0, 1], weights=[0.75, 0.25])[0]
                    elif time_start + 30 < t <= time_start + 60:
                        w_ijt[i][j][t] = w_ijt[i][j][t-1] + random.choices([0, 1], weights=[(120-t)/120, t/120])[0]
                    elif time_start + 60 < t <= time_start + 90:
                        w_ijt[i][j][t] = w_ijt[i][j][t-1] + random.choices([0, 1], weights=[t/120, (120-t)/120])[0]
                    elif time_start + 90 < t <= time_start + 130:
                        w_ijt[i][j][t] = w_ijt[i][j][t-1] + random.choices([0, 1], weights=[0.75, 0.25])[0]
                    else:
                        w_ijt[i][j][t] = w_ijt[i][j][t - 1]
                else:
                    w_ijt[i][j][t] = 0
    return w_ijt

w_ijt = simulate_passenger_demand()

plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 18

"""""""""
Sum of demand
"""""""""

# 计算每一分钟的总到达人数（跨所有站点 i<j）
arrival_per_minute = []
for t in range(1, 200):
    total_arrival = 0
    for i in range(13):
        for j in range(i + 1, 13):
            total_arrival += w_ijt[i][j][t] - w_ijt[i][j][t - 1]
    arrival_per_minute.append(total_arrival)

# 平滑函数（滑动平均）
def moving_average(data, window_size=5):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 应用平滑
smoothed_arrival = moving_average(arrival_per_minute, window_size=20)

# 绘图
plt.figure(figsize=(12, 8))
plt.plot(range(1, 200), arrival_per_minute, label="Passenger Arrival", alpha=0.4, linestyle='--')
plt.plot(range(10, 190), smoothed_arrival, label="Smoothed Passenger Arrival (Window=20)", linewidth=2)
plt.xlabel("Time (minute)")
plt.ylabel("New Passenger Arrivals")
# plt.title("Simulated Passenger Arrival Pattern Over Time (Sum of All Stops)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('/home2/chenghanyu/Hanyu_codes/modular_bus/trained_real_model/demand_figure/sum_of_all_stops.png')
plt.show()


"""""""""
Each stop
"""""""""

# 存储每个站点的 smoothed 到达人数序列（以站点 i 为起点）
each_stop = []

for i in range(13):
    arrival_per_minute = []
    for t in range(1, 200):
        total_arrival = 0
        for j in range(i + 1, 13):
            total_arrival += w_ijt[i][j][t] - w_ijt[i][j][t - 1]
        arrival_per_minute.append(total_arrival)

    smoothed_arrival = moving_average(arrival_per_minute, window_size=50)
    each_stop.append(smoothed_arrival)


fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(111, projection='3d')

time_range = np.arange(25, 175)               # X: 时间
spacing = 5                                   # 每个站点在 Y 轴上的间距
stop_indices = np.arange(13)
stop_positions = stop_indices * spacing       # Y: 站点 index 拉开间距

# 使用 colormap 给每条线配色
colors = cm.viridis(np.linspace(0, 1, len(stop_indices)))

for i, y in enumerate(stop_positions):
    xs = time_range
    ys = np.full_like(xs, y)
    zs = each_stop[i]
    ax.plot(xs, ys, zs, color=colors[i], linewidth=1.5)

# 设置轴标签和图例
ax.set_xlabel('Time (minute)', labelpad=10)
ax.set_ylabel('Stop Index', labelpad=10)
ax.set_zlabel('Smoothed Passenger Arrival (Window=50)', labelpad=10)
# ax.set_title('3D Line Plot of Passenger Arrival by Stop', pad=15)
ax.set_xlim(time_range[-1], time_range[0])

# 设置刻度显示实际 Stop 编号（虽然位置间距拉大了）
ax.set_yticks(stop_positions)
ax.set_yticklabels([str(i) for i in stop_indices])

# 视角和图例
ax.view_init(elev=30, azim=120)
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig('/home2/chenghanyu/Hanyu_codes/modular_bus/trained_real_model/demand_figure/passenger_arrival_at_each_step.png', bbox_inches='tight')
plt.show()