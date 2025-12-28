import matplotlib.pyplot as plt
import numpy as np

# Thiết lập style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

# Tạo figure với 2 subplot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Dữ liệu - thêm PoseFormer
methods = ['MLP-Mixer', 'GCN', 'PoseFormer', 'GraphMLP']
x = np.arange(len(methods))
bar_width = 0.5

# ===== Biểu đồ (a) Human3.6M =====
h36m_values = [52.0, 51.3, 49.6, 49.2]  # PoseFormer = 49.6
colors_h36m = ['#8B0000', '#8B0000', '#8B0000', '#8B0000']  # Dark red

bars1 = ax1.bar(x, h36m_values, width=bar_width, color=colors_h36m, edgecolor='black', linewidth=0.5)

# Thêm số liệu trên cột
for i, (bar, val) in enumerate(zip(bars1, h36m_values)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'{val}', ha='center', va='bottom', fontsize=11)

# Thêm mũi tên và chú thích độ giảm
ax1.annotate('', xy=(0, 49.2), xytext=(0, 52.0),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax1.text(0.25, 50.6, '2.8', fontsize=10, va='center')

ax1.annotate('', xy=(1, 49.2), xytext=(1, 51.3),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax1.text(1.25, 50.2, '2.1', fontsize=10, va='center')

ax1.annotate('', xy=(2, 49.2), xytext=(2, 49.6),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax1.text(2.25, 49.4, '0.4', fontsize=10, va='center')

# Đường nét đứt
ax1.axhline(y=49.2, color='gray', linestyle='--', linewidth=1)

ax1.set_ylabel('MPJPE (mm)', fontsize=12)
ax1.set_title('Human3.6M', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
ax1.set_ylim(47, 53)
ax1.set_yticks([47, 48, 49, 50, 51, 52, 53])

# ===== Biểu đồ (b) MPI-INF-3DHP =====
mpi_values = [86.7, 87.0, 81.6, 80.1]
colors_mpi = ['#505050', '#505050', '#505050', '#505050']  # Dark gray

bars2 = ax2.bar(x, mpi_values, width=bar_width, color=colors_mpi, edgecolor='black', linewidth=0.5)

# Thêm số liệu trên cột
for i, (bar, val) in enumerate(zip(bars2, mpi_values)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
             f'{val}', ha='center', va='bottom', fontsize=11)

# Thêm mũi tên và chú thích độ giảm
ax2.annotate('', xy=(0, 80.1), xytext=(0, 86.7),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax2.text(0.25, 83.4, '6.6', fontsize=10, va='center')

ax2.annotate('', xy=(1, 80.1), xytext=(1, 87.0),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax2.text(1.25, 83.5, '6.9', fontsize=10, va='center')

ax2.annotate('', xy=(2, 80.1), xytext=(2, 81.6),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.5))
ax2.text(2.25, 80.85, '1.5', fontsize=10, va='center')

# Đường nét đứt
ax2.axhline(y=80.1, color='gray', linestyle='--', linewidth=1)

ax2.set_ylabel('MPJPE (mm)', fontsize=12)
ax2.set_title('MPI-INF-3DHP', fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(methods)
ax2.set_ylim(76, 88)
ax2.set_yticks([76, 78, 80, 82, 84, 86, 88])

# Thêm nhãn (a) và (b)
ax1.text(0.5, -0.15, '(a)', transform=ax1.transAxes, ha='center', fontsize=12)
ax2.text(0.5, -0.15, '(b)', transform=ax2.transAxes, ha='center', fontsize=12)

plt.tight_layout()
plt.savefig('comparison_chart.png', dpi=300, bbox_inches='tight')
plt.savefig('comparison_chart.pdf', bbox_inches='tight')
plt.show()

print("Đã lưu biểu đồ: comparison_chart.png và comparison_chart.pdf")