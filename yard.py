import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from matplotlib.path import Path
import matplotlib.patheffects as pe
import numpy as np
import warnings

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
except:
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
warnings.filterwarnings('ignore', category=UserWarning)

# 箱区坐标列表，格式: (箱区编号, 左上角X, 左上角Y, 右下角X, 右下角Y)
yard_coords = [
    # A区 (A01-09, A13)
    ("A01", 830, 100, 1000, 180),
    ("A02", 830, 200, 970, 280),
    ("A03", 830, 300, 950, 380),
    ("A04", 830, 400, 930, 480),
    ("A05", 830, 500, 910, 580),
    ("A06", 830, 600, 890, 680),
    ("A07", 830, 700, 870, 780),
    ("A08", 830, 800, 850, 880),
    ("A09", 830, 900, 850, 980),
    ("A12", 1050, 300, 1060, 1500),
    ("A13", 200, 1000, 230, 1300),

    # B区 (B01-15) - 竖向单列排列
    ("B01", 550, 100, 750, 180),
    ("B02", 550, 200, 750, 280),
    ("B03", 550, 300, 750, 380),
    ("B04", 550, 400, 750, 480),
    ("B05", 550, 500, 750, 580),
    ("B06", 550, 600, 750, 680),
    ("B07", 550, 700, 750, 780),
    ("B08", 550, 800, 750, 880),
    ("B09", 550, 900, 750, 980),
    ("B10", 550, 1000, 750, 1080),
    ("B11", 550, 1100, 750, 1180),
    ("B12", 550, 1200, 750, 1280),
    ("B13", 550, 1300, 750, 1380),
    ("B14", 550, 1400, 650, 1460),
    ("B15", 550, 1500, 600, 1620),

    # C区 (C01-20) - 竖向单列排列
    ("C01", 300, 100, 460, 180),
    ("C02", 300, 200, 460, 280),
    ("C03", 300, 300, 460, 380),
    ("C04", 300, 400, 460, 480),
    ("C05", 300, 500, 460, 580),
    ("C06", 300, 600, 460, 680),
    ("C07", 300, 700, 460, 780),
    ("C08", 300, 800, 460, 880),
    ("C09", 300, 900, 460, 980),
    ("C10", 300, 1000, 460, 1080),
    ("C11", 300, 1100, 460, 1180),
    ("C12", 300, 1200, 460, 1280),
    ("C13", 300, 1300, 460, 1380),
    ("C14", 300, 1400, 460, 1480),
    ("C16", 300, 1500, 380, 1600),
    ("C17", 300, 1620, 380, 1700),
    ("C18", 300, 1720, 440, 1800),
    ("C19", 300, 1820, 440, 1900),
    ("C20", 300, 1920, 440, 2000),

    # D区 (D14-19)
    ("D14", 140, 300, 220, 380),
    ("D15", 140, 400, 220, 480),
    ("D16", 140, 500, 220, 580),
    ("D17", 140, 600, 220, 680),
    ("D18", 140, 700, 220, 780),
    ("D19", 140, 800, 220, 880),
]
# 轨道宽度（像素单位）
TRACK_WIDTH = 5
# 轨道间距（像素单位）
TRACK_SPACING = 8

def generate_yard_model(yard_coords, track_width=TRACK_WIDTH, track_spacing=TRACK_SPACING):
    """生成堆场模型，包括箱区和轨道"""
    yards = []
    tracks = []
    track_id = 1

    for yard_id, x1, y1, x2, y2 in yard_coords:
        width = x2 - x1
        height = y2 - y1
        yards.append({
            'id': yard_id,
            'x': x1,
            'y': y1,
            'width': width,
            'height': height,
            'center_x': x1 + width / 2,
            'center_y': y1 + height / 2
        })

        # 顶部轨道：1条分为左右两段
        track_y_top = y1 - track_width - track_spacing
        # 左段
        tracks.append({
            'id': f'T{track_id}',
            'name': f'{yard_id}_T_Left',
            'type': 'horizontal',
            'x1': x1,
            'y1': track_y_top,
            'x2': x1 + width / 2,
            'y2': track_y_top,
            'width': track_width
        })
        track_id += 1
        # 右段
        tracks.append({
            'id': f'T{track_id}',
            'name': f'{yard_id}_T_Right',
            'type': 'horizontal',
            'x1': x1 + width / 2,
            'y1': track_y_top,
            'x2': x2,
            'y2': track_y_top,
            'width': track_width
        })
        track_id += 1

        # 底部轨道：1条分为左右两段
        track_y_bottom = y2 + track_spacing
        # 左段
        tracks.append({
            'id': f'T{track_id}',
            'name': f'{yard_id}_B_Left',
            'type': 'horizontal',
            'x1': x1,
            'y1': track_y_bottom,
            'x2': x1 + width / 2,
            'y2': track_y_bottom,
            'width': track_width
        })
        track_id += 1
        # 右段
        tracks.append({
            'id': f'T{track_id}',
            'name': f'{yard_id}_B_Right',
            'type': 'horizontal',
            'x1': x1 + width / 2,
            'y1': track_y_bottom,
            'x2': x2,
            'y2': track_y_bottom,
            'width': track_width
        })
        track_id += 1

        # 左侧轨道：1条完整的垂直轨道
        track_x_left = x1 - track_width
        tracks.append({
            'id': f'T{track_id}',
            'name': f'{yard_id}_L',
            'type': 'vertical',
            'x1': track_x_left,
            'y1': y1,
            'x2': track_x_left,
            'y2': y2,
            'width': track_width
        })
        track_id += 1

        # 右侧轨道：1条完整的垂直轨道
        track_x_right = x2
        tracks.append({
            'id': f'T{track_id}',
            'name': f'{yard_id}_R',
            'type': 'vertical',
            'x1': track_x_right,
            'y1': y1,
            'x2': track_x_right,
            'y2': y2,
            'width': track_width
        })
        track_id += 1

    return yards, tracks

def visualize_yard_model(yards, tracks, figsize=(24, 28), show_tracks=True, show_labels=True):
    fig, ax = plt.subplots(figsize=figsize)
    # 定义区域颜色
    area_colors = {
        'A': 'blue',
        'B': 'green',
        'C': 'coral',
        'D': 'yellow'
    }

    # 绘制箱区
    for yard in yards:
        area = yard['id'][0]  # 获取区域标识（A/B/C/D）
        rect = Rectangle(
            (yard['x'], yard['y']),
            yard['width'],
            yard['height'],
            fill=True,
            facecolor=area_colors.get(area, 'lightgray'),
            edgecolor='darkblue',
            linewidth=1,
            alpha=0.7
        )
        ax.add_patch(rect)

        # 添加箱区标签
        if show_labels:
            ax.text(
                yard['center_x'],
                yard['center_y'],
                yard['id'],
                ha='center',
                va='center',
                fontsize=7,
                fontweight='bold',
                color='darkblue',
                backgroundcolor='white',
                alpha=0.8
            )

    # 绘制轨道
    if show_tracks:
        for track in tracks:
            if track['type'] == 'horizontal':
                # 水平轨道
                ax.plot([track['x1'], track['x2']], [track['y1'], track['y2']],
                        color='red', linewidth=1, alpha=0.8)
            else:
                # 垂直轨道
                ax.plot([track['x1'], track['x2']], [track['y1'], track['y2']],
                        color='red', linewidth=1, alpha=0.8)

            # 添加轨道标签（简化显示）
            if show_labels:
                if track['type'] == 'horizontal':
                    label_x = (track['x1'] + track['x2']) / 2
                    label_y = track['y1']
                else:
                    label_x = track['x1']
                    label_y = (track['y1'] + track['y2']) / 2

                ax.text(
                    label_x,
                    label_y,
                    track['name'],
                    ha='center',
                    va='center',
                    fontsize=4,
                    fontweight='bold',
                    color='darkred',
                    path_effects=[pe.withStroke(linewidth=1, foreground='white')]
                )

    # 设置坐标轴范围
    x_min = min(yard['x'] for yard in yards) - 40
    x_max = max(yard['x'] + yard['width'] for yard in yards) + 40
    y_min = min(yard['y'] for yard in yards) - 40
    y_max = max(yard['y'] + yard['height'] for yard in yards) + 40
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # 添加标题和图例
    ax.set_title('二期堆场模型', fontsize=18)
    ax.set_xlabel('X 坐标（像素）')
    ax.set_ylabel('Y 坐标（像素）')

    # 显示网格
    ax.grid(True, linestyle='--', alpha=0.5)

    # 添加区域图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color, markersize=10, label=f'{area}区')
        for area, color in area_colors.items()
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig('yard_model_segmented_tracks.png', dpi=300, bbox_inches='tight')
    plt.show()


def export_yard_data(yards, tracks, filename='yard_data_segmented_tracks.csv'):
    """导出堆场数据到CSV文件"""
    import pandas as pd
    yards_data = [
        {'type': 'yard', 'id': y['id'], 'x': y['x'], 'y': y['y'],
         'width': y['width'], 'height': y['height']}
        for y in yards
    ]
    tracks_data = [
        {'type': 'track', 'id': t['id'], 'name': t['name'], 'track_type': t['type'],
         'x1': t['x1'], 'y1': t['y1'], 'x2': t['x2'], 'y2': t['y2'],
         'width': t['width']}
        for t in tracks
    ]
    all_data = pd.DataFrame(yards_data + tracks_data)
    all_data.to_csv(filename, index=False)
    print(f"数据已导出到 {filename}")


def analyze_track_distribution(yards, tracks):
    """分析轨道分布统计"""
    yard_track_count = {}
    yard_track_details = {}

    for track in tracks:
        # 从轨道名称中提取箱区ID
        yard_id = track['name'].split('_')[0]
        if yard_id not in yard_track_count:
            yard_track_count[yard_id] = 0
            yard_track_details[yard_id] = []
        yard_track_count[yard_id] += 1
        yard_track_details[yard_id].append(track['name'])

    print("=== 轨道分布统计 ===")
    print(f"总箱区数: {len(yards)}")
    print(f"总轨道数: {len(tracks)}")

    # 检查每个箱区的轨道数
    incorrect_yards = []
    for yard in yards:
        track_count = yard_track_count.get(yard['id'], 0)
        if track_count != 6:
            incorrect_yards.append(f"{yard['id']}({track_count}条)")


    # 按区域统计
    area_stats = {}
    for yard in yards:
        area = yard['id'][0]
        if area not in area_stats:
            area_stats[area] = {'yards': 0, 'tracks': 0}
        area_stats[area]['yards'] += 1
        area_stats[area]['tracks'] += yard_track_count.get(yard['id'], 0)

    print("\n=== 按区域统计 ===")
    for area, stats in area_stats.items():
        print(f"{area}区: {stats['yards']}个箱区, {stats['tracks']}条轨道")

    # 显示轨道分段详情（前3个箱区作为示例）
    print("\n=== 轨道分段示例 (前3个箱区) ===")
    for i, yard in enumerate(yards[:3]):
        yard_id = yard['id']
        tracks_list = yard_track_details.get(yard_id, [])
        print(f"{yard_id}: {tracks_list}")


# 主函数
if __name__ == "__main__":
    # 生成堆场模型
    yards, tracks = generate_yard_model(yard_coords)
    # 分析轨道分布
    analyze_track_distribution(yards, tracks)
    # 可视化模型
    visualize_yard_model(yards, tracks, show_tracks=True, show_labels=True)
    # 导出数据
    export_yard_data(yards, tracks)
    print(f"\n=== 最终统计 ===")
    print(f"生成了 {len(yards)} 个箱区和 {len(tracks)} 条轨道")
