import numpy as np
import random
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
import os
import math

# 箱区布局
BOX_LAYOUT = {
    1: {"row": 0, "col": 2},  # Block 1
    2: {"row": 1, "col": 2},  # Block 2
    3: {"row": 0, "col": 1},  # Block 3
    4: {"row": 1, "col": 1},  # Block 4
    5: {"row": 0, "col": 0},  # Block 5
    6: {"row": 1, "col": 0}  # Block 6
}

# 参数定义
NUM_GENERATIONS = 300 # 最大迭代次数
POPULATION_SIZE = 300  # 种群大小
CROSSOVER_RATE = 0.9  # 交叉概率
MUTATION_RATE = 0.4  # 变异概率

# 任务优先级定义
task_priority = {
    "临时重点箱": 1,
    "紧急任务": 2,
    "装船": 3,
    "卸船": 3,
    "进箱": 4,
    "提箱": 4,
}

# 场桥和箱区
num_cranes = 4  # 场桥数量
num_blocks = 6  # 箱区数量
time_slots = [0, 150, 300, 450, 600]  # 时间段划分

# 定义场桥状态枚举
STATE_MOVING = 1  # 正在移动到目的贝位
STATE_EXECUTING = 2  # 在目的贝位执行任务
STATE_IDLE = 3  # 空闲状态
STATE_FAULT = 4  #故障状态

# 尝试设置字体，如果找不到特定字体则使用默认字体
try:
    prop = fm.FontProperties(fname='C:/Windows/Fonts/simhei.ttf')  # Windows 下黑体字体路径
except:
    # 如果找不到指定字体，使用系统默认字体
    prop = fm.FontProperties()
    print("无法找到指定字体，使用系统默认字体")

# 读取Excel文件
file_path = r"D:\Desktop\青岛轮胎吊数据\调度算法\GA数据1.xlsx"
try:
    df = pd.read_excel(file_path)
    task_data = df.values.tolist()  # 将DataFrame转换为列表的列表
    task_data = sorted(task_data, key=lambda x: x[3])
    print(f"成功读取任务数据，共 {len(task_data)} 个任务")
except Exception as e:
    print(f"读取文件失败: {e}")

# 计算两个箱区之间的物理距离
def calculate_block_distance(block1, block2):
    """基于箱区布局计算两个箱区之间的物理距离"""
    if block1 == block2:
        return 0, 0  # 同一箱区，行距离和列距离都为0
    row1, col1 = BOX_LAYOUT[block1]["row"], BOX_LAYOUT[block1]["col"]
    row2, col2 = BOX_LAYOUT[block2]["row"], BOX_LAYOUT[block2]["col"]
    # 计算行距离和列距离
    row_distance = abs(row1 - row2)
    col_distance = abs(col1 - col2)
    # 返回行距离和列距离的元组
    return row_distance, col_distance

# 计算场桥的移动时间
def calculate_move_time(current_position, target_position, block_current, block_target):
    # 水平方向的移动时间：目标贝位和当前贝位的水平移动加上过道的距离
    block_row_distance, block_col_distance = calculate_block_distance(block_current, block_target)
    if block_row_distance == 0:
        horizontal_move_time = abs(current_position - target_position) * 0.28
    else:
        if current_position < 10 and target_position <10:
            horizontal_move_time = (current_position + target_position) * 0.28 + 3
        elif current_position > 10 and target_position > 10:
            horizontal_move_time = abs(current_position - 60) * 0.28+ abs(target_position - 60) * 0.28 + 3
        elif current_position > 10 and target_position < 10:
            horizontal_move_time = ((19- current_position) + target_position) * 0.28 + 3
        else:
            horizontal_move_time = (abs(19 - target_position) + current_position) * 0.28 + 3
    # 垂直方向的移动时间：基于箱区布局的物理距离计算
    if block_col_distance == 0 or block_col_distance == 1:
        vertical_move_time = block_col_distance * 10
    else:
        vertical_move_time = 13
    # 返回移动总时间
    return horizontal_move_time + vertical_move_time + 3 #加上吊具展开的时间3分钟

# 优化的移动时间计算
#  def calculate_move_time_optimized(current_pos, target_pos, current_block, target_block, task_type):
#     base_move_time = calculate_move_time(current_pos, target_pos, current_block, target_block)
#     # 根据任务类型调整移动时间
#     if task_type in ['临时重点箱','装船', '卸船']:
#         # 装卸船任务可能需要额外的精确定位时间
#         base_move_time += 2
#     return base_move_time

# 优化的初始化种群
def initialize_population_improved():
    population = []
    for _ in range(POPULATION_SIZE):
        chromosome = []
        for task in task_data:
            task_idx = task_data.index(task)
            prep_time = task[3]
            # 选择满足准备时间约束的时间段
            valid_time_slots = []
            for i in range(len(time_slots) - 1):
                if time_slots[i] >= prep_time:
                    valid_time_slots.append(i)
            # 如果有有效时间段，随机选择一个
            if valid_time_slots:
                time_slot = random.choice(valid_time_slots)
            else:
                # 如果没有有效时间段，选择最接近准备时间的时间段
                time_slot = min(range(len(time_slots) - 1),
                                key=lambda i: abs(time_slots[i] - prep_time) if time_slots[i] >= prep_time else float(
                                    'inf'))
            crane_idx = random.choice(range(num_cranes))
            chromosome.append((time_slot, crane_idx))
        population.append(chromosome)
    return population
# 基于启发式规则创建调度
def create_heuristic_schedule():
    chromosome = []
    # 按准备时间和优先级排序任务
    sorted_tasks = sorted(enumerate(task_data),
                          key=lambda x: (x[1][3], task_priority.get(x[1][5],4)))
    # 为每个箱区分配场桥
    block_crane_assignment = {}
    crane_workload = {i: 0 for i in range(num_cranes)}
    for task_idx, task in sorted_tasks:
        block = task[2]
        prep_time = task[3]
        # 选择时间段
        valid_slots = [i for i in range(len(time_slots) - 1) if time_slots[i] >= prep_time]
        time_slot = valid_slots[0] if valid_slots else len(time_slots) - 2
        # 选择场桥（优先选择已在该箱区工作的场桥）
        if block in block_crane_assignment:
            candidate_cranes = block_crane_assignment[block]
        else:
            candidate_cranes = list(range(num_cranes))
            block_crane_assignment[block] = []
        # 选择工作量最少的场桥
        crane_idx = min(candidate_cranes, key=lambda c: crane_workload[c])
        if crane_idx not in block_crane_assignment[block]:
            block_crane_assignment[block].append(crane_idx)
        crane_workload[crane_idx] += task[4] * 3  # 操作时间
        chromosome.append((time_slot, crane_idx))
    return chromosome

# 基于任务优先级创建调度
def create_priority_based_schedule():
    chromosome = []
    high_priority_tasks = []
    low_priority_tasks = []
    for i, task in enumerate(task_data):
        if task_priority.get(task[5], 4) == 1 or task_priority.get(task[5], 4)==2:
            high_priority_tasks.append((i, task))
        else:
            low_priority_tasks.append((i, task))
    # 先调度高优先级任务
    all_tasks = high_priority_tasks + low_priority_tasks
    for task_idx, task in all_tasks:
        prep_time = task[3]
        # 高优先级任务优先安排在早期时间段
        if task_priority.get(task[5], 2) == 1 or task_priority.get(task[5], 2) == 2:
            valid_slots = [i for i in range(len(time_slots) - 1) if time_slots[i] >= prep_time]
            time_slot = valid_slots[0] if valid_slots else len(time_slots) - 2
        else:
            # 低优先级任务平均分布
            valid_slots = [i for i in range(len(time_slots) - 1) if time_slots[i] >= prep_time]
            time_slot = random.choice(valid_slots) if valid_slots else len(time_slots) - 2
        crane_idx = random.randint(0, num_cranes - 1)
        chromosome.append((time_slot, crane_idx))
    return chromosome

def create_random_schedule_improved():
    """
    改进的随机调度生成
    """
    chromosome = []
    for task in task_data:
        prep_time = task[3]
        # 随机选择但倾向于较早的时间段
        valid_slots = [i for i in range(len(time_slots) - 1) if time_slots[i] >= prep_time]
        if valid_slots:
            # 使用指数分布倾向于选择较早的时间段
            weights = [2 ** (len(valid_slots) - i) for i in range(len(valid_slots))]
            time_slot = random.choices(valid_slots, weights=weights)[0]
        else:
            time_slot = len(time_slots) - 2
        crane_idx = random.randint(0, num_cranes - 1)
        chromosome.append((time_slot, crane_idx))
    return chromosome

# 改进版解码函数 - 确保每个任务都被分配且满足新约束
def decode_chromosome(chromosome):
    """解码染色体为调度方案，确保满足新约束"""
    # 创建空的调度数据结构
    schedule = {i: {c: [] for c in range(num_cranes)} for i in range(len(time_slots) - 1)}
    # 直接根据染色体分配任务
    for task_idx, gene in enumerate(chromosome):
        time_slot, crane = gene
        task = task_data[task_idx]
        # 添加任务到时间段和场桥
        schedule[time_slot][crane].append(task)
    return schedule

# 检查箱区场桥数量约束
def check_block_crane_limit(tasks_by_block_time):
    """检查每个时间点每个箱区的场桥数量是否超过2台"""
    violations = 0
    for time, blocks in tasks_by_block_time.items():
        for block, cranes in blocks.items():
            if len(cranes) > 2:  # 同一箱区最多允许2台场桥
                violations += len(cranes) - 2  # 每多一台场桥记一次违反
    return violations

# 检查场桥是否违反交叉约束;根据时间检查时间段内场桥是否冲突
def check_crane_crossing(timeline):
    violations = 0
    conflicts = []  # 存储冲突信息用于调试
    # 遍历每个箱区
    for block_num, block in timeline.items():
        block_time = []
        for crane_num, task_crane in block.items():
            for task_item in task_crane:
                task_item_copy = task_item.copy()
                task_item_copy.append(crane_num)
                block_time.append(task_item_copy)
        block_time = sorted(block_time, key=lambda x: x[0])
        for i in range(len(block_time) - 1):
            for j in range(i + 1, len(block_time)):
                # 检查两个任务是否在时间上重叠且位置上有冲突
                if (block_time[i][1] > block_time[j][0] and
                        block_time[j][1] > block_time[i][0]):
                    # 检查场桥位置是否冲突
                    crane1_pos = block_time[i][2]
                    crane2_pos = block_time[j][2]
                    crane1_idx = block_time[i][3]
                    crane2_idx = block_time[j][3]

                    if crane1_idx != crane2_idx and abs(crane1_pos - crane2_pos) < 3:
                        violations += 1
                        conflicts.append([block_num, block_time[i], block_time[j]])
    if violations > 0:
        print(f"发现场桥位置冲突，共 {violations} 处")
    return violations, conflicts

# 获取场桥下一个任务的信息
def get_next_task_info(crane_idx, current_task_end_time, task_timings, schedule):
    """获取场桥的下一个任务信息"""
    future_tasks = []
    for task_id, data in task_timings.items():
        if (data['crane_idx'] == crane_idx and
                data['planned_start_time'] >= current_task_end_time):
            future_tasks.append((task_id, data))
    # 按计划开始时间排序
    future_tasks.sort(key=lambda x: x[1]['planned_start_time'])

    # 如果有下一个任务，返回它的信息
    if future_tasks:
        next_task_id, next_task_data = future_tasks[0]
        return {
            'task_id': next_task_id,
            'target_block': next_task_data['block'],
            'target_position': next_task_data['position'],
            'planned_start_time': next_task_data['planned_start_time']
        }
    return None

# 分析冲突类型和严重程度
def analyze_conflict(current_pos, target_pos, start_time, arrival_time, end_time,
                     other_pos, other_target_pos, other_start, other_arrival, other_end):
    conflict_details = {
        'type': None,
        'severity': 0,  # 0-10的严重程度
        'description': ''
    }
    # 检查路径交叉
    if ((current_pos < other_pos and target_pos > other_target_pos) or
            (current_pos > other_pos and target_pos < other_target_pos)):
        # 移动阶段冲突
        if (start_time < other_arrival and arrival_time > other_start):
            conflict_details['type'] = 'path_crossing_movement'
            conflict_details['severity'] = 8
            conflict_details['description'] = '移动路径交叉冲突'
            return True, conflict_details
    # 检查位置争夺
    if abs(target_pos - other_target_pos) <= 3:  # 安全距离
        if (arrival_time < other_end and end_time > other_arrival):
            conflict_details['type'] = 'position_conflict'
            conflict_details['severity'] = 9
            conflict_details['description'] = '目标位置冲突'
            return True, conflict_details
    # 检查通过冲突
    if ((current_pos < other_target_pos < target_pos) or
            (target_pos < other_target_pos < current_pos)):
        if (start_time < other_end and arrival_time > other_arrival):
            conflict_details['type'] = 'pass_through_conflict'
            conflict_details['severity'] = 7
            conflict_details['description'] = '通过执行位置冲突'
            return True, conflict_details
    return False, conflict_details

# 检查任务是否变为紧急状态（距离终止时间少于20分钟）
def check_task_urgency(task_info, current_time):
    # 确保任务信息包含必要字段
    if not isinstance(task_info, dict) or 'task' not in task_info:
        return False
    task_data = task_info['task']
    if len(task_data) < 5:  # 确保数据结构包含任务终止时间
        return False
    task_deadline = task_data[4]  # 任务终止时间
    is_urgent = (task_deadline - current_time) <= 20  # 距离终止时间少于20分钟
    # 如果任务变为紧急状态，更新任务类型
    if is_urgent:
        # 转换为可变列表（如果是元组）
        if not isinstance(task_data, list):
            task_data = list(task_data)
            task_info['task'] = task_data
        # 更新任务类型为紧急任务
        task_data[5] = "紧急任务"
        task_info['task_type'] = "紧急任务"
    return is_urgent

# 时间承诺，计算任务延迟成本
def calculate_delay_cost(task_info, task_queue):
    # 基础延迟成本
    base_cost = 1.0
    # 获取任务类型和优先级
    task_type = task_info.get('task_type', "进箱")
    priority = task_priority.get(task_type, 4)
    # 根据优先级调整基础成本
    if priority == 0:  #
        base_cost *= 10.0  # 最高优先级
    elif priority == 1:  # 紧急任务
        base_cost *= 5.0
    elif priority == 2:  # 高优先级
        base_cost *= 2.0
    # 计算理想完成时间和实际完成时间
    if 'task' in task_info and isinstance(task_info['task'], list) and len(task_info['task']) > 3:
        prep_time = task_info['task'][3]
        move_time = task_info.get('move_time', 0)
        operation_time = task_info.get('operation_time', 0)
        ideal_completion = prep_time + task_info.get('move_time', 0) + task_info.get('operation_time', 0)
        # 获取实际完成时间（处理未完成任务）
        actual_completion = task_info.get('actual_end_time')
        if actual_completion is None:
            # 如果任务未完成，使用计划完成时间或当前时间
            actual_completion = task_info.get('planned_start_time', 0) + move_time + operation_time
        # 计算延迟
        delay = max(0, actual_completion - ideal_completion)
        # 延迟惩罚（只对高优先级任务）
        if priority <= 2 and delay > 15:  # 15分钟阈值
            base_cost += delay * (5 if priority == 0 else 2)  # 紧急任务惩罚更重
    else:
        delay = 0
        base_cost += delay * 0.5  # 未知任务惩罚
    # 考虑任务队列长度（排队成本）
    queue_length = len(task_queue)
    queue_cost = queue_length * 0.5
    # 考虑连锁反应（后续任务延迟成本）
    subsequent_tasks = len([t for t in task_queue
                            if t[1]['planned_start_time'] > task_info.get('planned_start_time', 0)])
    chain_cost = subsequent_tasks * 0.3
    return base_cost + chain_cost + queue_cost

# 智能冲突解决策略
def intelligent_conflict_resolution(task_data_id, task_info, comp_task, task_queue, comp_queue):
    # 获取任务优先级
    task_priority_level = task_priority.get(task_info['task'][5], 4)
    # 获取竞争任务优先级
    if 'task' in comp_task and isinstance(comp_task['task'], (list, tuple)) and len(comp_task['task']) > 5:
        comp_priority_level = task_priority.get(comp_task['task'][5], 4)
    else:
        comp_priority_level = task_priority.get(comp_task.get('task_type', '进箱'), 4)

    # 计算延迟成本
    task_delay_cost = calculate_delay_cost(task_info, task_queue)
    comp_delay_cost = calculate_delay_cost(comp_task, comp_queue)

    # 决策逻辑
    if task_priority_level < comp_priority_level:
        # 当前任务优先级更高，竞争任务等待
        return {
            'action': 'proceed',  # 改为proceed表示当前任务继续执行
            'wait_time': 0,  # 当前任务不需要等待
            'reason': 'higher_priority_task'
        }
    elif task_priority_level > comp_priority_level:
        # 竞争任务优先级更高，当前任务等待
        return {
            'action': 'wait',
            'wait_time': max(0, comp_task['end_time'] - task_info['planned_start_time']),
            'reason': 'lower_priority_task'
        }
    else:
        # 比较成本，决定哪个任务应该等待
        if task_delay_cost > comp_delay_cost:
            return {'action': 'proceed', 'wait_time': 0}
        else:
            wait_time = max(0, comp_task['end_time'] - task_info['planned_start_time'])
            return {'action': 'wait', 'wait_time': wait_time, 'reason': '成本更高'}
# 更新时间线记录
def update_timeline(block_timeline, updated_timeline, crane_idx, target_block,
                    actual_start_time, actual_arrival_time, actual_end_time,
                    current_position, target_position):
    # 更新箱区时间线
    if target_block not in block_timeline:
        block_timeline[target_block] = {}
    if crane_idx not in block_timeline[target_block]:
        block_timeline[target_block][crane_idx] = []
    block_timeline[target_block][crane_idx].append([actual_arrival_time, actual_end_time, target_position])

    # 更新全局时间线
    for time_point, position in [(actual_start_time, current_position),
                                 (actual_arrival_time, target_position),
                                 (actual_end_time, target_position)]:
        if time_point not in updated_timeline:
            updated_timeline[time_point] = {}
        updated_timeline[time_point][crane_idx] = position

# 优化的冲突检测和解决函数
def detect_and_resolve_crane_conflicts_optimized(timeline, task_timings):
    sorted_tasks = sorted(
        [(task_id, data) for task_id, data in task_timings.items()],
        key=lambda x: x[1]['planned_start_time']
    )

    # 初始化场桥状态
    crane_positions = {i: 1 for i in range(num_cranes)}
    crane_blocks = {i: None for i in range(num_cranes)}
    crane_available_times = {i: 0 for i in range(num_cranes)}
    crane_task_queue = {i: [] for i in range(num_cranes)}  # 每个场桥的任务队列

    # 构建场桥任务队列
    for task_id, data in sorted_tasks:
        crane_idx = data['crane_idx']
        crane_task_queue[crane_idx].append((task_id, data))
    # 对每个场桥的任务队列按时间排序
    for crane_idx in crane_task_queue:
        crane_task_queue[crane_idx].sort(key=lambda x: x[1]['planned_start_time'])
    # 跟踪场桥状态的详细信息
    crane_task_states = {i: [] for i in range(num_cranes)}
    updated_task_timings = task_timings.copy()
    block_timeline = {}
    updated_timeline = {}
    # 全局冲突计数器
    total_conflicts = 0
    # 处理每个任务
    for task_id, data in sorted_tasks:
        crane_idx = data['crane_idx']
        target_position = data['position']
        target_block = data['block']
        planned_start_time = data['planned_start_time']
        move_time = data['move_time']
        operation_time = data['operation_time']
        prep_time = data['prep_time']
        # 获取当前场桥状态
        current_position = crane_positions[crane_idx]
        current_block = crane_blocks[crane_idx] if crane_blocks[crane_idx] is not None else target_block
        current_time = crane_available_times[crane_idx] # ????
        # 初始化冲突分析
        conflicts = []
        wait_time = 0
        conflict_type = None
        # 计算初始时间
        task_start_time = max(planned_start_time, current_time, prep_time)
        task_move_end_time = task_start_time + move_time
        task_end_time = task_move_end_time + operation_time
        # 增加当前时间到任务信息中，用于检查紧急状态
        data['current_time'] = task_start_time
        # 检查并更新任务紧急状态
        is_urgent = check_task_urgency(data, data['current_time'])

        # 获取所有在同一箱区的其他场桥任务
        competing_tasks = []
        for other_crane in range(num_cranes):
            if other_crane == crane_idx:
                continue
            # 检查其他场桥在目标箱区的任务
            for other_task_id, other_data in task_timings.items():
                if (other_data['crane_idx'] == other_crane and
                        other_data['block'] == target_block):
                    # 计算其他任务的时间
                    other_start = other_data.get('actual_start_time', other_data['planned_start_time'])
                    other_move_time = other_data['move_time']
                    other_operation_time = other_data['operation_time']

                    if other_start is None:
                        other_start = other_data['planned_start_time']

                    other_arrival = other_start + other_move_time
                    other_end = other_arrival + other_operation_time

                    # 添加移动和执行阶段
                    competing_tasks.append({
                        'crane_idx': other_crane,
                        'task_id': other_task_id,
                        'start_time': other_start,
                        'arrival_time': other_arrival,
                        'end_time': other_end,
                        'position': other_data['position'],
                        'move_phase': (other_start, other_arrival),
                        'exec_phase': (other_arrival, other_end),
                        'prep_time': other_data['prep_time']
                    })

        # 高级冲突检测和解决
        for comp_task in competing_tasks:
            # 检查时间重叠
            if task_end_time <= comp_task['start_time'] or task_start_time >= comp_task['end_time']:
                continue  # 没有时间重叠
            # 分析冲突类型
            conflict_found, conflict_details = analyze_conflict(
                current_position, target_position,
                task_start_time, task_move_end_time, task_end_time,
                crane_positions[comp_task['crane_idx']], comp_task['position'],
                comp_task['start_time'], comp_task['arrival_time'], comp_task['end_time']
            )

            if conflict_found:
                total_conflicts += 1
                # 智能冲突解决策略
                resolution = intelligent_conflict_resolution(
                    task_id, data, comp_task,
                    crane_task_queue[crane_idx], crane_task_queue[comp_task['crane_idx']]
                )
                # 看状态
                if resolution['action'] == 'wait':
                    wait_time = max(wait_time, resolution['wait_time'])
                    conflict_type = resolution['reason']
                elif resolution['action'] == 'reschedule':
                    # 重新调度冲突任务（这里可以进一步优化）
                    wait_time = max(wait_time, resolution['wait_time'])

                conflicts.append({
                    'other_crane': comp_task['crane_idx'],
                    'type': conflict_details['type'],
                    'severity': conflict_details['severity'],
                    'resolution': resolution
                })

        # 应用解决方案
        actual_start_time = task_start_time
        actual_arrival_time = actual_start_time + wait_time+ move_time
        actual_end_time = actual_arrival_time + operation_time
        # 更新任务时间
        updated_task_timings[task_id].update({
            'actual_start_time': actual_start_time,
            'actual_arrival_time': actual_arrival_time,
            'actual_end_time': actual_end_time,
            'conflict_wait_time': wait_time,
            'conflict_type': conflict_type,
            'total_conflicts': len(conflicts),
            'is_urgent': is_urgent  # 记录任务是否紧
        })
        # 记录场桥状态变化
        crane_task_states[crane_idx].extend([
            (actual_start_time, actual_arrival_time, STATE_MOVING, target_block, current_position, target_position),
            (actual_arrival_time, actual_end_time, STATE_EXECUTING, target_block, target_position, target_position)
        ])
        # 更新时间线
        update_timeline(block_timeline, updated_timeline, crane_idx, target_block,
                        actual_start_time, actual_arrival_time, actual_end_time,
                        current_position, target_position)
        # 更新场桥状态
        crane_positions[crane_idx] = target_position
        crane_blocks[crane_idx] = target_block
        crane_available_times[crane_idx] = actual_end_time

    return updated_task_timings, updated_timeline, block_timeline, total_conflicts

# 计算场桥利用率
def calculate_crane_utilization(task_timings, makespan):
    crane_work_time = {i: 0 for i in range(num_cranes)}
    for task_id, data in task_timings.items():
        crane_idx = data['crane_idx']
        work_duration = data['actual_end_time'] - data['actual_start_time']
        crane_work_time[crane_idx] += work_duration
    utilization = []
    for crane_idx in range(num_cranes):
        util = crane_work_time[crane_idx] / max(makespan, 1)
        utilization.append(util)
    return utilization

# 计算时间段负载均衡
def calculate_time_slot_balance(schedule):
    slot_loads = []
    for slot_idx in range(len(time_slots) - 1):
        total_tasks = 0
        if slot_idx in schedule:
            for crane_idx in schedule[slot_idx]:
                total_tasks += len(schedule[slot_idx][crane_idx])
        slot_loads.append(total_tasks)
    if not slot_loads:
        return 0
    avg_load = sum(slot_loads) / len(slot_loads)
    balance_penalty = sum(abs(load - avg_load) for load in slot_loads) * 50
    return balance_penalty

# 计算箱区切换惩罚
def calculate_block_switch_penalty(schedule):
    penalty = 0
    for slot in schedule.values():
        for crane_idx, tasks in slot.items():
            blocks = [task[2] for task in tasks]
            unique_blocks = len(set(blocks))
            if unique_blocks > 1:
                penalty += (unique_blocks - 1) * 300
    return penalty

# 计算空闲时间惩罚
def calculate_idle_time_penalty_optimized(task_timings, makespan):
    crane_tasks = {i: [] for i in range(num_cranes)}
    # 收集每个场桥的任务
    for task_id, data in task_timings.items():
        crane_idx = data['crane_idx']
        crane_tasks[crane_idx].append((data['actual_start_time'], data['actual_end_time']))
    total_idle_penalty = 0

    for crane_idx in range(num_cranes):
        if not crane_tasks[crane_idx]:
            # 完全空闲的场桥
            total_idle_penalty += makespan * 5
            continue
        # 按开始时间排序
        sorted_tasks = sorted(crane_tasks[crane_idx])
        # 计算间隙
        total_idle_time = 0
        last_end = 0
        for start, end in sorted_tasks:
            if start > last_end:
                gap = start - last_end
                total_idle_time += gap
            last_end = max(last_end, end)
        # 最后一个任务到makespan的空闲时间
        if last_end < makespan:
            total_idle_time += (makespan - last_end) * 0.5  # 降低结尾空闲的权重
        total_idle_penalty += total_idle_time * 8
    return total_idle_penalty

# 优化的适应度计算函数
def calculate_fitness_optimized(chromosome):
    schedule = decode_chromosome(chromosome)
    # 初始化
    crane_positions = {i: 1 for i in range(num_cranes)}
    crane_blocks = {i: None for i in range(num_cranes)}
    crane_available_times = {i: 0 for i in range(num_cranes)}
    task_timings = {}
    timeline = {}
    tasks_by_block_time = {}
    task_assigned = set()

    # 第一步：初步时间计算
    for slot_idx in sorted(schedule.keys()):
        slot = schedule[slot_idx]
        slot_start_time = time_slots[slot_idx]

        for crane_idx in slot:
            tasks = slot[crane_idx]
            # 按准备时间和任务优先级排序
            sorted_tasks = sorted(tasks, key=lambda x: (x[3], task_priority.get(x[5], 4)))

            prev_position = crane_positions[crane_idx]
            prev_block = crane_blocks[crane_idx]
            prev_completion_time = crane_available_times[crane_idx]

            for task in sorted_tasks:
                task_id, position, block, prep_time, container_qty, task_type = task[:6]

                if task_id in task_assigned:
                    return float('inf')
                task_assigned.add(task_id)

                # 改进的移动时间计算
                move_time = calculate_move_time(
                    prev_position, position,
                    prev_block if prev_block is not None else block,
                    block)
                start_time = max(prep_time, prev_completion_time, slot_start_time)
                operation_time = container_qty * 3
                task_timings[task_id] = {
                    'task': task,
                    'crane_idx': crane_idx,
                    'position': position,
                    'block': block,
                    'prev_position': prev_position,
                    'prev_block': prev_block if prev_block is not None else block,
                    'prep_time': prep_time,
                    'move_time': move_time,
                    'operation_time': operation_time,
                    'planned_start_time': start_time,
                    'planned_end_time': start_time + move_time + operation_time,
                    'actual_start_time': None,
                    'actual_arrival_time': None,
                    'actual_end_time': None,
                    'conflict_wait_time': 0
                }
                prev_position = position
                prev_block = block
                prev_completion_time = start_time + move_time  + operation_time
            crane_positions[crane_idx] = prev_position
            crane_blocks[crane_idx] = prev_block
            crane_available_times[crane_idx] = prev_completion_time
    # 第二步：优化的冲突检测
    task_timings, timeline, block_timeline, total_conflicts = detect_and_resolve_crane_conflicts_optimized(timeline,
                                                                                                       task_timings)
    # 检查硬约束
    # 场桥跨越
    crossing_violations, _ = check_crane_crossing(block_timeline)
    if crossing_violations > 0:
        return float('inf')
    # 检查准备时间约束
    for task_id, data in task_timings.items():
        if data['actual_start_time'] < data['prep_time']:
            return float('inf')
    # 检查任务完整性
    missing_tasks = set(task[0] for task in task_data) - task_assigned
    if missing_tasks:
        return float('inf')
    # 目标1：计算目标函数组件
    makespan = max([data['actual_end_time'] for data in task_timings.values()]) if task_timings else 0
    # 目标2：负载均衡计算
    crane_utilization = calculate_crane_utilization(task_timings, makespan)
    utilization_variance = np.var(crane_utilization)
    # 时间段負載均衡
    time_slot_balance = calculate_time_slot_balance(schedule)
    # 箱区切换惩罚
    block_switch_penalty = calculate_block_switch_penalty(schedule)
    #
    # 任务延迟
    task_info = [(task_id, data['actual_start_time'], data['actual_end_time'], data['task'][4]) for task_id, data in
                 task_timings.items()]
    task_queue = [data['task'] for data in task_timings.values()]
    # 计算任务延迟惩罚
    priority_delay_penalty = 0
    for task_id, data in task_timings.items():
        task = data['task']
        task_type = data['task_type']
        priority = task_priority.get(task_type, 4)
        if priority <= 1:  # 只考虑紧急任务和临时重点箱
            # 确保传入字典格式
            task_dict = {
                'task': task,
                'task_type': task_type,
                'move_time': data['move_time'],
                'operation_time': data['operation_time'],
                'planned_start_time': data['planned_start_time'],
                'actual_end_time': data['actual_end_time']
            }

            # 传递空队列，因为此处只计算单个任务延迟
            task_delay_cost = calculate_delay_cost(task_dict, [])
            priority_delay_penalty += task_delay_cost
    # 冲突惩罚
    conflict_penalty = total_conflicts * 100
    # 空闲时间惩罚（优化版）
    idle_time_penalty = calculate_idle_time_penalty_optimized(task_timings, makespan)
    # 组合目标函数
    total_operation_time = sum(task[4] * 3 for task in task_data)
    total_move_time = len(task_data) * 20
    max_possible_makespan = total_operation_time + total_move_time
    normalized_makespan = makespan / max_possible_makespan if max_possible_makespan > 0 else 0
    normalized_variance = utilization_variance /max(crane_utilization)  # 假设最大方差为0.25
    combined_objective = 0.5 * normalized_makespan + 0.5 * normalized_variance
    objective_value = combined_objective * 1000
    # 调整权重以获得更好的平衡
    total_fitness = (
            0.7 * objective_value +
            0.05 * time_slot_balance +
            0.05 * block_switch_penalty +
            0.07 * priority_delay_penalty +
            0.1 * conflict_penalty +
            0.03 * idle_time_penalty
    )
    return total_fitness

# 基本适应度计算函数
def calculate_fitness(chromosome):
    return calculate_fitness_optimized(chromosome)

# 调试函数 - 显示适应度计算详情
def debug_fitness_calculation(chromosome):
    """打印适应度计算的详细过程，包括规范化目标函数"""
    schedule = decode_chromosome(chromosome)
    # 初始化数据结构
    crane_positions = {i: 1 for i in range(num_cranes)}
    crane_blocks = {i: None for i in range(num_cranes)}
    crane_available_times = {i: 0 for i in range(num_cranes)}
    task_timings = {}
    timeline = {}
    tasks_by_block_time = {}
    task_assigned = set()
    print("\n开始计算适应度...")
    # 执行与标准适应度函数相同的计算过程
    # 第一步：初步时间计算
    for slot_idx in sorted(schedule.keys()):
        slot = schedule[slot_idx]
        slot_start_time = time_slots[slot_idx]

        for crane_idx in slot:
            tasks = slot[crane_idx]
            sorted_tasks = sorted(tasks, key=lambda x: (x[3], task_priority.get(x[5], 2)))
            prev_position = crane_positions[crane_idx]
            prev_block = crane_blocks[crane_idx]
            prev_completion_time = crane_available_times[crane_idx]

            for task in sorted_tasks:
                task_id, position, block, prep_time, container_qty, task_type = task[:6]
                if task_id in task_assigned:
                    print(f"发现重复分配的任务: {task_id}")
                    return float('inf')
                task_assigned.add(task_id)

                move_time = calculate_move_time(
                    prev_position, position,
                    prev_block if prev_block is not None else block,
                    block,
                )

                start_time = max(prep_time, prev_completion_time, slot_start_time)
                operation_time = container_qty * 3
                task_timings[task_id] = {
                    'task': task,
                    'crane_idx': crane_idx,
                    'position': position,
                    'block': block,
                    'prev_position': prev_position,
                    'prev_block': prev_block if prev_block is not None else block,
                    'prep_time': prep_time,
                    'move_time': move_time,
                    'operation_time': operation_time,
                    'planned_start_time': start_time,
                    'planned_end_time': start_time + move_time + operation_time,
                    'actual_start_time': None,
                    'actual_arrival_time': None,
                    'actual_end_time': None,
                    'conflict_wait_time': 0
                }

                prev_position = position
                prev_block = block
                prev_completion_time = start_time + move_time + operation_time

            crane_positions[crane_idx] = prev_position
            crane_blocks[crane_idx] = prev_block
            crane_available_times[crane_idx] = prev_completion_time

    print("第一步：初步计算任务时间完成")
    # 第二步：优化的冲突检测
    print("第二步：检测和解决场桥冲突")
    task_timings, timeline, block_timeline, total_conflicts = detect_and_resolve_crane_conflicts_optimized(timeline,
                                                                                                           task_timings)
    # 第三步：检查硬约束
    print("第三步：检查硬约束")
    crossing_violations, _ = check_crane_crossing(block_timeline)
    if crossing_violations > 0:
        print(f"严重违反硬约束: 场桥跨越，违反次数: {crossing_violations}")
        return float('inf')

    # 检查准备时间约束
    prep_time_violations = []
    for task_id, data in task_timings.items():
        if data['actual_start_time'] < data['prep_time']:
            prep_time_violations.append((task_id, data['actual_start_time'], data['prep_time']))

    if prep_time_violations:
        print("严重违反硬约束: 任务准备时间")
        for task_id, start_time, prep_time in prep_time_violations:
            print(f"  任务 {task_id}: 开始时间 {start_time} 早于准备时间 {prep_time}")
        return float('inf')

    # 检查任务完整性
    missing_tasks = set(task[0] for task in task_data) - task_assigned
    if missing_tasks:
        print(f"严重违反硬约束: 未分配任务: {missing_tasks}")
        return float('inf')

    # 计算各项指标
    makespan = max([data['actual_end_time'] for data in task_timings.values()]) if task_timings else 0
    crane_utilization = calculate_crane_utilization(task_timings, makespan)
    utilization_variance = np.var(crane_utilization)
    time_slot_balance = calculate_time_slot_balance(schedule)
    block_switch_penalty = calculate_block_switch_penalty(schedule)
    priority_delay_penalty = 0
    for task_id, data in task_timings.items():
        task = data['task']
        task_type = data['task_type']
        priority = task_priority.get(task_type, 4)
        if priority <= 1:  # 只考虑紧急任务和临时重点箱
            # 确保传入字典格式
            task_dict = {
                'task': task,
                'task_type': task_type,
                'move_time': data['move_time'],
                'operation_time': data['operation_time'],
                'planned_start_time': data['planned_start_time'],
                'actual_end_time': data['actual_end_time']
            }

            # 传递空队列，因为此处只计算单个任务延迟
            task_delay_cost = calculate_delay_cost(task_dict, [])
            priority_delay_penalty += task_delay_cost
    conflict_penalty = total_conflicts * 100
    idle_time_penalty = calculate_idle_time_penalty_optimized(task_timings, makespan)

    # 规范化计算
    total_operation_time = sum(task[4] * 3 for task in task_data)
    total_move_time = len(task_data) * 20
    max_possible_makespan = total_operation_time + total_move_time
    normalized_makespan = makespan / max_possible_makespan if max_possible_makespan > 0 else 0
    normalized_variance = utilization_variance / max(crane_utilization)  # 假设最大方差为0.25
    combined_objective = 0.5 * normalized_makespan + 0.5 * normalized_variance
    objective_value = combined_objective * 1000
    # 调整权重以获得更好的平衡
    total_fitness = (
            0.7 * objective_value +
            0.05 * time_slot_balance +
            0.05 * block_switch_penalty +
            0.07 * priority_delay_penalty +
            0.1 * conflict_penalty +
            0.03 * idle_time_penalty
    )
    # 打印详细情况
    print("\n适应度计算明细:")
    print(f"Makespan: {makespan:.2f} (原始值)")
    print(f"规范化Makespan: {normalized_makespan:.4f}")
    print(f"场桥利用率: {[f'{u:.3f}' for u in crane_utilization]}")
    print(f"利用率方差: {utilization_variance:.4f}")
    print(f"规范化利用率方差: {normalized_variance:.4f}")
    print(f"时间段负载均衡惩罚: {time_slot_balance:.2f}")
    print(f"箱区切换惩罚: {block_switch_penalty:.2f}")
    print(f"高优先级任务延迟惩罚: {priority_delay_penalty:.2f}")
    print(f"冲突惩罚: {conflict_penalty:.2f} (冲突数: {total_conflicts})")
    print(f"空闲时间惩罚: {idle_time_penalty:.2f}")
    print(f"总适应度: {objective_value:.2f}")
    print(f"场桥交叉: 无违反 (硬约束已满足)")
    print(f"准备时间: 无违反 (硬约束已满足)")
    return total_fitness

# 改进的交叉操作
def crossover(parent1, parent2):
    """改进的两点交叉操作"""
    if random.random() > CROSSOVER_RATE:
        return parent1[:], parent2[:]

    size = len(parent1)
    # 选择两个交叉点
    point1, point2 = sorted(random.sample(range(size), 2))
    # 通过交换中间部分创建子代
    child1 = parent1.copy()
    child2 = parent2.copy()
    # 交换交叉点之间的片段
    child1[point1:point2] = parent2[point1:point2]
    child2[point1:point2] = parent1[point1:point2]
    return child1, child2

# 改进的变异操作
def mutate(chromosome):
    """增强的变异操作，提供多种可能的变化"""
    mutated = chromosome.copy()
    # 根据概率应用变异
    if random.random() < MUTATION_RATE:
        # 选择变异类型
        mutation_type = random.choice(["swap", "change_time", "change_crane"])
        if mutation_type == "swap":
            # 交换两个位置
            i, j = random.sample(range(len(mutated)), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        elif mutation_type == "change_time":
            # 为随机任务改变时间段
            idx = random.randint(0, len(mutated) - 1)
            current_time, crane = mutated[idx]
            new_time = random.randint(0, len(time_slots) - 2)
            while new_time == current_time and len(time_slots) - 2 > 1:
                new_time = random.randint(0, len(time_slots) - 2)
            mutated[idx] = (new_time, crane)
        else:  # change_crane
            # 为随机任务改变场桥
            idx = random.randint(0, len(mutated) - 1)
            time_slot, current_crane = mutated[idx]
            new_crane = random.randint(0, num_cranes - 1)
            while new_crane == current_crane and num_cranes > 1:
                new_crane = random.randint(0, num_cranes - 1)
            mutated[idx] = (time_slot, new_crane)
    return mutated

# 自适应变异率函数
def adaptive_mutation_rate(generation, max_generations, stagnation_counter):
    """根据代数和停滞状态调整变异率"""
    base_rate = MUTATION_RATE
    max_rate = 0.8
    # 基于停滞计数器增加变异率
    stagnation_factor = min(stagnation_counter / 20, 1.0)  # 最大提升到停滞20代时
    # 基于当前代数的自适应调整
    generation_factor = generation / max_generations
    # 综合调整
    adjusted_rate = base_rate + (max_rate - base_rate) * (0.7 * stagnation_factor + 0.3 * generation_factor)
    return min(adjusted_rate, max_rate)

# 改进的遗传算法主流程
def genetic_algorithm():
    # 使用优化的初始化函数
    population =initialize_population_improved()
    best_fitness = float('inf')
    best_chromosome = None
    fitness_history = []
    stagnation_counter = 0
    last_improvement_gen = 0
    best_fitness_debug = None  # 存储最佳适应度的详细计算
    global MUTATION_RATE
    print(f"开始运行遗传算法...")
    print(f"种群大小: {POPULATION_SIZE}, 迭代次数: {NUM_GENERATIONS}")
    print(f"交叉率: {CROSSOVER_RATE}, 变异率: {MUTATION_RATE}")

    for generation in range(NUM_GENERATIONS):
        # 基于停滞情况调整变异率
        current_mutation_rate = adaptive_mutation_rate(generation, NUM_GENERATIONS, stagnation_counter)
        # 评估种群
        fitnesses = []
        for ch in population:
            fitness = calculate_fitness_optimized(ch)
            fitnesses.append(fitness)
        # 找到当前最优解
        current_min_idx = np.argmin(fitnesses)
        current_best = fitnesses[current_min_idx]
        fitness_history.append(current_best)

        # 检查是否有改进
        if current_best < best_fitness:
            improvement = best_fitness - current_best
            best_fitness = current_best
            best_chromosome = population[current_min_idx].copy()
            last_improvement_gen = generation
            stagnation_counter = 0
            # 打印改进详情
            print(f"第{generation}代, 新的最佳适应度: {best_fitness:.2f}, 改进: {improvement:.2f}")
            # 记录详细的适应度计算
            if generation % 50 == 0 or improvement > 100:  # 每50代或有显著改进时详细记录
                print("详细适应度分析:")
                best_fitness_debug = debug_fitness_calculation(best_chromosome)
        else:
            stagnation_counter += 1
            if generation % 50 == 0:
                print(f"第{generation}代, 最佳适应度: {best_fitness:.2f}, 停滞代数: {stagnation_counter}")

        # 应用锦标赛选择
        new_population = []
        elite_size = max(1, int(POPULATION_SIZE * 0.1))  # 保留10%的精英
        # 添加精英
        elite_indices = np.argsort(fitnesses)[:elite_size]
        for idx in elite_indices:
            new_population.append(population[idx].copy())

        # 锦标赛选择生成剩余人口
        while len(new_population) < POPULATION_SIZE:
            # 锦标赛选择
            tournament_size = 3
            tournament1 = random.sample(range(len(population)), tournament_size)
            tournament_fitness1 = [fitnesses[i] for i in tournament1]
            parent1_idx = tournament1[np.argmin(tournament_fitness1)]

            tournament2 = random.sample(range(len(population)), tournament_size)
            tournament_fitness2 = [fitnesses[i] for i in tournament2]
            parent2_idx = tournament2[np.argmin(tournament_fitness2)]

            parent1, parent2 = population[parent1_idx].copy(), population[parent2_idx].copy()
            # 创建新个体
            child1, child2 = crossover(parent1, parent2)
            # 应用当前变异率
            old_rate = MUTATION_RATE
            MUTATION_RATE = current_mutation_rate
            child1 = mutate(child1)
            child2 = mutate(child2)
            MUTATION_RATE = old_rate  # 恢复原始变异率
            new_population.extend([child1, child2])

        # 确保种群大小正确
        population = new_population[:POPULATION_SIZE]

        # 如果停滞太久，注入多样性
        if stagnation_counter > 30:
            inject_count = int(POPULATION_SIZE * 0.3)  # 替换30%为随机个体
            rand_population = initialize_population_improved()
            for i in range(inject_count):
                population[-(i + 1)] = rand_population[i]
            print(f"第{generation}代, 因停滞注入多样性")
            stagnation_counter = 0

    # 最后记录最优解的详细信息
    if best_chromosome:
        print("\n最终最优解详细分析:")
        debug_fitness_calculation(best_chromosome)

    # 绘制适应度历史
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(fitness_history)), fitness_history, linewidth=2)
    plt.title('适应度历史', fontproperties=prop, fontsize=16)
    plt.xlabel('代数', fontproperties=prop, fontsize=14)
    plt.ylabel('适应度 (越小越好)', fontproperties=prop, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数刻度更好地显示变化
    plt.tight_layout()
    plt.show()

    return decode_chromosome(best_chromosome)

# 绘制甘特图
def plot_schedule(schedule, output_excel_path='task_timings.xlsx'):
    # 创建图表
    fig, ax = plt.subplots(figsize=(16, 10))
    # 任务类型颜色定义
    type_colors = {
        "临时重点箱": "#FFD700",  # 金黄色
        "装船": "#FF4444",  # 红色
        "卸船": "#4488FF",  # 蓝色
        "进箱": "#44FF44",  # 绿色
        "提箱": "#FF8844"  # 橙色
    }
    # 定义不同等待类型的颜色
    prep_wait_color = "#FFF59D"  # 浅黄色 - 准备时间等待
    conflict_wait_color = "orange"  # 橙黄色 - 冲突等待
    # 创建箱区颜色映射
    block_cmap = LinearSegmentedColormap.from_list('BlockColors',
                                                   ['#8DD3C7', '#FFFFB3', '#BEBADA',
                                                    '#FB8072', '#80B1D3', '#FDB462',
                                                    '#B3DE69', '#FCCDE5', '#D9D9D9',
                                                    '#BC80BD'], N=10)

    # 获取所有出现的箱区编号
    all_blocks = set()
    for slot in schedule.values():
        for tasks in slot.values():
            for task in tasks:
                all_blocks.add(task[2])

    # 为所有出现的箱区分配颜色
    block_colors = {block: block_cmap(i % 10) for i, block in enumerate(sorted(all_blocks))}
    # 初始化任务数据结构和时间线数据结构
    task_timings = {}
    timeline = {}
    crane_positions = {i: 1 for i in range(num_cranes)}
    crane_blocks = {i: None for i in range(num_cranes)}
    crane_available_times = {i: 0 for i in range(num_cranes)}

    # 第一步：初步计算任务时间，不考虑冲突
    for slot_idx in sorted(schedule.keys()):
        slot = schedule[slot_idx]
        slot_start_time = time_slots[slot_idx]

        for crane_idx in slot:
            tasks = slot[crane_idx]
            sorted_tasks = sorted(tasks, key=lambda x: x[3])  # 按准备时间排序

            prev_position = crane_positions[crane_idx]
            prev_block = crane_blocks[crane_idx]
            prev_completion_time = crane_available_times[crane_idx]

            for task in sorted_tasks:
                task_id, position, block, prep_time, container_qty, task_type = task[:6]

                # 计算移动时间
                move_time = calculate_move_time(
                    prev_position, position,
                    prev_block if prev_block is not None else block,
                    block
                )

                # 确定初步开始时间
                start_time = max(prep_time, prev_completion_time, slot_start_time)
                # 计算操作时间
                operation_time = container_qty * 3
                # 记录初步时间安排
                task_timings[task_id] = {
                    'task': task,
                    'crane_idx': crane_idx,
                    'position': position,
                    'block': block,
                    'prev_position': prev_position,
                    'prev_block': prev_block if prev_block is not None else block,
                    'prep_time': prep_time,
                    'move_time': move_time,
                    'operation_time': operation_time,
                    'planned_start_time': start_time,
                    'planned_end_time': start_time + move_time + operation_time,
                    'actual_start_time': None,
                    'actual_arrival_time': None,
                    'actual_end_time': None,
                    'conflict_wait_time': 0,
                    'original_task_type': task_type  # 保存原始任务类型
                }
                # 更新前一个任务状态
                prev_position = position
                prev_block = block
                prev_completion_time = start_time + move_time + operation_time

            # 更新场桥位置和时间
            crane_positions[crane_idx] = prev_position
            crane_blocks[crane_idx] = prev_block
            crane_available_times[crane_idx] = prev_completion_time

    # 第二步：检测和解决场桥冲突
    task_timings, timeline, block_timeline, _ = detect_and_resolve_crane_conflicts_optimized(timeline, task_timings)
    # 用于记录任务数据结构
    task_data_visualization = []
    # 收集可视化所需的数据
    for task_id, data in task_timings.items():
        crane_idx = data['crane_idx']
        block = data['block']
        # 获取当前任务类型和原始任务类型
        current_task_type = data['task'][5] if len(data['task']) > 5 else "未知"
        original_task_type = data.get('original_task_type', current_task_type)

        # 记录用于可视化的任务数据
        task_data_visualization.append({
            'task_id': task_id,
            'crane_idx': crane_idx,
            'block': block,
            'task_type': current_task_type,
            'original_task_type': original_task_type,
            'container_qty': data['task'][4],
            'prep_time': data['prep_time'],
            'actual_start_time': data['actual_start_time'],
            'actual_arrival_time': data['actual_arrival_time'],
            'actual_end_time': data['actual_end_time'],
            'prep_wait_time': max(0, data['prep_time'] - data['planned_start_time']) if data['prep_time'] > data[
                'planned_start_time'] else 0,
            'conflict_wait_time': data['conflict_wait_time'],
            'move_time': data['move_time'],
            'operation_time': data['operation_time']
        })
    # 绘制背景网格，增强可读性
    ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.3)
    # 为每个时间段添加背景色，以便区分
    for i in range(len(time_slots) - 1):
        ax.axvspan(time_slots[i], time_slots[i + 1],
                   alpha=0.1,
                   color='gray' if i % 2 == 0 else 'lightgray')
    # 绘制垂直线表示时间段边界
    for t in time_slots:
        ax.axvline(x=t, color='gray', linestyle='-', alpha=0.5, linewidth=1)
        ax.text(t, -0.5, f"{t}", ha='center', va='top',
                fontproperties=prop, fontsize=9)
    # 绘制任务块和等待时间
    for task_data in task_data_visualization:
        crane_idx = task_data['crane_idx']
        block = task_data['block']
        task_id = task_data['task_id']
        task_type = task_data['task_type']
        original_task_type = task_data['original_task_type']

        # 获取显示颜色：如果是紧急任务，使用原始任务类型的颜色
        display_task_type = original_task_type if task_type == "紧急任务" else task_type
        display_color = type_colors.get(display_task_type, type_colors["进箱"])

        # 绘制准备时间等待
        if task_data['prep_wait_time'] > 0:
            start_time = task_data['actual_start_time'] - task_data['prep_wait_time'] - task_data['conflict_wait_time']
            ax.barh(
                crane_idx,
                width=task_data['prep_wait_time'],
                left=start_time,
                height=0.6,
                color=prep_wait_color,
                edgecolor='black',
                alpha=0.7,
                hatch='/'
            )
        # 绘制冲突等待
        if task_data['conflict_wait_time'] > 0:
            start_time = task_data['actual_start_time'] - task_data['conflict_wait_time']
            ax.barh(
                crane_idx,
                width=task_data['conflict_wait_time'],
                left=start_time,
                height=0.6,
                color=conflict_wait_color,
                edgecolor='black',
                alpha=0.8,
                hatch='x'
            )
        # 绘制任务移动时间
        move_start = task_data['actual_start_time']
        move_end = task_data['actual_arrival_time']
        rect = mpatches.Rectangle(
            xy=(move_start, crane_idx - 0.3),  # (x, y) of bottom left - adjusted for height
            width=move_end - move_start,
            height=0.6,
            color=display_color,
            alpha=0.6,
            linewidth=0
        )
        # 添加矩形到轴
        ax.add_patch(rect)
        # 添加自定义边框（只有上、下、左）
        ax.plot([move_start, move_start], [crane_idx - 0.3, crane_idx + 0.3], color='black', linewidth=1)
        ax.plot([move_start, move_end], [crane_idx + 0.3, crane_idx + 0.3], color='black', linewidth=1)
        ax.plot([move_start, move_end], [crane_idx - 0.3, crane_idx - 0.3], color='black', linewidth=1)

        # 绘制操作时间
        operation_start = task_data['actual_arrival_time']
        operation_end = task_data['actual_end_time']
        rect_op = mpatches.Rectangle(
            xy=(operation_start, crane_idx - 0.3),
            width=operation_end - operation_start,
            height=0.6,
            color=display_color,
            alpha=1.0,  # 较深的透明度
            linewidth=0
        )
        ax.add_patch(rect_op)
        ax.plot([operation_start, operation_end], [crane_idx + 0.3, crane_idx + 0.3], color='black', linewidth=1)
        ax.plot([operation_start, operation_end], [crane_idx - 0.3, crane_idx - 0.3], color='black', linewidth=1)
        ax.plot([operation_end, operation_end], [crane_idx - 0.3, crane_idx + 0.3], color='black', linewidth=1)

        # 添加箱区标识并显示箱区数字
        box_width = task_data['actual_end_time'] - task_data['actual_start_time']
        ax.add_patch(plt.Rectangle(
            (task_data['actual_start_time'], crane_idx + 0.3),
            box_width,
            0.1,
            facecolor=block_colors[block],
            alpha=1.0,
            linewidth=1,
            edgecolor='black'
        ))

        # 在箱区色条上添加数字标记
        ax.text(
            task_data['actual_start_time'] + box_width / 2,
            crane_idx + 0.35,
            f"{block}",
            ha='center',
            va='center',
            fontproperties=prop,
            fontsize=7,
            color='black',
            fontweight='bold'
        )

        # 在任务块上添加任务信息（仅当空间足够时）
        execution_start = task_data['actual_start_time']
        execution_end = task_data['actual_end_time']
        task_duration = execution_end - execution_start
        if task_duration > 15:  # 只在任务持续时间足够长时添加标签
            # 显示任务ID，如果是紧急任务则添加标识
            task_label = f"{task_id}"
            if task_type == "紧急任务":
                task_label += "(!)"  # 紧急任务添加感叹号标识
            ax.text(
                (execution_start + execution_end) / 2,
                crane_idx,
                task_label,
                ha='center',
                va='center',
                fontproperties=prop,
                fontsize=9,
                fontweight='bold'
            )

    # 计算makespan
    makespan = max([task['actual_end_time'] for task in task_data_visualization])
    # 设置坐标轴
    ax.set_yticks(range(num_cranes))
    ax.set_yticklabels([f'场桥 {i + 1}' for i in range(num_cranes)],
                       fontproperties=prop, fontsize=12)
    ax.set_xlabel('时间 (分钟)', fontproperties=prop, fontsize=12)
    ax.set_title('场桥调度甘特图', fontproperties=prop, fontsize=16)
    # 设置X轴范围，确保显示所有内容
    ax.set_xlim(-5, makespan + 30)
    # 设置Y轴范围，为标签预留空间
    ax.set_ylim(-1, num_cranes)
    # 调整绘图区域，为图例腾出更多空间
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # 为图例腾出更多空间
    # 创建图例
    legend_elements = []
    # 任务类型图例 - 不再区分移动和操作时间
    for task_type in type_colors:
        legend_elements.append(
            mpatches.Patch(color=type_colors[task_type], label=f'{task_type}（任务执行）', alpha=0.9)
        )
    # 添加紧急任务说明
    legend_elements.append(
        mpatches.Patch(color='white', label='紧急任务标记为(!)，使用原始类型颜色', alpha=0)
    )
    # 等待时间图例
    legend_elements.extend([
        mpatches.Patch(color=prep_wait_color, label='准备时间等待', alpha=0.8, hatch='/'),
        mpatches.Patch(color=conflict_wait_color, label='冲突等待', alpha=0.8, hatch='x')
    ])
    # 箱区图例
    for block in sorted(all_blocks):
        legend_elements.append(
            mpatches.Patch(facecolor=block_colors[block], label=f'箱区 {block}')
        )
    # 放置图例在图的外部右侧，使用两列以节省空间
    ax.legend(handles=legend_elements, loc='center left',
              bbox_to_anchor=(1, 0.5),
              prop=prop, frameon=True, fontsize=10, ncol=1)
    # 布局调整
    plt.tight_layout()
    plt.subplots_adjust(right=0.80)  # 为图例腾出空间
    plt.savefig('gantt_chart.png', dpi=300, bbox_inches='tight')  # 保存为高分辨率图片
    plt.show()
    # 创建数据框并导出到Excel
    excel_data = []
    for task in task_data_visualization:
        excel_data.append({
            '任务ID': task['task_id'],
            '任务类型': task['task_type'],
            '场桥编号': task['crane_idx'] + 1,
            '箱区编号': task['block'],
            '集装箱数量': task['container_qty'],
            '准备时间': task['prep_time'],
            '准备等待时间': int(task['prep_wait_time']),
            '冲突等待时间': int(task['conflict_wait_time']),
            '实际开始时间': int(task['actual_start_time']),
            '移动时间': int(task['move_time']),
            '到达目标位置时间': int(task['actual_arrival_time']),
            '操作时间': int(task['operation_time']),
            '完成时间': int(task['actual_end_time']),
            '总持续时间': int(task['actual_end_time'] - task['actual_start_time'])
        })
    # 创建DataFrame并导出到Excel
    df = pd.DataFrame(excel_data)
    df.to_excel(output_excel_path, index=False, sheet_name='任务时间详情')
    print(f"任务时间详情已导出到: {output_excel_path}")

    return {task['task_id']: task for task in task_data_visualization}
# 分析调度方案
def analyze_schedule(schedule):
    # 计算基本统计信息
    total_tasks = 0
    crane_task_counts = [0] * num_cranes
    slot_task_counts = [0] * (len(time_slots) - 1)
    block_task_counts = {}

    for slot_idx, slot in schedule.items():
        for crane_idx, tasks in slot.items():
            task_count = len(tasks)
            total_tasks += task_count
            crane_task_counts[crane_idx] += task_count
            slot_task_counts[slot_idx] += task_count
            # 统计箱区使用情况
            for task in tasks:
                block = task[2]
                if block not in block_task_counts:
                    block_task_counts[block] = 0
                block_task_counts[block] += 1

    print(f"总任务数: {total_tasks}")
    print(f"场桥负载分布: {crane_task_counts}")
    print(f"时间段负载分布: {slot_task_counts}")
    print(f"箱区使用情况: {dict(sorted(block_task_counts.items()))}")
    # 计算负载均衡指标
    crane_avg = np.mean(crane_task_counts)
    crane_std = np.std(crane_task_counts)
    crane_cv = crane_std / crane_avg if crane_avg > 0 else 0
    slot_avg = np.mean(slot_task_counts)
    slot_std = np.std(slot_task_counts)
    slot_cv = slot_std / slot_avg if slot_avg > 0 else 0
    print(f"\n负载均衡分析:")
    print(f"场桥负载变异系数: {crane_cv:.3f} (越小越好)")
    print(f"时间段负载变异系数: {slot_cv:.3f} (越小越好)")
    # 分析优先级任务分布
    priority_distribution = {}
    for slot_idx, slot in schedule.items():
        for crane_idx, tasks in slot.items():
            for task in tasks:
                task_type = task[5]
                priority = task_priority.get(task_type, 2)
                if priority not in priority_distribution:
                    priority_distribution[priority] = []
                priority_distribution[priority].append(slot_idx)

    print(f"\n优先级任务分布:")
    for priority in sorted(priority_distribution.keys()):
        slots = priority_distribution[priority]
        avg_slot = np.mean(slots)
        print(f"优先级{priority}任务平均安排在时间段: {avg_slot:.1f}")

# 主程序入口
if __name__ == "__main__":
    # 设置随机种子以获得可重现的结果
    np.random.seed(42)
    random.seed(42)
    best_schedule = genetic_algorithm()
    print("最终调度方案:")
    plot_schedule(best_schedule)