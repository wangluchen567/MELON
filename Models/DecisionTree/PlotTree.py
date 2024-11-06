import numpy as np
import networkx as nx
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors


def recursion_info(graph, node, parent_id=None, branch_name=None, class_names=None,
                   colors=None, node_colors=None, max_indicator=1):
    # 初始化节点颜色信息
    if node_colors is None:
        node_colors = []
    # 当前节点的名称
    # 需要判断是分类树还是回归树
    if hasattr(node, 'values'):
        node_label = ""  # 分类树
        if node.node_name is not None:
            node_label += f"{node.node_name}\n"
        node_label += (f"{node.ind_type}={node.indicator:.3f}\nsamples={node.samples}\n"
                       f"values={node.values}\nclass={node.category}")
    else:
        node_label = "" # 回归树
        if node.node_name is not None:
            node_label += f"{node.node_name}\n"
        if node.indicator > 1.e4:  # 若数字太大使用科学计数法表示
            node_label += f"{node.ind_type}={node.indicator:.2e}\n"
        else:
            node_label += f"{node.ind_type}={node.indicator:.3f}\n"
        node_label += f"samples={node.samples}\n"
        if node.predict_value > 1.e4:  # 若数字太大使用科学计数法表示
            node_label += f"predict={node.predict_value:.2e}\n"
        else:
            node_label += f"predict={node.predict_value:.3f}\n"
    # 设置当前节点下标为位置信息
    node_id = str(node.pos)
    # 将节点添加到图中
    graph.add_node(node_id, label=node_label, pos=(node.pos[0], -node.pos[1]))
    # 如果父节点存在，添加一条从父节点到当前节点的边
    if parent_id is not None:
        graph.add_edge(parent_id, node_id, label=branch_name)
    # 节点颜色配置
    indicator = (node.indicator / max_indicator) if max_indicator > 1 else node.indicator
    alpha = 0.9 - 0.8 * indicator  # 设置透明度
    # node_color = 'skyblue'  # 节点默认颜色
    node_color = mpl_colors.to_rgba('skyblue', alpha=alpha)
    if class_names is not None and colors is not None:
        # 得到节点的颜色信息
        node_color = mpl_colors.to_rgba(colors[np.where(class_names==node.category)[0]], alpha=alpha)
    node_colors.append(node_color)
    # 递归地得到每个子节点的信息
    for i, (branch_name, child_node) in enumerate(node.branches.items()):
        recursion_info(graph, child_node, node_id, branch_name, class_names, colors, node_colors, max_indicator)
    return node_colors


def plot_tree(root, class_names=None, font_size=6.6, font_family='Microsoft Yahei'):
    # 画图中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = [font_family]
    plt.rcParams['axes.unicode_minus'] = False
    # 创建一个有向图
    graph = nx.DiGraph()
    # 初始化颜色列表
    colors = None
    # 若是分类问题则根据结果绘制颜色
    if class_names is not None:
        # 获取一个colormap对象
        cmap = cm.get_cmap('rainbow')
        # 使用linspace生成一个从0到1的等间隔的数组，长度为num_class
        colors = cmap(np.linspace(0.1, 0.9, len(class_names)))
    # 从根节点开始递归得到树中每个节点的信息
    node_colors = recursion_info(graph, root, class_names=class_names, colors=colors, max_indicator=root.max_indicator)
    # 根据树的深度和宽度设置画布大小
    plt.figure(figsize=(root.max_width*2, (root.max_depth + 1)*2))
    positions = nx.get_node_attributes(graph, 'pos')
    node_labels = nx.get_node_attributes(graph, 'label')
    nx.draw_networkx(graph, positions, arrows=True, with_labels=True, labels=node_labels, node_shape='s',
                     node_color=node_colors, node_size=3000, font_size=font_size)

    # 获取边的标签属性
    edge_labels = nx.get_edge_attributes(graph, 'label')
    # 在图形上添加边标签
    nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels,
                                 font_size=font_size)
    plt.title('Decision Tree')
    plt.tight_layout()
    plt.show()
