import networkx as nx
import matplotlib.pyplot as plt


def recursion_plot(node, parent_name, graph, pos=None):
    if pos is None:
        pos = {}
    # 当前节点的名称
    node_name = "state: {}\nentropy={:.6f}\nnext_attr={}\nclass={}".format(node.state, node.indicator, node.attr_name, node.category)
    # 将节点添加到图中
    graph.add_node(node_name)
    # 从上向下绘制加负号
    pos[node_name] = (node.pos[0], -node.pos[1])
    # 如果父节点存在，添加一条从父节点到当前节点的边
    if parent_name:
        graph.add_edge(parent_name, node_name)
    # 递归地绘制每个子节点
    for i, (branch_name, child_node) in enumerate(node.branches.items()):
        recursion_plot(child_node, node_name, graph, pos)
    return pos


def plot_tree(root):
    # 画图中文显示会有问题，需要这两行设置默认字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 创建一个有向图
    graph = nx.DiGraph()
    # 从根节点开始绘制树
    pos = recursion_plot(root, None, graph)
    plt.figure(figsize=(12, 10))
    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, node_shape='s', node_color='skyblue', node_size=8000)
    # 绘制边
    nx.draw_networkx_edges(graph, pos)
    # 绘制标签
    labels = {node: node for node in graph.nodes()}
    nx.draw_networkx_labels(graph, pos, labels, font_size=10, font_weight='bold')
    plt.title('Decision Tree')
    plt.show()
