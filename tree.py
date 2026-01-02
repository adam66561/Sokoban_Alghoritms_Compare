from collections import deque, defaultdict
import subprocess
from level import Level, State
import os

def generate_bfs_tree(level: Level, start: State, max_nodes: int = 5000):
    q = deque([start])
    visited = {start}

    parent: dict[State, State | None] = {start: None}
    depth: dict[State, int] = {start: 0}

    children: dict[State, list[State]] = defaultdict(list)
    edge_action: dict[State, str] = {}   # child -> action (NOWE)

    nodes = 1

    while q and nodes < max_nodes:
        s = q.popleft()

        for action in level.legal_moves(s):   # action: "U","D","L","R"
            res = level.step(s, action)
            if res is None:
                continue
            ns, _ = res

            if ns in visited:
                continue

            visited.add(ns)
            parent[ns] = s
            edge_action[ns] = action          # <- zapamiętujemy etykietę
            children[s].append(ns)
            depth[ns] = depth[s] + 1

            q.append(ns)
            nodes += 1
            if nodes >= max_nodes:
                break

    return parent, children, depth, edge_action


def export_tree_to_dot(parent, depth, edge_action, filename="bfs_tree.dot"):
    lines = [
        "digraph BFS {",
        "rankdir=TB;",
        "node [shape=circle, fontsize=10];",
        "edge [fontsize=10];",
    ]

    # węzły
    for node in parent.keys():
        node_id = id(node)
        lines.append(f'"{node_id}" [label="d={depth[node]}"];')

    # krawędzie z etykietą ruchu
    for child, par in parent.items():
        if par is None:
            continue
        a = edge_action.get(child, "?")
        lines.append(f'"{id(par)}" -> "{id(child)}" [label="{a}"];')

    lines.append("}")
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def render_and_open_dot(dot_file="tree.dot", png_file="tree.png"):
    subprocess.run(
        ["dot", "-Tpng", dot_file, "-o", png_file],
        check=True
    )
    os.startfile(png_file)