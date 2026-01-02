import sys
from PySide6.QtWidgets import ( # type: ignore
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QMessageBox, QGraphicsView, QGraphicsScene
)
from PySide6.QtGui import QKeyEvent, QFont, QPixmap, QColor, QBrush # type: ignore
from PySide6.QtCore import QTimer, Qt, QSize # type: ignore
from level import Level, State 
from solver import bfs_solve # type: ignore
from typing import Callable
import time
from tree import generate_bfs_tree, export_tree_to_dot, render_and_open_dot # type: ignore
import math


def load_single_maze(path: str, maze_number: int) -> list[str] | None:
    with open(path, encoding="utf-8") as f:
        lines = iter(f.readlines())

    current_maze = None
    size_y = None

    for line in lines:
        if line.startswith("Maze:"):
            current_maze = int(line.split(":")[1].strip())
            size_y = None
            continue

        if current_maze == maze_number and line.startswith("Size Y:"):
            size_y = int(line.split(":")[1].strip())
            continue

        if current_maze == maze_number and size_y is not None and line.strip() == "":
            board: list[str] = []
            for _ in range(size_y):
                row = next(lines).rstrip("\n")
                board.append(row)
            return board

    return None


class SokobanWindow(QMainWindow):
    def __init__(self, maps_path: str):
        super().__init__()
        self.setWindowTitle("Sokoban (PySide6)")

        self.first_depth: dict[tuple[int,int], int] = {}   # (x,y) -> minimalny depth
        self.current_depth: int = 0
        self._last_ui_update = 0.0

        self.heat_player: dict[tuple[int,int], int] = {}
        self.last_bfs_state: State | None = None
        self._last_ui_update_t = 0.0

        self.maps_path = maps_path
        self.level: Level | None = None
        self.state: State | None = None
        self.start_state: State | None = None

        self.layer_depth = -1
        self.layer_cells: set[tuple[int, int]] = set()

        self.visualize_mode = False

        self.moves = 0
        self.pushes = 0

        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._anim_step)
        self.solution_moves: list[str] = []
        self.solution_i = 0

        self.game_over = False

        # --- UI layout ---
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        top = QHBoxLayout()
        layout.addLayout(top)

        top.addWidget(QLabel("Mapa #:"))
        self.map_spin = QSpinBox()
        self.map_spin.setRange(1, 10_000)
        self.map_spin.setValue(1)
        top.addWidget(self.map_spin)

        self.btn_load = QPushButton("Wczytaj")
        self.btn_load.clicked.connect(self.load_map)
        top.addWidget(self.btn_load)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_map)
        top.addWidget(self.btn_reset)

        self.btn_vis = QPushButton("Visualize OFF")
        self.btn_vis.clicked.connect(self.toggle_visualize)
        top.addWidget(self.btn_vis)

        self.btn_bfs = QPushButton("Solve BFS")
        self.btn_bfs.clicked.connect(self.solve_bfs)
        top.addWidget(self.btn_bfs)

        self.stats = QLabel("ruchy=0, pchnięcia=0")
        top.addWidget(self.stats)
        top.addStretch(1)

        # Plansza
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        layout.addWidget(self.view, 1)

        hint = QLabel("Sterowanie: WASD lub strzałki. R = reset. Kliknij okno i graj.")
        layout.addWidget(hint)

        self.tile = 32
        self.sprites = self.load_sprites(self.tile)

        QTimer.singleShot(0, self.fit_view)

        # startowo mapa 1
        self.load_map()


    def fit_view(self):
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


    def load_sprites(self, tile: int) -> dict[str, QPixmap]:
        def load(path: str) -> QPixmap:
            pm = QPixmap(path)
            if pm.isNull():
                raise FileNotFoundError(f"Nie mogę wczytać obrazka: {path}")
            return pm.scaled(tile, tile, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

        return {
            "goal": load("assets/goal.png"),
            "bricks": load("assets/bricks.png"),
            "worker": load("assets/worker.png"),
            "box": load("assets/box.png"),
            "floor": load("assets/floor.png"),
        }


    def toggle_visualize(self):
        self.visualize_mode = not self.visualize_mode
        self.btn_vis.setText("Visualize ON" if not self.visualize_mode else "Visualize OFF")
        self.layer_cells.clear()
        self.redraw()


    def update_stats(self, text: str | None = None):
        if text is None:
            text = f"ruchy={self.moves}, pchnięcia={self.pushes}"
        self.stats.setText(text)


    def load_map(self):
        self.game_over = False
        n = self.map_spin.value()
        board = load_single_maze(self.maps_path, n)
        if board is None:
            QMessageBox.warning(self, "Brak mapy", f"Nie znaleziono mapy nr {n}.")
            return

        self.level, self.state = Level.from_lines(board)
        self.start_state = self.state
        self.moves = 0
        self.pushes = 0
        self.update_stats()
        self.redraw()


    def reset_map(self):
        self.game_over = False
        if self.level is None or self.start_state is None:
            return
        self.state = self.start_state
        self.moves = 0
        self.pushes = 0
        self.update_stats()
        self.redraw()


    def keyPressEvent(self, event: QKeyEvent):
        if self.level is None or self.state is None:
            return

        key = event.key()
        
        if key == Qt.Key_R:
            self.reset_map()
            return
        if self.anim_timer.isActive():
            return
        if self.game_over:
            return
        
        action = None
        
        if self.game_over:
            return
        if key in (Qt.Key_W, Qt.Key_Up):
            action = "U"
        elif key in (Qt.Key_S, Qt.Key_Down):
            action = "D"
        elif key in (Qt.Key_A, Qt.Key_Left):
            action = "L"
        elif key in (Qt.Key_D, Qt.Key_Right):
            action = "R"
        else:
            return

        res = self.level.step(self.state, action)
        if res is None:
            return

        self.state, pushed = res
        self.moves += 1
        self.pushes += int(pushed)
        self.update_stats()
        self.redraw()

        if self.level.is_solved(self.state):
            self.game_over = True
            QMessageBox.information(self, "Wygrana!", f"Ukończono! ruchy={self.moves}, pchnięcia={self.pushes}")


    def _on_bfs_progress(self, s: State, depth: int, visited: int, expanded: int, dt: float, layer_end: bool):
        if self.visualize_mode == False:
            return

        if depth != self.layer_depth:
            self.layer_depth = depth
            self.layer_cells.clear()

        self.layer_cells.add(s.player)

        self._last_ui_update = time.perf_counter()
        self.update_stats(
            f"BFS: depth={depth}, visited={visited}, expanded={expanded}, {int(expanded/max(dt,1e-9))}/s"
        )
        self.redraw()
        QApplication.processEvents()
        if layer_end:
            time.sleep(0.5)


    def redraw(self):
        if self.level is None or self.state is None:
            return

        self.scene.clear()

        tile = self.tile
        w, h = self.level.width, self.level.height
        boxes = set(self.state.boxes)
        font = QFont("Consolas", 14)
        WINDOW = 5

        for y in range(h):
            for x in range(w):
                p = (x, y)

                rect = self.scene.addRect(x * tile, y * tile, tile, tile)
                rect.setPen(Qt.NoPen)

                if p in self.level.walls:
                    self.scene.addPixmap(self.sprites["bricks"]).setPos(x * tile, y * tile)
                elif p == self.state.player:
                    self.scene.addPixmap(self.sprites["worker"]).setPos(x * tile, y * tile)
                elif p in boxes:
                    self.scene.addPixmap(self.sprites["box"]).setPos(x * tile, y * tile)
                elif p in self.level.goals:
                    self.scene.addPixmap(self.sprites["goal"]).setPos(x * tile, y * tile)
                else:
                    self.scene.addPixmap(self.sprites["floor"]).setPos(x * tile, y * tile)

                if p in self.layer_cells:
                    color = QColor()
                    hue = (self.layer_depth * 17) % 360
                    color.setHsv(hue, 255, 255, 180)
                    overlay = self.scene.addRect(x * tile, y * tile, tile, tile)
                    overlay.setPen(Qt.NoPen)
                    overlay.setBrush(QBrush(color))

        self.scene.setSceneRect(0, 0, w * tile, h * tile)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


    def start_animation(self, moves: list[str], interval_ms: int = 60):
        if self.level is None or self.state is None:
            return
        if not moves:
            return

        # zatrzymaj poprzednia animację jesli trwa
        if self.anim_timer.isActive():
            self.anim_timer.stop()

        self.solution_moves = moves
        self.solution_i = 0

        # blokuj ręczne ruchy w trakcie animacji
        self.game_over = False
        self.anim_timer.start(interval_ms)


    def _anim_step(self):
        if self.level is None or self.state is None:
            self.anim_timer.stop()
            return

        # koniec animacji
        if self.solution_i >= len(self.solution_moves):
            self.anim_timer.stop()
            return

        a = self.solution_moves[self.solution_i]
        self.solution_i += 1

        res = self.level.step(self.state, a)
        if res is None:
            self.anim_timer.stop()
            return

        self.state, pushed = res
        self.moves += 1
        self.pushes += int(pushed)

        self.update_stats()
        self.redraw()

        if self.level.is_solved(self.state):
            self.anim_timer.stop()
            self.game_over = True
            QMessageBox.information(self, "Wygrana!", f"Ukończono! ruchy={self.moves}, pchnięcia={self.pushes}")


    def solve_bfs(self):
        self.first_depth.clear()
        self.current_depth = 0
        self._last_ui_update = 0.0
        
        if self.level is None or self.state is None:
            return

        self.layer_depth = -1
        self.layer_cells.clear()

        level, start = Level.from_lines(load_single_maze("./maps/sokoban-maps-60-plain.txt", self.map_spin.value()))
        parent, children, depth, edge_action = generate_bfs_tree(level, start, max_nodes=500)
        export_tree_to_dot(parent, depth, edge_action, "tree.dot")
        render_and_open_dot()
        
        res = bfs_solve(self.level, self.state, max_states=100_000_000,     on_progress=self._on_bfs_progress)
        state_to_show = res.last_state if res.last_state is not None else self.state

        #stan końcowy solvera
        self.state = state_to_show
        self.redraw()

        if res.last_moves:
            self.reset_map()  # start
            self.start_animation(res.last_moves)


def main():
    app = QApplication(sys.argv)
    win = SokobanWindow("./maps/sokoban-maps-60-plain.txt")
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
