import sys
from PySide6.QtWidgets import ( # type: ignore
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QMessageBox, QGraphicsView, QGraphicsScene
)
from PySide6.QtGui import QKeyEvent, QFont, QPixmap, QColor, QBrush # type: ignore
from PySide6.QtCore import QTimer, Qt, QSize, QThread, Signal # type: ignore
from level import Level, State 
from solver import bfs_solve, dfs_solve, a_star_solve, gbfs_solve # type: ignore
from typing import Callable
from tree import generate_bfs_tree, export_tree_to_dot, render_and_open_dot # type: ignore
import math
from push_env import PushSokobanEnv # type: ignore
from dqn_train import train_dqn, play_dqn, DQNAgent # type: ignore
import json
from datetime import datetime, time
import time
from pathlib import Path


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

SOLUTIONS_FILE = Path("solutions.json")
ALGO_KEYS = {"BFS", "DFS", "A*", "GBFS"}

def moves_to_str(moves: list[str]) -> str:
    return "".join(moves)

def str_to_moves(s: str) -> list[str]:
    s = s.strip().upper()
    return [ch for ch in s if ch in ("U", "D", "L", "R")]


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

        self.last_solution_algo: str | None = None
        self.last_solution_moves: list[str] = []
        self.last_solve_result = None  


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

        self.btn_dfs = QPushButton("Solve DFS")
        self.btn_dfs.clicked.connect(self.solve_dfs)
        top.addWidget(self.btn_dfs)

        self.btn_astar = QPushButton("Solve A*")
        self.btn_astar.clicked.connect(self.solve_astar)
        top.addWidget(self.btn_astar)

        self.btn_gbfs = QPushButton("Solve GBFS")
        self.btn_gbfs.clicked.connect(self.solve_gbfs)
        top.addWidget(self.btn_gbfs)

        self.btn_train_rl = QPushButton("Train DQN")
        self.btn_train_rl.clicked.connect(self.train_dqn_clicked)
        top.addWidget(self.btn_train_rl)

        self.btn_play_rl = QPushButton("Play DQN")
        self.btn_play_rl.clicked.connect(self.play_dqn_clicked)
        top.addWidget(self.btn_play_rl)

        self.dqn_agent: DQNAgent | None = None
        self._rl_thread: QThread | None = None


        self.stats = QLabel("ruchy=0, pchnięcia=0")
        top.addWidget(self.stats)
        top.addStretch(1)

        mid = QHBoxLayout()
        layout.addLayout(mid)
        self.btn_save_sol = QPushButton("Zapisz rozwiązanie")
        self.btn_save_sol.clicked.connect(self.save_solution_clicked)
        self.btn_save_sol.setEnabled(False) 
        mid.addWidget(self.btn_save_sol)

        self.btn_load_sol_bfs = QPushButton("Wczytaj rozwiązanie BFS")
        self.btn_load_sol_bfs.clicked.connect(lambda: self.load_solution_clicked("BFS"))
        mid.addWidget(self.btn_load_sol_bfs)

        self.btn_load_sol_dfs = QPushButton("Wczytaj rozwiązanie DFS")
        self.btn_load_sol_dfs.clicked.connect(lambda: self.load_solution_clicked("DFS"))
        mid.addWidget(self.btn_load_sol_dfs)

        self.btn_load_sol_astar = QPushButton("Wczytaj rozwiązanie A*")
        self.btn_load_sol_astar.clicked.connect(lambda: self.load_solution_clicked("A*"))
        mid.addWidget(self.btn_load_sol_astar)

        self.btn_load_sol_gbfs = QPushButton("Wczytaj rozwiązanie GBFS")
        self.btn_load_sol_gbfs.clicked.connect(lambda: self.load_solution_clicked("GBFS"))
        mid.addWidget(self.btn_load_sol_gbfs)

        mid.addStretch(1)

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
        
        if isinstance(self._rl_thread, DQNTrainWorker):
            self._rl_thread.visualize_enabled = self.visualize_mode
            self._rl_thread.step_delay_ms = 60 if self.visualize_mode else 0

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


    def _on_progress(self, s: State, depth: int, visited: int, expanded: int, dt: float, layer_end: bool = False):
        if self.visualize_mode == False:
            return

        """if depth != self.layer_depth:
            self.layer_depth = depth
            self.layer_cells.clear()

        self.layer_cells.add(s.player)

        self._last_ui_update = time.perf_counter()
        self.update_stats(
            f"BFS: depth={depth}, visited={visited}, expanded={expanded}, {int(expanded/max(dt,1e-9))}/s"
        )"""
        self.state = s
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
        
        res = bfs_solve(self.level, self.state, max_states=100_000_000,     on_progress=self._on_progress)
        state_to_show = res.last_state if res.last_state is not None else self.state

        #stan końcowy solvera
        self.state = state_to_show
        self.redraw()

        if res.last_moves:
            self.reset_map()  # start
            self.start_animation(res.last_moves)

        if res.solved:
            self._store_last_solution("BFS", res)
        else:
            self.btn_save_sol.setEnabled(False)



    def solve_dfs(self):
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
        
        res = dfs_solve(self.level, self.state, max_depth=500, max_states=100_000_000,     on_progress=self._on_progress)
        state_to_show = res.last_state if res.last_state is not None else self.state

        #stan końcowy solvera
        self.state = state_to_show
        self.redraw()

        if res.last_moves:
            self.reset_map()  # start
            self.start_animation(res.last_moves)

        if res.solved:
            self._store_last_solution("DFS", res)
        else:
            self.btn_save_sol.setEnabled(False)

    
    def solve_astar(self):
        self.layer_depth = -1
        self.layer_cells.clear()
        self._last_ui_update = 0.0

        if self.level is None or self.state is None:
            return

        res = a_star_solve(
            self.level,
            self.state,
            max_states=100_000_000,  
            on_progress=self._on_progress,  
        )

        state_to_show = res.last_state if res.last_state is not None else self.state
        self.state = state_to_show
        self.redraw()

        if res.last_moves:
            self.reset_map()
            self.start_animation(res.last_moves)

        if res.solved:
            self._store_last_solution("A*", res)
        else:
            self.btn_save_sol.setEnabled(False)


    def solve_gbfs(self):
        self.layer_depth = -1
        self.layer_cells.clear()
        self._last_ui_update = 0.0

        if self.level is None or self.state is None:
            return

        res = gbfs_solve(
            self.level,
            self.state,
            max_states=100_000_000,   
            on_progress=self._on_progress,
        )

        self.state = res.last_state if res.last_state is not None else self.state
        self.redraw()

        if res.last_moves:
            self.reset_map()
            self.start_animation(res.last_moves)

        if res.solved:
            self._store_last_solution("GBFS", res)
        else:
            self.btn_save_sol.setEnabled(False)


###########################################################        
##################                       ##################
##################    LOAD AND SAVE      ##################
##################                       ##################
###########################################################   

    def _load_solutions_db(self) -> dict:
        if not SOLUTIONS_FILE.exists():
            return {"version": 1, "items": []}
        try:
            return json.loads(SOLUTIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "items": []}

    def _save_solutions_db(self, db: dict) -> None:
        SOLUTIONS_FILE.write_text(json.dumps(db, ensure_ascii=False, indent=2), encoding="utf-8")

    def _current_map_id(self) -> int:
        return int(self.map_spin.value())

    def _store_last_solution(self, algo: str, res):
        self.last_solution_algo = algo
        self.last_solution_moves = res.moves[:] if res.solved else []
        self.last_solve_result = res
        self.btn_save_sol.setEnabled(res.solved and bool(res.moves))


    def save_solution_clicked(self):
        res = self.last_solve_result
        if res is None or not res.solved:
            QMessageBox.information(self, "Zapis", "Brak rozwiązania do zapisania.")
            return


        map_id = self._current_map_id()
        algo = self.last_solution_algo
        moves_s = moves_to_str(self.last_solution_moves)

        pushes = None
        if self.level is not None and self.start_state is not None:
            from solver import count_pushes  # type: ignore
            pushes = count_pushes(self.level, self.start_state, self.last_solution_moves)

        db = self._load_solutions_db()
        items: list[dict] = db.get("items", [])

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_item = {
            "map_id": map_id,
            "algo": algo,
            "moves": moves_s,
            "len_moves": len(self.last_solution_moves),
            "pushes": int(getattr(res, "pushes", 0)),
            "visited_states": int(getattr(res, "visited_states", 0)),
            "expanded_states": int(getattr(res, "expanded_states", 0)),
            "time_ms": float(getattr(res, "time_ms", 0.0)),
            "saved_at": now,
        }


        replaced = False
        for i, it in enumerate(items):
            if int(it.get("map_id", -1)) == map_id and str(it.get("algo")) == algo:
                items[i] = new_item
                replaced = True
                break
        if not replaced:
            items.append(new_item)

        db["items"] = items
        self._save_solutions_db(db)

        QMessageBox.information(self, "Zapis", f"Zapisano rozwiązanie: mapa {map_id}, {algo} ({len(self.last_solution_moves)} ruchów).")

    def load_solution_clicked(self, algo: str):
        map_id = self._current_map_id()
        db = self._load_solutions_db()
        items: list[dict] = db.get("items", [])

        cand = [
            it for it in items
            if int(it.get("map_id", -1)) == map_id and str(it.get("algo")) == algo
        ]
        if not cand:
            QMessageBox.information(self, "Odczyt", f"Brak zapisanego rozwiązania {algo} dla mapy {map_id}.")
            return

        cand.sort(key=lambda it: int(it.get("len_moves", 10**9)))
        best = cand[0]

        moves = str_to_moves(str(best.get("moves", "")))
        if not moves:
            QMessageBox.warning(self, "Odczyt", "Zapis znaleziony, ale ruchy są puste/niepoprawne.")
            return

        self.reset_map()
        self.update_stats(f"Loaded: mapa={map_id} algo={algo} ruchy={len(moves)}")
        self.start_animation(moves, interval_ms=60)




###########################################################        
##################                       ##################
################## REINFORCMENT LEARNING ##################
##################                       ##################
###########################################################        
        
    def _make_env_for_current_map(self):
        if self.level is None or self.start_state is None:
            return None
        return PushSokobanEnv(self.level, self.start_state, max_pushes=80)
    
    def _on_dqn_step_visualize(self, moves):
        """Wywoływane po każdym kroku DQN, jeśli wizualizacja jest aktywna."""
        self.start_animation(moves, interval_ms=20)

    def _on_dqn_step(self, payload: dict):
        # payload["state"] to State z env po macro-kroku (push) :contentReference[oaicite:6]{index=6}
        s: State = payload["state"]
        self.state = s

        ep = payload.get("ep")
        eps = payload.get("eps")
        rew = payload.get("reward")
        pushes = payload.get("pushes")

        self.update_stats(f"DQN VIS: ep={ep} eps={eps:.2f} pushes={pushes} r={rew:.1f}")
        self.redraw()


    def train_dqn_clicked(self):
        env = self._make_env_for_current_map()
        if env is None:
            return

        self.update_stats("DQN: training...")
        self.btn_train_rl.setEnabled(False)
        self.btn_play_rl.setEnabled(False)

        # callback sprawdza czy wizualizacja jest włączona.
        show_viz_checker = lambda: self.cb_show_viz.isChecked() if hasattr(self, 'cb_show_viz') else False

        worker = DQNTrainWorker(env, show_viz_checker)
        self._rl_thread = worker

        worker.log.connect(self.update_stats)
        worker.step.connect(self._on_dqn_step)
        worker.done.connect(self._on_dqn_trained)
        worker.start()

    def _on_dqn_trained(self, agent: DQNAgent):
        self.dqn_agent = agent
        self.update_stats("DQN: trained")
        self.btn_train_rl.setEnabled(True)
        self.btn_play_rl.setEnabled(True)

    def play_dqn_clicked(self):
        if self.dqn_agent is None:
            QMessageBox.information(self, "DQN", "Najpierw kliknij Train DQN.")
            return

        env = self._make_env_for_current_map()
        if env is None:
            return

        moves = play_dqn(env, self.dqn_agent, max_steps=120) 
        self.reset_map()
        self.start_animation(moves, interval_ms=60)



class DQNTrainWorker(QThread):
    log = Signal(str)
    done = Signal(object)  # agent
    step = Signal(object)

    def __init__(self, env: PushSokobanEnv, show_viz_callback):
        super().__init__()
        self.env = env
        self.visualize_enabled = False  
        self.step_delay_ms = 0

    def run(self):
        def should_vis():
            return self.visualize_enabled

        def on_step(ep, t, eps, payload):
            payload["ep"] = ep
            payload["t"] = t
            payload["eps"] = eps
            self.step.emit(payload)

            if self.step_delay_ms > 0:
                self.msleep(self.step_delay_ms)

        def on_log(ep, total_steps, eps, avg_ret, solved_rate):
            self.log.emit(f"DQN: ep={ep} steps={total_steps} eps={eps:.2f} avg_ret={avg_ret:.1f} solved={solved_rate:.2f}")

        agent = train_dqn(
            self.env,
            episodes=3000,
            on_log=on_log,
            on_step=on_step,
            should_visualize=should_vis,
        )
        self.done.emit(agent)





def main():
    app = QApplication(sys.argv)
    win = SokobanWindow("./maps/sokoban-maps-60-plain.txt")
    win.resize(1600, 900)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
