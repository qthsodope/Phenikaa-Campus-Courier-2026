"""Kiểu dữ liệu chung: bản đồ campus + yêu cầu, dựng từ scenes.json (hoặc từ khâu đọc ảnh/tin nhắn)."""
from dataclasses import dataclass, field

ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
DELTA = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # action -> (dr, dc)
HEADING = {name: i for i, name in enumerate(ACTIONS)}


@dataclass
class Edge:
    status: str            # normal / crowded / covered / closed
    stairs: bool
    oneway_to: tuple | None


@dataclass
class Mission:
    goal: str
    via: str | None = None
    goal_rc: tuple | None = None   # bản cụ thể khi có tham chiếu không gian
    via_rc: tuple | None = None
    urgent: bool = False
    fragile: bool = False


@dataclass
class Scene:
    rows: int
    cols: int
    nodes: set
    edges: dict              # frozenset({a, b}) -> Edge
    landmarks: dict          # type -> list[rc]
    robot: tuple
    heading: int             # 0..3 như ACTIONS
    rain: bool
    mission: Mission
    night: bool = False      # V2: ảnh kiểu "đêm" (nền tối) — điều kiện ẩn làm nhiều robot đổi cách đi
    adj: dict = field(default_factory=dict)  # rc -> list[(action, rc2, Edge)]

    def __post_init__(self):
        self.adj = {n: [] for n in self.nodes}
        for a in self.nodes:
            for act, (dr, dc) in DELTA.items():
                b = (a[0] + dr, a[1] + dc)
                e = self.edges.get(frozenset((a, b)))
                if e is not None:
                    self.adj[a].append((act, b, e))

    def can_move(self, a, b, e, legged=False):
        if e.status == "closed":
            return False
        if e.oneway_to is not None and e.oneway_to != b:
            return False
        if e.stairs and not legged:
            return False
        return True


def _rc(x):
    return tuple(x) if x is not None else None


def scene_from_json(s):
    nodes = {tuple(n["rc"]) for n in s["nodes"]}
    edges = {frozenset((tuple(e["a"]), tuple(e["b"]))): Edge(e["status"], e["stairs"], _rc(e["oneway_to"]))
             for e in s["edges"]}
    lms = {}
    for lm in s["landmarks"]:
        lms.setdefault(lm["type"], []).append(tuple(lm["rc"]))
    m = s["mission"]
    mission = Mission(
        goal=m["goal"], via=m["via"],
        goal_rc=_rc(m["goal_ref"]["rc"]) if m["goal_ref"] else None,
        via_rc=_rc(m["via_ref"]["rc"]) if m["via_ref"] else None,
        urgent=m["urgent"], fragile=m["fragile"])
    return Scene(rows=s["grid"]["rows"], cols=s["grid"]["cols"], nodes=nodes, edges=edges, landmarks=lms,
                 robot=tuple(s["robot"]["rc"]), heading=HEADING[s["robot"]["heading"]],
                 rain=s["weather"] == "rain", mission=mission, night=s.get("style") == "night")
