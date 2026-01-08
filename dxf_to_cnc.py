from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence, TypeAlias

import sys
import math
import ezdxf
import numpy as np

from dueffe_cnc_builder import (
    MachineState,
    SingleHeadCoordinates,
    emit_program,
    end_block,
    save_program,
    starting_block,
    tool_down_single,
    fmt
)

from dueffe_cnc_visualizer import show_interactive

from python_tsp.heuristics import solve_tsp_simulated_annealing



Point: TypeAlias = tuple[float, float]
Vertex: TypeAlias = tuple[float, float, float]  # (x, y, bulge)
TAU = 2.0 * math.pi


@dataclass(frozen=True, slots=True)
class PolylinePath:
    """A single drawable path (polyline/line/arc) represented as bulge vertices."""
    vertices: list[Vertex]
    closed: bool
    handle: str
    dxftype: str
    centroid: Point

@dataclass(slots=True)
class Point2D:
    x: float
    y: float

    def to_tuple(self) -> Point:
        return (self.x, self.y)

    @classmethod
    def from_tuple(cls, point: Point) -> "Point2D":
        return cls(float(point[0]), float(point[1]))


@dataclass(slots=True)
class BulgeVertex:
    x: float
    y: float
    bulge: float

    def to_tuple(self) -> Vertex:
        return (float(self.x), float(self.y), float(self.bulge))

    @classmethod
    def from_tuple(cls, vertex: Vertex) -> "BulgeVertex":
        return cls(float(vertex[0]), float(vertex[1]), float(vertex[2]))


class GeometryProcessor:
    """Geometry helpers and bulge/vertex conversions used across the pipeline."""

    @staticmethod
    def squared_distance(a: Point, b: Point) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return dx * dx + dy * dy

    @staticmethod
    def weld_points(a: Point, b: Point) -> Point:
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    @staticmethod
    def centroid_of(vertices: Sequence[Vertex]) -> Point:
        if not vertices:
            return (0.0, 0.0)
        sx = sum(v[0] for v in vertices)
        sy = sum(v[1] for v in vertices)
        n = len(vertices)
        return (sx / n, sy / n)

    @staticmethod
    def start_point(vertices: Sequence[Vertex]) -> Point:
        return (vertices[0][0], vertices[0][1])

    @staticmethod
    def end_point(vertices: Sequence[Vertex]) -> Point:
        return (vertices[-1][0], vertices[-1][1])

    @staticmethod
    def rotate_closed(vertices: list[Vertex], start_index: int) -> list[Vertex]:
        return vertices[start_index:] + vertices[:start_index]

    @staticmethod
    def reverse_open(vertices: list[Vertex]) -> list[Vertex]:
        n = len(vertices)
        if n < 2:
            return vertices[:]

        coords = [(x, y) for x, y, _ in vertices]
        bulges = [b for _, _, b in vertices]

        coords_rev = coords[::-1]
        reversed_vertices: list[Vertex] = []
        for i, (x, y) in enumerate(coords_rev):
            bulge = -bulges[n - 2 - i] if i < n - 1 else 0.0
            reversed_vertices.append((x, y, bulge))
        return reversed_vertices

    @staticmethod
    def bulge_to_sweep_deg(bulge: float) -> float:
        return math.degrees(4.0 * math.atan(float(bulge)))

    @staticmethod
    def angle_rad(deg: float) -> float:
        return math.radians(float(deg))

    @staticmethod
    def normalize_0_2pi(angle_rad: float) -> float:
        value = angle_rad % TAU
        return value + TAU if value < 0 else value

    @staticmethod
    def to_bulge_vertices(vertices: Sequence[Vertex]) -> list[BulgeVertex]:
        return [BulgeVertex.from_tuple(v) for v in vertices]

    @staticmethod
    def to_legacy_vertices(vertices: Iterable[BulgeVertex]) -> list[Vertex]:
        return [v.to_tuple() for v in vertices]


class DxfParser:
    """Extracts drawable entities (polylines, lines, arcs) from a DXF into PolylinePath objects."""

    def parse_file(
        self,
        dxf_path: str,
        *,
        include_inserts: bool,
        force_close_tol: float,
    ) -> list[PolylinePath]:
        doc = ezdxf.readfile(dxf_path)
        return self.parse_modelspace(
            doc.modelspace(),
            include_inserts=include_inserts,
            force_close_tol=force_close_tol,
        )

    def parse_modelspace(
        self,
        modelspace,
        *,
        include_inserts: bool,
        force_close_tol: float,
    ) -> list[PolylinePath]:
        paths: list[PolylinePath] = []
        for entity in self.iter_entities(modelspace, include_inserts=include_inserts):
            entity_type = entity.dxftype()

            if entity_type in {"LWPOLYLINE", "POLYLINE"}:
                path = self.extract_polyline_path(entity, force_close_tol=force_close_tol)
                if path:
                    paths.append(path)
                continue

            if entity_type == "LINE":
                vertices, centroid = self.line_to_vertices(entity)
                paths.append(
                    PolylinePath(
                        vertices=vertices,
                        closed=False,
                        handle=self.get_handle(entity),
                        dxftype=entity_type,
                        centroid=centroid,
                    )
                )
                continue

            if entity_type == "ARC":
                vertices, centroid = self.arc_to_vertices_with_bulge(entity)
                paths.append(
                    PolylinePath(
                        vertices=vertices,
                        closed=False,
                        handle=self.get_handle(entity),
                        dxftype=entity_type,
                        centroid=centroid,
                    )
                )
                continue

        return paths

    def iter_entities(self, modelspace, *, include_inserts: bool) -> Iterable:
        for entity in modelspace:
            if entity.dxftype() == "INSERT" and include_inserts:
                try:
                    yield from entity.virtual_entities()
                except Exception:
                    continue
            else:
                yield entity

    def extract_polyline_path(self, entity, *, force_close_tol: float) -> PolylinePath | None:
        entity_type = entity.dxftype()

        if entity_type == "LWPOLYLINE":
            vertices = self.extract_vertices_lwpolyline(entity)
        elif entity_type == "POLYLINE":
            vertices = self.extract_vertices_polyline(entity)
        else:
            return None

        if not vertices or len(vertices) < 2:
            return None

        closed = self.is_closed(entity)
        if not closed:
            (x0, y0, _), (x1, y1, _) = vertices[0], vertices[-1]
            closed = (math.hypot(x1 - x0, y1 - y0) <= force_close_tol)

        return PolylinePath(
            vertices=vertices,
            closed=closed,
            handle=self.get_handle(entity),
            dxftype=entity_type,
            centroid=GeometryProcessor.centroid_of(vertices),
        )

    def extract_vertices_lwpolyline(self, entity) -> list[Vertex] | None:
        try:
            raw_points = list(entity.get_points())
        except Exception:
            return None

        if len(raw_points) < 2:
            return None

        vertices: list[Vertex] = []
        for point_data in raw_points:
            x = float(point_data[0])
            y = float(point_data[1])
            bulge = float(point_data[4]) if len(point_data) >= 5 else 0.0
            vertices.append((x, y, bulge))
        return vertices

    def extract_vertices_polyline(self, entity) -> list[Vertex] | None:
        if self.bool_attr_or_call(entity, "is_poly_face_mesh") or self.bool_attr_or_call(entity, "is_polygon_mesh"):
            return None

        vertex_entities = []
        try:
            vertex_container = getattr(entity, "vertices", None)
            if vertex_container is not None:
                vertex_entities = list(vertex_container)
        except Exception:
            vertex_entities = []

        if len(vertex_entities) >= 2:
            vertices: list[Vertex] = []
            for vertex_entity in vertex_entities:
                location = getattr(vertex_entity.dxf, "location", None)
                if location is None:
                    return None
                x = float(location.x)
                y = float(location.y)
                bulge = float(getattr(vertex_entity.dxf, "bulge", 0.0) or 0.0)
                vertices.append((x, y, bulge))
            return vertices

        try:
            points = list(entity.points())
        except Exception:
            return None

        if len(points) < 2:
            return None

        fallback_vertices: list[Vertex] = []
        for point in points:
            try:
                x = float(point.x)
                y = float(point.y)
            except Exception:
                x = float(point[0])
                y = float(point[1])
            fallback_vertices.append((x, y, 0.0))
        return fallback_vertices

    def arc_to_vertices_with_bulge(self, arc_entity) -> tuple[list[Vertex], Point]:
        center = arc_entity.dxf.center
        radius = float(arc_entity.dxf.radius)

        start_angle = GeometryProcessor.normalize_0_2pi(
            GeometryProcessor.angle_rad(arc_entity.dxf.start_angle)
        )
        end_angle = GeometryProcessor.normalize_0_2pi(
            GeometryProcessor.angle_rad(arc_entity.dxf.end_angle)
        )

        sweep = (end_angle - start_angle) % (2.0 * math.pi)
        if sweep < 1e-12:
            sweep = 2.0 * math.pi

        x0 = float(center.x) + radius * math.cos(start_angle)
        y0 = float(center.y) + radius * math.sin(start_angle)
        x1 = float(center.x) + radius * math.cos(end_angle)
        y1 = float(center.y) + radius * math.sin(end_angle)

        bulge = math.tan(sweep / 4.0)
        vertices: list[Vertex] = [
            (x0, y0, bulge),
            (x1, y1, 0.0),
        ]

        mid_angle = start_angle + sweep / 2.0
        centroid_x = float(center.x) + radius * math.cos(mid_angle)
        centroid_y = float(center.y) + radius * math.sin(mid_angle)
        return vertices, (centroid_x, centroid_y)

    def line_to_vertices(self, line_entity) -> tuple[list[Vertex], Point]:
        start = line_entity.dxf.start
        end = line_entity.dxf.end
        x0, y0 = float(start.x), float(start.y)
        x1, y1 = float(end.x), float(end.y)
        vertices: list[Vertex] = [(x0, y0, 0.0), (x1, y1, 0.0)]
        return vertices, ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    def get_handle(self, entity) -> str:
        return getattr(getattr(entity, "dxf", None), "handle", "?")

    def is_closed(self, entity) -> bool:
        for attr in ("closed", "is_closed"):
            value = getattr(entity, attr, None)
            if isinstance(value, bool):
                return value
            if callable(value):
                try:
                    return bool(value())
                except Exception:
                    pass
        return False

    def bool_attr_or_call(self, entity, name: str) -> bool:
        value = getattr(entity, name, None)
        if value is None:
            return False
        if callable(value):
            try:
                return bool(value())
            except Exception:
                return False
        return bool(value)

    def print_entities(self, file_path: str) -> None:
        dxf_document = ezdxf.readfile(file_path)
        model_space = dxf_document.modelspace()

        for drawing_entity in model_space:
            entity_type = drawing_entity.dxftype()
            print(f"Type: {entity_type}")

            if entity_type == "LWPOLYLINE":
                for point_data in drawing_entity.get_points():
                    print(f"  Vertex: {point_data[:2]} | Bulge: {point_data[4]}")

            elif entity_type == "POLYLINE":
                for vertex_entity in drawing_entity.vertices:
                    print(f"  Vertex: {vertex_entity.dxf.location} | Bulge: {vertex_entity.dxf.bulge}")

            elif entity_type == "LINE":
                print(f"  Start Point: {drawing_entity.dxf.start}")
                print(f"  End Point: {drawing_entity.dxf.end}")

            elif entity_type in {"CIRCLE", "ARC"}:
                print(f"  Center Point: {drawing_entity.dxf.center}")

            elif entity_type == "POINT":
                print(f"  Location: {drawing_entity.dxf.location}")

            elif entity_type == "SPLINE":
                print(f"  Control Points: {drawing_entity.control_points}")


class PathStitcher:
    """Merges open paths when endpoints touch (within tolerance), preserving bulge semantics at joins."""

    def stitch_connected_paths(
        self,
        paths: list[PolylinePath],
        *,
        tol: float = 0.05,
        force_close_gaps: bool = False,
    ) -> list[PolylinePath]:
        tol2 = tol * tol

        closed_paths = [p for p in paths if p.closed]
        open_paths = [p for p in paths if not p.closed]

        if not open_paths:
            return paths

        start_map: dict[tuple[int, int], set[int]] = defaultdict(set)
        end_map: dict[tuple[int, int], set[int]] = defaultdict(set)

        def cell_key(point: Point) -> tuple[int, int]:
            return (int(round(point[0] / tol)), int(round(point[1] / tol)))

        def neighbor_keys(point: Point):
            base_x, base_y = cell_key(point)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    yield (base_x + dx, base_y + dy)

        def register(path_index: int) -> None:
            start = GeometryProcessor.start_point(open_paths[path_index].vertices)
            end = GeometryProcessor.end_point(open_paths[path_index].vertices)
            for key in neighbor_keys(start):
                start_map[key].add(path_index)
            for key in neighbor_keys(end):
                end_map[key].add(path_index)

        def unregister(path_index: int) -> None:
            start = GeometryProcessor.start_point(open_paths[path_index].vertices)
            end = GeometryProcessor.end_point(open_paths[path_index].vertices)
            for key in neighbor_keys(start):
                start_map[key].discard(path_index)
            for key in neighbor_keys(end):
                end_map[key].discard(path_index)

        for idx in range(len(open_paths)):
            register(idx)

        remaining = set(range(len(open_paths)))
        stitched: list[PolylinePath] = []

        def best_match(point: Point) -> tuple[int, str] | None:
            best: tuple[float, int, str] | None = None

            for key in neighbor_keys(point):
                for candidate_index in start_map.get(key, ()):
                    if candidate_index not in remaining:
                        continue
                    candidate_start = GeometryProcessor.start_point(open_paths[candidate_index].vertices)
                    d2 = GeometryProcessor.squared_distance(point, candidate_start)
                    if d2 <= tol2 and (best is None or d2 < best[0]):
                        best = (d2, candidate_index, "start")

                for candidate_index in end_map.get(key, ()):
                    if candidate_index not in remaining:
                        continue
                    candidate_end = GeometryProcessor.end_point(open_paths[candidate_index].vertices)
                    d2 = GeometryProcessor.squared_distance(point, candidate_end)
                    if d2 <= tol2 and (best is None or d2 < best[0]):
                        best = (d2, candidate_index, "end")

            return (best[1], best[2]) if best else None

        while remaining:
            seed_index = remaining.pop()
            unregister(seed_index)

            chain = GeometryProcessor.to_bulge_vertices(open_paths[seed_index].vertices)
            handles = [open_paths[seed_index].handle]
            types = [open_paths[seed_index].dxftype]
            closed = False

            while True:
                extended = False

                chain_start = (chain[0].x, chain[0].y)
                chain_end = (chain[-1].x, chain[-1].y)

                forward_match = best_match(chain_end)
                if forward_match:
                    candidate_index, match_side = forward_match
                    remaining.remove(candidate_index)
                    unregister(candidate_index)

                    candidate_vertices = open_paths[candidate_index].vertices
                    if match_side == "end":
                        candidate_vertices = GeometryProcessor.reverse_open(candidate_vertices)

                    candidate = GeometryProcessor.to_bulge_vertices(candidate_vertices)

                    weld_point = GeometryProcessor.weld_points(chain_end, (candidate[0].x, candidate[0].y))
                    chain[-1].x = weld_point[0]
                    chain[-1].y = weld_point[1]
                    chain[-1].bulge = candidate[0].bulge

                    candidate_end_point = (candidate[-1].x, candidate[-1].y)
                    if GeometryProcessor.squared_distance(candidate_end_point, chain_start) <= tol2:
                        close_weld = GeometryProcessor.weld_points(chain_start, candidate_end_point)
                        chain[0].x = close_weld[0]
                        chain[0].y = close_weld[1]

                        if len(candidate) > 2:
                            chain.extend(candidate[1:-1])

                        closed = True
                        handles.append(open_paths[candidate_index].handle)
                        types.append(open_paths[candidate_index].dxftype)
                        break

                    if len(candidate) > 1:
                        chain.extend(candidate[1:])

                    handles.append(open_paths[candidate_index].handle)
                    types.append(open_paths[candidate_index].dxftype)
                    extended = True

                if closed:
                    break

                if extended:
                    continue

                chain_start = (chain[0].x, chain[0].y)
                backward_match = best_match(chain_start)
                if backward_match:
                    candidate_index, match_side = backward_match
                    remaining.remove(candidate_index)
                    unregister(candidate_index)

                    candidate_vertices = open_paths[candidate_index].vertices
                    if match_side == "start":
                        candidate_vertices = GeometryProcessor.reverse_open(candidate_vertices)

                    candidate = GeometryProcessor.to_bulge_vertices(candidate_vertices)

                    candidate_end = (candidate[-1].x, candidate[-1].y)
                    weld_point = GeometryProcessor.weld_points(candidate_end, chain_start)

                    outgoing_bulge = chain[0].bulge
                    candidate[-1].x = weld_point[0]
                    candidate[-1].y = weld_point[1]
                    candidate[-1].bulge = outgoing_bulge

                    chain = candidate + chain[1:]

                    handles.append(open_paths[candidate_index].handle)
                    types.append(open_paths[candidate_index].dxftype)
                    extended = True

                if not extended:
                    break

            if not closed and force_close_gaps:
                start_pt = (chain[0].x, chain[0].y)
                end_pt = (chain[-1].x, chain[-1].y)
                if GeometryProcessor.squared_distance(start_pt, end_pt) <= tol2:
                    weld_point = GeometryProcessor.weld_points(start_pt, end_pt)
                    chain[0].x = weld_point[0]
                    chain[0].y = weld_point[1]
                    chain[-1].x = weld_point[0]
                    chain[-1].y = weld_point[1]
                    chain[-1].bulge = 0.0
                    closed = True
                    chain = chain[:-1]

            legacy_vertices = GeometryProcessor.to_legacy_vertices(chain)
            stitched.append(
                PolylinePath(
                    vertices=legacy_vertices,
                    closed=closed,
                    handle=",".join(handles),
                    dxftype="CHAIN[" + ",".join(sorted(set(types))) + "]",
                    centroid=GeometryProcessor.centroid_of(legacy_vertices),
                )
            )

        return closed_paths + stitched


class TspOptimizer:
    """Plans a near-optimal traversal order of paths to reduce travel moves."""

    def plan_tsp_order(
        self,
        paths: list[PolylinePath],
        *,
        start_xy: Point = (0.0, 0.0),
        verbose: bool = True,
    ) -> list[tuple[PolylinePath, list[Vertex]]]:
        if not paths:
            return []

        nodes: list[Point] = [start_xy] + [p.centroid for p in paths]

        if verbose:
            print(f"Calculating TSP path for {len(paths)} entities...")

        permutation, _ = solve_tsp_simulated_annealing(self.distance_matrix(nodes))
        ordered_nodes = self.rotate_to_start(list(permutation), 0)

        planned: list[tuple[PolylinePath, list[Vertex]]] = []
        current_xy = start_xy

        for node_idx in ordered_nodes[1:]:
            path = paths[node_idx - 1]
            chosen_vertices, _, end_xy = self.choose_entry_for_next_path(path, current_xy)
            planned.append((path, chosen_vertices))
            current_xy = end_xy

        return planned

    def choose_entry_for_next_path(
        self,
        path: PolylinePath,
        current_xy: Point,
    ) -> tuple[list[Vertex], Point, Point]:
        vertices = path.vertices

        if path.closed:
            best_index = min(
                range(len(vertices)),
                key=lambda i: GeometryProcessor.squared_distance(
                    current_xy, (vertices[i][0], vertices[i][1])
                ),
            )
            chosen = GeometryProcessor.rotate_closed(vertices, best_index)
            start_x, start_y, _ = chosen[0]
            start_point = (start_x, start_y)
            return chosen, start_point, start_point

        forward_start = (vertices[0][0], vertices[0][1])
        forward_cost = GeometryProcessor.squared_distance(current_xy, forward_start)

        reversed_vertices = GeometryProcessor.reverse_open(vertices)
        reverse_start = (reversed_vertices[0][0], reversed_vertices[0][1])
        reverse_cost = GeometryProcessor.squared_distance(current_xy, reverse_start)

        if reverse_cost < forward_cost:
            start_x, start_y, _ = reversed_vertices[0]
            end_x, end_y, _ = reversed_vertices[-1]
            return reversed_vertices, (start_x, start_y), (end_x, end_y)

        start_x, start_y, _ = vertices[0]
        end_x, end_y, _ = vertices[-1]
        return vertices, (start_x, start_y), (end_x, end_y)

    def distance_matrix(self, points: list[Point]) -> np.ndarray:
        coords = np.asarray(points, dtype=float)
        diff = coords[:, None, :] - coords[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=-1))

    def rotate_to_start(self, permutation: list[int], start_node: int = 0) -> list[int]:
        if start_node not in permutation:
            return permutation
        index = permutation.index(start_node)
        return permutation[index:] + permutation[:index]


class CncGenerator:
    """Converts vertex sequences into CNC program blocks for a single head."""

    def polyline_to_cnc_single(
        self,
        state: MachineState,
        vertices: Sequence[Vertex],
        *,
        closed: bool,
        bulge_eps: float = 1e-9,
        add_semicolon_line: bool = False,
    ) -> str:
        if len(vertices) < 2:
            return ""

        x0, y0, _ = vertices[0]
        lines: list[str] = [f"MR X{fmt(x0)}Y{fmt(y0)}"]

        if add_semicolon_line:
            lines.append(";")

        lines.extend(tool_down_single(state))

        in_arc_run = False
        n = len(vertices)
        segment_count = n if closed else n - 1

        for i in range(segment_count):
            _, _, bulge = vertices[i]
            x1, y1, _ = vertices[(i + 1) % n]

            if abs(bulge) <= bulge_eps:
                if in_arc_run:
                    lines.append("SYNC")
                    in_arc_run = False
                lines.append(f"MI X{fmt(x1)}Y{fmt(y1)}")
                continue

            if not in_arc_run:
                lines.append("FREEZE")
                in_arc_run = True

            sweep = GeometryProcessor.bulge_to_sweep_deg(bulge)
            lines.append(f"ARC X{fmt(x1)}Y{fmt(y1)} a={fmt(sweep)}")

        if in_arc_run:
            lines.append("SYNC")

        lines.append("CALL UP1")
        return "\n".join(lines)


def pt(xy: Vertex) -> Point:
    return (xy[0], xy[1])


def dist2(a: Point, b: Point) -> float:
    return GeometryProcessor.squared_distance(a, b)


def weld(a: Point, b: Point) -> Point:
    return GeometryProcessor.weld_points(a, b)


def bulge_to_sweep_deg(bulge: float) -> float:
    return GeometryProcessor.bulge_to_sweep_deg(bulge)


def squared_distance(a: Point, b: Point) -> float:
    return GeometryProcessor.squared_distance(a, b)


def centroid_of(vertices: Sequence[Vertex]) -> Point:
    return GeometryProcessor.centroid_of(vertices)


def rotate_closed(vertices: list[Vertex], start_index: int) -> list[Vertex]:
    return GeometryProcessor.rotate_closed(vertices, start_index)


def reverse_open(vertices: list[Vertex]) -> list[Vertex]:
    return GeometryProcessor.reverse_open(vertices)


def stitch_connected_paths(
    paths: list[PolylinePath],
    *,
    tol: float = 0.05,
    force_close_gaps: bool = False,
) -> list[PolylinePath]:
    """
    Merge paths that touch end-to-start (within tol) into longer continuous paths.
    Preserves bulge at joins by moving the next path's start bulge onto the shared vertex.
    """
    return PathStitcher().stitch_connected_paths(paths, tol=tol, force_close_gaps=force_close_gaps)


def distance_matrix(points: Sequence[Point]) -> np.ndarray:
    return TspOptimizer().distance_matrix(list(points))


def rotate_to_start(permutation: list[int], start_node: int = 0) -> list[int]:
    return TspOptimizer().rotate_to_start(permutation, start_node)


def choose_entry_for_next_path(path: PolylinePath, current_xy: Point) -> tuple[list[Vertex], Point, Point]:
    return TspOptimizer().choose_entry_for_next_path(path, current_xy)


def plan_tsp_order(
    paths: list[PolylinePath],
    *,
    start_xy: Point = (0.0, 0.0),
    verbose: bool = True,
) -> list[tuple[PolylinePath, list[Vertex]]]:
    return TspOptimizer().plan_tsp_order(paths, start_xy=start_xy, verbose=verbose)


def polyline_to_cnc_single(
    state: MachineState,
    vertices: Sequence[Vertex],
    *,
    closed: bool,
    bulge_eps: float = 1e-9,
    add_semicolon_line: bool = False,
) -> str:
    return CncGenerator().polyline_to_cnc_single(
        state,
        vertices,
        closed=closed,
        bulge_eps=bulge_eps,
        add_semicolon_line=add_semicolon_line,
    )


def iter_entities(modelspace, *, include_inserts: bool) -> Iterable:
    return DxfParser().iter_entities(modelspace, include_inserts=include_inserts)


def extract_polyline_path(entity, *, force_close_tol: float = 0.05) -> PolylinePath | None:
    return DxfParser().extract_polyline_path(entity, force_close_tol=force_close_tol)


def dxf_to_cnc_single_polylines(
    dxf_path: str,
    *,
    design_name: str | None = None,
    include_inserts: bool = True,
    force_close_tol: float = 0.05,
    bulge_eps: float = 1e-9,
    add_semicolon_line: bool = False,
    optimize_travel: bool = True,
) -> list[str]:
    parser = DxfParser()
    stitcher = PathStitcher()
    tsp = TspOptimizer()
    generator = CncGenerator()

    name = design_name or Path(dxf_path).stem
    state = MachineState()

    program: list[str] = [starting_block(SingleHeadCoordinates(0, 0), name)]

    paths = parser.parse_file(dxf_path, include_inserts=include_inserts, force_close_tol=force_close_tol)

    print(f"Raw entities: {len(paths)}")
    paths = stitcher.stitch_connected_paths(paths, tol=0.05, force_close_gaps=True)
    print(f"Stitched continuous paths: {len(paths)}")

    planned = tsp.plan_tsp_order(paths, start_xy=(0.0, 0.0)) if optimize_travel else [(p, p.vertices) for p in paths]

    for path, chosen_vertices in planned:
        try:
            block = generator.polyline_to_cnc_single(
                state,
                chosen_vertices,
                closed=path.closed,
                bulge_eps=bulge_eps,
                add_semicolon_line=add_semicolon_line,
            )
            if block.strip():
                program.append(block)
        except Exception as exc:
            program.append(f"; skipped polyline handle={path.handle} type={path.dxftype} error={exc}")

    program.append(end_block())
    return program


def print_dxf_entities(file_path: str) -> None:
    DxfParser().print_entities(file_path)


def main() -> None:
    default_dxf = r"C:\Users\server\PycharmProjects\dc2-dueffe-converter\DXF\TESLA_190x140.dxf"
    dxf_path = sys.argv[1] if len(sys.argv) >= 2 else default_dxf
    out_path = sys.argv[2] if len(sys.argv) >= 3 else str(Path(dxf_path).with_suffix(".CNC"))

    print_dxf_entities(dxf_path)

    program_lines = dxf_to_cnc_single_polylines(
        dxf_path,
        design_name=Path(dxf_path).stem,
        include_inserts=True,
        force_close_tol=0.05,
        bulge_eps=1e-9,
        add_semicolon_line=False,
        optimize_travel=True,
    )

    cnc_text = emit_program(program_lines, crlf=False)
    save_program(cnc_text, out_path, crlf=True)
    print(f"Saved CNC: {out_path}")
    show_interactive(f"{out_path}")


if __name__ == "__main__":
    main()


