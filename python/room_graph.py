#! /usr/bin/env python
"""Room graph: room-layer serialization for the multi-layer scene graph (v2.2).

Rooms are computed by the consumer repo (apexnav object_mapping/room_mapping);
this module only defines storage: ``rooms.json`` (nodes) + ``edges_room.txt``
(room adjacency; the weight is the number of cross-room trav edges -- frames
the robot actually walked between, i.e. real door-crossing evidence. Covis
edges are deliberately NOT used: facing views through a doorway also overlap,
which would connect every room to every other).

Edge-list I/O is overridden here: ``BaseGraph.write_edge_list`` goes through
``np.savetxt(fmt='%d %d %.6f')`` which cannot format str ids ("room_0"), and
``read_edge_list`` int-casts the first two columns. ObjectGraph never hit this
only because ``edges_object.txt`` has been empty so far.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.base_graph import BaseGraph  # litevloc read-only base

from room_node import RoomNode

# rooms.json is a v2.2 side-car of objects.json; both carry the same version.
from object_node import SCHEMA_VERSION

ROOM_FRAME = "ros_zup"  # x fwd / y left / z up, origin at episode start ground


class RoomGraph(BaseGraph):
    """Graph of detected rooms (str ids) with room-adjacency edges."""

    def __init__(self, map_root: Path, edge_type: str = "room") -> None:
        super().__init__(map_root, edge_type)

    def embedding_dims(self) -> Dict[str, int]:
        for node in self.nodes.values():
            return {k: int(np.asarray(v).reshape(-1).shape[0]) for k, v in node.embeddings.items()}
        return {}

    def write_edge_list(self, edge_list_path: Path) -> None:
        """Plain-text `id id weight` rows; str ids make np.savetxt unusable here."""
        lines = []
        for node in self.nodes.values():
            for neighbor, weight in node.edges.values():
                if str(node.id) < str(neighbor.id):  # any strict order dedups the undirected pair
                    lines.append(f"{node.id} {neighbor.id} {float(weight):.6f}")
        Path(edge_list_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def read_edge_list(self, edge_list_path: Path) -> None:
        edge_list_path = Path(edge_list_path)
        if not edge_list_path.exists():
            print(f"Edge list {str(edge_list_path)} file not found")
            return
        for line in edge_list_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 3:
                raise ValueError(f"Malformed room edge line: {line!r}")
            node0, node1 = self.get_node(parts[0]), self.get_node(parts[1])
            if node0 is not None and node1 is not None:
                self.add_edge_undirected(node0, node1, float(parts[2]))

    def save_to_file(self, edge_only: bool = False) -> None:
        """Write rooms.json (schema_version + rooms) and edges_room.txt."""
        if not edge_only:
            payload = {
                "schema_version": SCHEMA_VERSION,
                "edge_type": self.edge_type,
                "frame": ROOM_FRAME,
                "embedding_dims": self.embedding_dims(),
                "rooms": [node.to_dict() for node in self.nodes.values()],
            }
            with open(self.map_root / "rooms.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
        self.write_edge_list(self.map_root / f"edges_{self.edge_type}.txt")


class RoomGraphLoader:
    """Loads a RoomGraph from rooms.json + edges_room.txt."""

    @staticmethod
    def load_data(map_root: Path, edge_type: str = "room") -> RoomGraph:
        graph = RoomGraph(map_root, edge_type)
        rooms_path = map_root / "rooms.json"
        if rooms_path.exists():
            payload = json.loads(rooms_path.read_text(encoding="utf-8"))
            for room_dict in payload.get("rooms", []):
                graph.add_node(RoomNode.from_dict(room_dict))
        graph.read_edge_list(map_root / f"edges_{edge_type}.txt")
        return graph
