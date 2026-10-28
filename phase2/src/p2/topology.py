"""Parse an OMNeT++ .ned topology (BNN-UPC RouteNet style) into capacities.

Only what the file states is used. Units: the NED declares ``datarate = Nkbps``;
the capacity is carried in these units ("capacity units", CU) and never rescaled
here. Each ``<-->`` connection is treated as a duplex link with ``datarate`` in
EACH direction (OMNeT++ duplex-channel semantics; an assumption not checked
against the simulator).
"""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_CHAN = re.compile(r"channel\s+(\w+)\s+extends\s+ned\.DatarateChannel\s*\{[^}]*?datarate\s*=\s*([\d.]+)\s*(k|M|G)?bps", re.S)
_NODE = re.compile(r"node(\d+)\s*:\s*\w+\s*\{.*?port\[(\d+)\]", re.S)
_CONN = re.compile(r"node(\d+)\.port\[(\d+)\]\s*<-->\s*(\w+)\s*<-->\s*node(\d+)\.port\[(\d+)\]\s*;")
_MULT = {None: 1e-3, "k": 1.0, "M": 1e3, "G": 1e6}  # -> kbps


@dataclass(frozen=True)
class Topology:
    n_nodes: int
    undirected: tuple          # ((u, v, capacity_kbps, port_u, port_v), ...), u < v
    declared_ports: tuple      # declared number of ports per node
    source_sha256: str

    @property
    def n_links(self):
        return len(self.undirected)

    def directed_edges(self):
        """Directed edges as (u, v, capacity); edge 2i is u->v, 2i+1 is v->u."""
        out = []
        for (u, v, c, _, _) in self.undirected:
            out.append((u, v, c))
            out.append((v, u, c))
        return out

    def capacity_vector(self):
        return np.array([c for (_, _, c) in self.directed_edges()], dtype=np.float64)


def load_ned(path):
    import hashlib
    path = Path(path)
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    chans = {name: float(val) * _MULT[unit] for name, val, unit in _CHAN.findall(text)}
    ports = {}
    for m in re.finditer(r"node(\d+)\s*:\s*Server\s*\{(.*?)\n\s*\}", text, re.S):
        pm = re.search(r"port\[(\d+)\]", m.group(2))
        ports[int(m.group(1))] = int(pm.group(1)) if pm else 0
    links = []
    for a, pa, ch, b, pb in _CONN.findall(text):
        a, pa, b, pb = int(a), int(pa), int(b), int(pb)
        if ch not in chans:
            raise ValueError(f"unknown channel type {ch}")
        u, v, pu, pv = (a, b, pa, pb) if a < b else (b, a, pb, pa)
        links.append((u, v, chans[ch], pu, pv))
    n = max(ports) + 1
    links.sort()
    return Topology(n, tuple(links), tuple(ports[i] for i in range(n)),
                    hashlib.sha256(raw).hexdigest())
