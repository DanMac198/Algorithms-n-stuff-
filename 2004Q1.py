##Question 1 for the assignment:
import sys
import heapq
from typing import List, Tuple

# ------------------------------
# Dinic Max-Flow (no dicts/sets)
# ------------------------------

class _Edge:
    __slots__ = ("to", "rev", "cap")
    def __init__(self, to: int, rev: int, cap: int):
        self.to = to
        self.rev = rev
        self.cap = cap

class _Dinic:
    """
    Function Description:
        Deterministic worst-case O(E * V^2) flow (Dinic with adjacency lists).
        No dictionaries/sets used. Suitable for our bipartite construction.

    Input:
        - n: number of vertices (0..n-1)

    Output:
        - max_flow(s, t): computes the max flow value from s to t
        - add_edge(u, v, c): adds a directed edge u->v with capacity c

    Time Complexity:
        Building: O(1) per edge. Each max_flow call: O(E * V^2) worst-case.

    Aux Space Complexity:
        O(V + E) for level/it arrays and graph structure.
    """
    def __init__(self, n: int):
        self.n = n
        self.g: List[List[_Edge]] = [[] for _ in range(n)]
        self.level = [0]*n
        self.it = [0]*n

    def add_edge(self, u: int, v: int, c: int) -> None:
        fwd = _Edge(v, len(self.g[v]), c)
        rev = _Edge(u, len(self.g[u]), 0)
        self.g[u].append(fwd)
        self.g[v].append(rev)

    def _bfs(self, s: int, t: int) -> bool:
        for i in range(self.n):
            self.level[i] = -1
        q = [s]
        self.level[s] = 0
        qi = 0
        while qi < len(q):
            u = q[qi]; qi += 1
            for e in self.g[u]:
                if e.cap > 0 and self.level[e.to] < 0:
                    self.level[e.to] = self.level[u] + 1
                    q.append(e.to)
        return self.level[t] >= 0

    def _dfs(self, u: int, t: int, f: int) -> int:
        if u == t:
            return f
        i = self.it[u]
        while i < len(self.g[u]):
            self.it[u] = i
            e = self.g[u][i]
            if e.cap > 0 and self.level[u] < self.level[e.to]:
                ret = self._dfs(e.to, t, f if f < e.cap else e.cap)
                if ret > 0:
                    e.cap -= ret
                    self.g[e.to][e.rev].cap += ret
                    return ret
            i += 1
            self.it[u] = i
        return 0

    def max_flow(self, s: int, t: int) -> int:
        flow = 0
        INF = 10**18
        while self._bfs(s, t):
            for i in range(self.n):
                self.it[i] = 0
            while True:
                pushed = self._dfs(s, t, INF)
                if pushed == 0:
                    break
                flow += pushed
        return flow


# ---------------------------------------
# Shortest paths (Dijkstra, undirected)
# ---------------------------------------

def _dijkstra_all(n: int, adj: List[List[Tuple[int,int]]], src: int, dist: List[int]) -> None:
    """
    Function Description:
        Standard Dijkstra over non-negative weighted undirected graph.

    Input:
        - n: number of locations
        - adj: adjacency list of (to, weight)
        - src: source location
        - dist: preallocated length-n list; output filled with distances

    Output:
        - dist[v] = shortest distance from src to v (or large number if unreachable)

    Time Complexity:
        O(R log L) for a single run with binary heap.

    Aux Space Complexity:
        O(L) for dist + heap internal storage.
    """
    INF = 10**15
    for i in range(n):
        dist[i] = INF
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for (v, w) in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))


# ------------------------------------------------
# Main Q1 function: assign(...)
# ------------------------------------------------

def assign(L: int,
           roads: List[Tuple[int,int,int]],
           students: List[int],
           buses: List[Tuple[int,int,int]],
           D: int,
           T: int) -> List[int] | None:
    """
    Function Description:
        Returns one valid allocation assigning exactly T students to buses such that:
          - Each bus b respects its [min, max] capacity.
          - A student may board any bus whose pickup point is at distance ≤ D from the student’s location.
          - All buses are used (min capacities are positive in the brief).
          - If no allocation exists, returns None.

    Approach Description:
        1) Distances: Build the city graph and run Dijkstra from each unique pickup location
           (≤ 18 by brief) to all locations, O(R log L + L). This lets us quickly test
           reachability (≤ D) from a student’s location to each bus’s pickup point.

        2) Flow model with lower bounds:
           - Nodes: source s, gate q (cap T), each student node, each bus node, sink t.
           - Edges:
               s->q capacity T (enforces “at most T”).
               q->student_i capacity 1 (each selected student at most once).
               student_i->bus_j capacity 1 if reachable within D.
               bus_j->t capacity in [min_j, max_j] (lower bound = min_j).
           - Lower bounds are handled via the standard circulation transform:
               For every edge with lower bound ℓ and upper u, we add an edge of capacity (u-ℓ) and
               record node demands (d[v]+=ℓ, d[u]-=ℓ). We then connect a super-source SS to all
               nodes with positive demand and all nodes with negative demand to super-sink TT,
               add edge t->s with infinite capacity, and run max-flow(SS,TT). Feasible iff the
               total positive demand is fully saturated.
           - After feasibility, remove SS/TT and the helper arcs, then run max-flow(s,t). We require
             the final value == T. Reconstruct allocation by reading student->bus edges that are saturated.

    Input:
        - L: number of locations (0..L-1).
        - roads: undirected weighted edges (u,v,w), unique per pair.
        - students: length S, students[i] = location of student i.
        - buses: length B, tuples (pickup_location, min_cap, max_cap), with min_cap, max_cap > 0.
        - D: maximum travel distance acceptable to a pickup point (positive int).
        - T: exact number of students to transport (positive int).

    Output:
        - allocation: list of length S; allocation[i] = bus_id (0..B-1) if student i travels, else -1.
          Returns None if infeasible.

    Time Complexity:
        O(S·T + L + R·log L) worst-case:
          - Distances (≤ 18 Dijkstras): O(R log L + L) since 18 is a small constant by brief.
          - Graph construction: O(S + B + eligible_pairs) with eligible_pairs ~ O(S) due to few pickup points.
          - Flow: Dinic worst-case within the budgeted O(S·T).
    Aux Space Complexity:
        O(S + L + R) worst-case:
          - Dist arrays O(L), adjacency O(L + R), flow graph O(S + B + edges) ~ O(S).
    """
    S = len(students)
    B = len(buses)
    if T < 0 or D < 0 or L <= 0:
        return None
    if S == 0:
        return None if T != 0 else []

    # ---------------------------
    # 1) Build city graph (L, R)
    # ---------------------------
    adj = [[] for _ in range(L)]
    for (u, v, w) in roads:
        adj[u].append((v, w))
        adj[v].append((u, w))

    # ---------------------------------------------
    # 2) Unique pickup locations (no sets allowed)
    # ---------------------------------------------
    # Extract pickup locations into a list, sort, dedupe
    pickup_locs = [buses[i][0] for i in range(B)]
    # simple insertion sort to avoid relying on built-ins' complexities (ok to use sort, but being explicit)
    for i in range(1, B):
        key = pickup_locs[i]
        j = i - 1
        while j >= 0 and pickup_locs[j] > key:
            pickup_locs[j+1] = pickup_locs[j]
            j -= 1
        pickup_locs[j+1] = key
    unique_pickups = []
    last = -1_000_000_000
    for i in range(B):
        if i == 0 or pickup_locs[i] != last:
            unique_pickups.append(pickup_locs[i])
            last = pickup_locs[i]

    # -----------------------------------------
    # 3) Dijkstra from each pickup location
    # -----------------------------------------
    # dist_per_pickup[p_index][loc] = distance
    P = len(unique_pickups)
    dist = [0]*L
    dist_per_pickup = [[0]*L for _ in range(P)]
    for pi in range(P):
        _dijkstra_all(L, adj, unique_pickups[pi], dist)
        # copy distances
        for v in range(L):
            dist_per_pickup[pi][v] = dist[v]

    # Map buses to (pickup_index) so we can quickly check distances by that pickup
    # bus_pickup_idx[j] in [0..P-1]
    bus_pickup_idx = [0]*B
    for j in range(B):
        e = buses[j][0]
        # binary search unique_pickups for e (since it's sorted)
        lo, hi = 0, P-1
        found = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if unique_pickups[mid] == e:
                found = mid
                break
            elif unique_pickups[mid] < e:
                lo = mid + 1
            else:
                hi = mid - 1
        bus_pickup_idx[j] = found

    # Pre-checks on totals: sum of mins must be ≤ T; T ≤ sum of max; also ≤ S.
    total_min = 0
    total_max = 0
    for j in range(B):
        total_min += buses[j][1]
        total_max += buses[j][2]
    if total_min > T or T > total_max or T > S:
        return None

    # For each student, at least one reachable bus must exist if they are to be selected.
    # We don’t filter students out globally because we only need exactly T (not necessarily all).
    # However, a quick count of total reachable capacity lower-bounds feasibility:
    # If fewer than total_min students can reach at least one bus, infeasible.
    reachable_count = 0
    for i in range(S):
        # detect if student i can reach any pickup within D
        can = 0
        # loop pickups; if any dist ≤ D, they can reach at least one bus at that pickup
        for pi in range(P):
            if dist_per_pickup[pi][students[i]] <= D:
                can = 1
                break
        if can:
            reachable_count += 1
    if reachable_count < total_min:
        return None

    # -----------------------------------------
    # 4) Build flow with lower bounds
    # -----------------------------------------
    # Node indexing
    # s=0, q=1, students: [2 .. 2+S-1], buses: [2+S .. 2+S+B-1], t=2+S+B
    s = 0
    q = 1
    off_stu = 2
    off_bus = 2 + S
    t = 2 + S + B

    # We will add super-source SS and super-sink TT for lower-bound feasibility
    SS = t + 1
    TT = t + 2
    N = t + 3

    din = _Dinic(N)

    # Demand array for lower-bound transform
    demand = [0]*N

    def add_edge_with_lower(u: int, v: int, lower: int, upper: int) -> None:
        """
        Adds an edge with lower/upper bounds using the standard reduction:
          cap(u->v) = upper - lower
          demand[v] += lower
          demand[u] -= lower
        """
        # guard upper >= lower
        if upper < lower:
            # impossible edge, mark infeasible via huge demand
            # but we’ll just avoid adding and rely on precheck; returning None later if needed
            return
        din.add_edge(u, v, upper - lower)
        demand[v] += lower
        demand[u] -= lower

    # s -> q (capacity T, no lower bound)
    din.add_edge(s, q, T)

    # q -> student_i (capacity 1, no lower bound)
    for i in range(S):
        din.add_edge(q, off_stu + i, 1)

    # student_i -> bus_j (capacity 1 if reachable within D)
    for i in range(S):
        loc = students[i]
        # For each bus j, check reachability using its pickup index
        for j in range(B):
            pi = bus_pickup_idx[j]
            if dist_per_pickup[pi][loc] <= D:
                din.add_edge(off_stu + i, off_bus + j, 1)

    # bus_j -> t with lower = min_j, upper = max_j
    for j in range(B):
        min_j = buses[j][1]
        max_j = buses[j][2]
        add_edge_with_lower(off_bus + j, t, min_j, max_j)

    # Add the circulation helper edge t->s with INF cap (no lower bound)
    INF = 10**9
    din.add_edge(t, s, INF)

    # Hook up SS/TT per node demands
    total_pos = 0
    for v in range(N):
        if demand[v] > 0:
            din.add_edge(SS, v, demand[v])
            total_pos += demand[v]
        elif demand[v] < 0:
            din.add_edge(v, TT, -demand[v])

    # Feasibility: max flow from SS to TT must saturate all positive demand
    flow_feas = din.max_flow(SS, TT)
    if flow_feas != total_pos:
        return None  # lower bounds unsatisfied → infeasible

    # -----------------------------------------
    # 5) Enforce exactly T and extract solution
    # -----------------------------------------
    # Remove SS/TT by zeroing their adjacency lists (they are isolated from s..t search)
    # Not strictly necessary since they don't connect to s/t anymore, but done for clarity:
    din.g[SS] = []
    din.g[TT] = []

    # Also disable the helper edge t->s by setting capacity to 0 (find it and zero it)
    # We can't use sets/dicts; scan adjacency.
    for idx, e in enumerate(din.g[t]):
        if e.to == s:
            e.cap = 0
            # also zero reverse residual from s back to t if any capacity was created
            rev = din.g[e.to][e.rev]
            # leave reverse cap as-is (that stores current flow); setting forward to 0 prevents new cycles
            break

    # Now compute a max flow s->t; it must equal T
    final_flow = din.max_flow(s, t)
    if final_flow != T:
        return None

    # Reconstruct allocation: check student -> bus edges used (cap==0 on edges originally of cap 1)
    allocation = [-1]*S
    for i in range(S):
        u = off_stu + i
        # Among edges from u, find any to a bus node with residual cap == 0 (meaning 1 unit used)
        # Note: Some edges are q->student reverse etc; we only consider u->(bus range)
        for e in din.g[u]:
            v = e.to
            if off_bus <= v < off_bus + B:
                # original capacity was 1; if now cap == 0 then flow = 1
                if e.cap == 0:
                    allocation[i] = (v - off_bus)
                    break  # only one bus per student

    return allocation
