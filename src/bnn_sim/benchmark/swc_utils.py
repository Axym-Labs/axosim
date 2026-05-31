"""SWC morphology parsing and branch routing for M3."""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any


def load_swc(swc_path: str | Path) -> np.ndarray:
    """Load SWC file into array.
    
    Returns array of shape (n_nodes, 7) with columns:
    [id, type, x, y, z, radius, parent_id]
    """
    data = []
    with open(swc_path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= 7:
                data.append([float(p) for p in parts[:7]])
    return np.array(data)


def get_node_children(swc_data: np.ndarray) -> dict[int, list[int]]:
    """Build children mapping from parent IDs."""
    children = {}
    for node in swc_data:
        node_id = int(node[0])
        parent_id = int(node[6])
        if parent_id not in children:
            children[parent_id] = []
        children[parent_id].append(node_id)
    return children


def extract_branches(swc_data: np.ndarray, target_types: list[int] | None = None) -> list[list[int]]:
    """Extract unbranched segments (branches) from SWC.
    
    A branch is a path from a branch point (or soma) to a terminal or next branch point.
    
    Args:
        swc_data: SWC array
        target_types: Node types to include (1=soma, 2=axon, 3=basal, 4=apical)
    
    Returns:
        List of branches, each is a list of node IDs in order.
    """
    if target_types is None:
        target_types = [3, 4]  # basal and apical dendrites only
    
    # Filter to target types
    type_mask = np.isin(swc_data[:, 1], target_types)
    valid_nodes = set(swc_data[type_mask, 0].astype(int))
    
    children = get_node_children(swc_data)
    
    # Find branch points (nodes with >1 child) and terminals (nodes with 0 children)
    branch_points = set()
    terminals = set()
    for node_id in valid_nodes:
        node_children = [c for c in children.get(node_id, []) if c in valid_nodes]
        if len(node_children) > 1:
            branch_points.add(node_id)
        elif len(node_children) == 0:
            terminals.add(node_id)
    
    # Trace branches from branch points to next branch point or terminal
    branches = []
    visited_segments = set()
    
    def trace_from(start_id: int, prev_id: int | None = None) -> list[int] | None:
        """Trace a branch from start_id until branch point or terminal."""
        branch = [start_id]
        current = start_id
        
        while True:
            node_children = [c for c in children.get(current, []) 
                           if c in valid_nodes and c != prev_id]
            
            if len(node_children) == 0:
                # Terminal
                return branch
            elif len(node_children) > 1:
                # Branch point - stop here
                return branch
            else:
                # Continue along single path
                prev_id = current
                current = node_children[0]
                branch.append(current)
    
    # Start from soma (type 1) or branch points
    soma_nodes = swc_data[swc_data[:, 1] == 1, 0].astype(int)
    start_nodes = set(soma_nodes) | branch_points
    
    for start in start_nodes:
        if start not in valid_nodes:
            continue
        for child in children.get(start, []):
            if child not in valid_nodes:
                continue
            segment = tuple([start, child])
            if segment in visited_segments:
                continue
            
            branch = trace_from(child, start)
            if branch and len(branch) >= 2:
                full_branch = [start] + branch
                branches.append(full_branch)
                # Mark all segments as visited
                for i in range(len(full_branch) - 1):
                    visited_segments.add((full_branch[i], full_branch[i+1]))
    
    return branches


def select_branches(swc_data: np.ndarray, n_branches: int = 45, seed: int = 42) -> list[list[int]]:
    """Select n_branches from morphology, preferring longer branches.
    
    Strategy:
    1. Extract all basal and apical branches
    2. Sort by length (path distance)
    3. Select top n_branches
    4. If insufficient, include shorter branches
    """
    rng = np.random.default_rng(seed)
    
    # Extract all branches
    all_branches = extract_branches(swc_data, target_types=[3, 4])
    
    if len(all_branches) == 0:
        # Fallback: include axon or soma branches
        all_branches = extract_branches(swc_data, target_types=[2, 3, 4])
    
    # Compute branch lengths
    def branch_length(branch: list[int]) -> float:
        total = 0.0
        for i in range(len(branch) - 1):
            idx1 = np.where(swc_data[:, 0] == branch[i])[0][0]
            idx2 = np.where(swc_data[:, 0] == branch[i+1])[0][0]
            p1 = swc_data[idx1, 2:5]
            p2 = swc_data[idx2, 2:5]
            total += np.linalg.norm(p2 - p1)
        return total
    
    branch_lengths = [(b, branch_length(b)) for b in all_branches]
    branch_lengths.sort(key=lambda x: x[1], reverse=True)
    
    # Select top n_branches
    selected = [b for b, _ in branch_lengths[:n_branches]]
    
    if len(selected) < n_branches:
        print(f"Warning: Only {len(selected)} branches available, requested {n_branches}")
    
    return selected


def create_routing_from_branches(
    swc_data: np.ndarray,
    branches: list[list[int]],
    input_dim: int = 1278,
) -> np.ndarray:
    """Create branch routing matrix from SWC branches.
    
    For now, use uniform routing: distribute input_dim evenly across branches.
    In future, could use spatial proximity or morphological features.
    
    Returns:
        Routing matrix of shape (n_branches, input_dim)
    """
    n_branches = len(branches)
    branch_size = input_dim // n_branches
    
    routing = np.zeros((n_branches, input_dim))
    for b in range(n_branches):
        start = b * branch_size
        end = min(start + branch_size, input_dim)
        routing[b, start:end] = 1.0
    
    return routing


if __name__ == "__main__":
    # Test with downloaded L2/3 morphology
    swc_path = "/home/davwis/main/workspace/bnn_sim/data/morphologies/specimen_327962063.swc"
    swc_data = load_swc(swc_path)
    print(f"Loaded {len(swc_data)} nodes")
    
    branches = select_branches(swc_data, n_branches=45)
    print(f"Selected {len(branches)} branches")
    
    lengths = [len(b) for b in branches]
    print(f"Branch lengths: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths)/len(lengths):.1f}")
    
    routing = create_routing_from_branches(swc_data, branches)
    print(f"Routing matrix shape: {routing.shape}")
