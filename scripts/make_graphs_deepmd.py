#!/usr/bin/env python3
"""
Generate RadiusGraph files from DeePMD-kit data format.
This script converts deepmd data systems to pre-computed graph.pt files,
which can be used for faster training with graph caching.

Usage:
    python make_graphs_deepmd.py --system /path/to/deepmd/system --output ./graphs --type-map H O C

Author: deepmd-matris
"""

import os
import sys
import argparse
import json
from typing import Optional
import torch
import numpy as np

from deepmd.utils.data import DeepmdData
from matris.graph import GraphConverter
from pymatgen.core import Structure, Lattice


def frame_to_structure(frame: dict, type_map: list[str]) -> Structure:
    """Convert deepmd frame data to pymatgen Structure.
    
    Args:
        frame: Dict containing 'coord', 'atype', 'box' from DeepmdData
        type_map: List of element symbols
    
    Returns:
        pymatgen Structure object
    """
    coord = frame["coord"]  # (natoms, 3)
    atype = frame["atype"]  # (natoms,)
    box = frame["box"]      # (9,) or (3, 3)
    
    # Handle box format
    if box.size == 9:
        lattice_vec = box.reshape(3, 3)
    else:
        lattice_vec = box
    
    # Convert atom types to species
    species = [type_map[int(t)] for t in atype]
    
    # Create structure
    lattice = Lattice(lattice_vec)
    structure = Structure(
        lattice=lattice,
        species=species,
        coords=coord,
        coords_are_cartesian=True
    )
    
    return structure


def make_one_graph(
    frame_id: int,
    data_system: DeepmdData,
    graph_converter: GraphConverter,
    type_map: list[str],
    output_dir: str,
) -> dict | bool:
    """Convert one frame to RadiusGraph and save it.
    
    Args:
        frame_id: Frame index in the data system
        data_system: DeepmdData object
        graph_converter: GraphConverter instance
        type_map: List of element symbols
        output_dir: Directory to save the graph
    
    Returns:
        Dict of labels if successful, False otherwise
    """
    try:
        # Get frame data
        frame = data_system.get_item_torch(frame_id)
        
        # Convert to numpy for pymatgen
        coord = frame["coord"].cpu().numpy()
        atype = frame["atype"].cpu().numpy()
        box = frame["box"].cpu().numpy()
        
        # Convert to structure
        structure = frame_to_structure(
            {"coord": coord, "atype": atype, "box": box},
            type_map
        )
        
        # Convert to graph
        graph = graph_converter(structure, graph_id=str(frame_id), mp_id=frame_id)
        
        # Save graph
        graph_path = os.path.join(output_dir, f"{frame_id}.pt")
        torch.save(graph, graph_path)
        
        # Prepare labels
        labels = {}
        
        # Energy (per atom)
        if "energy" in frame:
            energy = frame["energy"].item()
            natoms = len(structure)
            labels["energy_per_atom"] = energy / natoms
            labels["energy"] = energy
        
        # Force
        if "force" in frame:
            labels["force"] = frame["force"].cpu().numpy().tolist()
        
        # Stress
        if "virial" in frame:
            virial = frame["virial"].cpu().numpy()
            # Convert virial to stress (depends on convention)
            labels["virial"] = virial.tolist()
        
        labels["fid"] = frame_id
        labels["natoms"] = len(structure)
        
        return labels
        
    except Exception as e:
        print(f"Failed to convert frame {frame_id}: {type(e).__name__}: {e}")
        return False


def make_graphs_from_deepmd(
    system_path: str,
    output_dir: str,
    type_map: list[str],
    cutoff: float = 6.0,
    line_graph_cutoff: float = 4.0,
    on_isolated_atoms: str = "warn",
) -> None:
    """Generate graph.pt files from a deepmd data system.
    
    Args:
        system_path: Path to deepmd data system
        output_dir: Directory to save graphs and labels
        type_map: List of element symbols
        cutoff: Cutoff radius for atom graph (default: 6.0)
        line_graph_cutoff: Cutoff for three-body interactions (default: 4.0)
        on_isolated_atoms: How to handle isolated atoms ("ignore", "warn", "error")
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load deepmd data
    print(f"Loading deepmd data from: {system_path}")
    data_system = DeepmdData(sys_path=system_path, type_map=type_map)
    nframes = data_system.nframes
    print(f"Total frames: {nframes}")
    
    # Create graph converter
    graph_converter = GraphConverter(
        atom_graph_cutoff=cutoff,
        line_graph_cutoff=line_graph_cutoff,
        verbose=False
    )
    
    # Convert all frames
    labels = {}
    failed_frames = []
    
    print("Converting frames to graphs...")
    for frame_id in range(nframes):
        result = make_one_graph(
            frame_id, data_system, graph_converter, type_map, output_dir
        )
        
        if result is not False:
            labels[frame_id] = result
        else:
            failed_frames.append(frame_id)
        
        if (frame_id + 1) % 100 == 0:
            print(f"Processed {frame_id + 1}/{nframes} frames")
    
    # Save labels and failed frames
    labels_path = os.path.join(output_dir, "labels.json")
    with open(labels_path, "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved labels to: {labels_path}")
    
    if failed_frames:
        failed_path = os.path.join(output_dir, "failed_frames.json")
        with open(failed_path, "w") as f:
            json.dump(failed_frames, f, indent=2)
        print(f"Failed frames ({len(failed_frames)}): {failed_path}")
    
    success_rate = (nframes - len(failed_frames)) / nframes * 100
    print(f"\nConversion complete!")
    print(f"Success: {nframes - len(failed_frames)}/{nframes} ({success_rate:.1f}%)")
    print(f"Failed: {len(failed_frames)}/{nframes}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate RadiusGraph files from DeePMD-kit data"
    )
    
    # Input methods (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--system", "-s",
        type=str,
        nargs="+",
        help="Path(s) to deepmd data system directory"
    )
    input_group.add_argument(
        "--input-json", "-i",
        type=str,
        help="Path to deepmd training input.json file"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for graph.pt files"
    )
    parser.add_argument(
        "--type-map", "-t",
        type=str,
        nargs="+",
        help="Type map (element symbols), e.g., H O C (required if not using --input-json)"
    )
    parser.add_argument(
        "--cutoff", "-c",
        type=float,
        default=6.0,
        help="Cutoff radius for atom graph (default: 6.0 Å)"
    )
    parser.add_argument(
        "--line-graph-cutoff", "-l",
        type=float,
        default=4.0,
        help="Cutoff for three-body interactions (default: 4.0 Å)"
    )
    parser.add_argument(
        "--on-isolated-atoms",
        type=str,
        default="warn",
        choices=["ignore", "warn", "error"],
        help="How to handle isolated atoms (default: warn)"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge all systems into a single output directory with global frame IDs"
    )
    
    args = parser.parse_args()
    
    # Get systems and type_map
    if args.input_json:
        with open(args.input_json, 'r') as f:
            config = json.load(f)
        systems = config["training"]["training_data"]["systems"]
        type_map = config["model"]["type_map"]
        print(f"Loaded {len(systems)} systems from {args.input_json}")
    else:
        systems = args.system
        type_map = args.type_map
        if not type_map:
            parser.error("--type-map is required when not using --input-json")
    
    # Process systems
    if args.merge:
        # Merge all systems into one output directory
        print(f"Merging {len(systems)} systems into {args.output}")
        global_frame_id = 0
        all_labels = {}
        all_failed = []
        
        os.makedirs(args.output, exist_ok=True)
        graph_converter = GraphConverter(
            atom_graph_cutoff=args.cutoff,
            line_graph_cutoff=args.line_graph_cutoff,
            verbose=False
        )
        
        for sys_idx, system_path in enumerate(systems):
            print(f"\n[{sys_idx+1}/{len(systems)}] Processing: {system_path}")
            try:
                data_system = DeepmdData(sys_path=system_path, type_map=type_map)
                nframes = data_system.nframes
                
                for local_fid in range(nframes):
                    result = make_one_graph(
                        local_fid, data_system, graph_converter, type_map, args.output
                    )
                    
                    # Rename to global frame ID
                    if result is not False:
                        old_path = os.path.join(args.output, f"{local_fid}.pt")
                        new_path = os.path.join(args.output, f"{global_frame_id}.pt")
                        os.rename(old_path, new_path)
                        
                        result["fid"] = global_frame_id
                        result["system"] = system_path
                        all_labels[global_frame_id] = result
                        global_frame_id += 1
                    else:
                        all_failed.append({"system": system_path, "local_fid": local_fid})
                
            except Exception as e:
                print(f"Error processing system {system_path}: {e}")
                continue
        
        # Save merged labels
        with open(os.path.join(args.output, "labels.json"), "w") as f:
            json.dump(all_labels, f, indent=2)
        
        if all_failed:
            with open(os.path.join(args.output, "failed_frames.json"), "w") as f:
                json.dump(all_failed, f, indent=2)
        
        print(f"\n✓ Total: {global_frame_id} graphs generated")
        print(f"✗ Failed: {len(all_failed)}")
        
    else:
        # Process each system separately with system_{sid} naming
        system_mapping = {}
        
        for sys_idx, system_path in enumerate(systems):
            output_subdir = os.path.join(args.output, f"system_{sys_idx}")
            system_mapping[sys_idx] = system_path
            
            print(f"\n[{sys_idx+1}/{len(systems)}] Processing: {system_path}")
            print(f"System ID: {sys_idx} -> {output_subdir}")
            
            try:
                make_graphs_from_deepmd(
                    system_path=system_path,
                    output_dir=output_subdir,
                    type_map=type_map,
                    cutoff=args.cutoff,
                    line_graph_cutoff=args.line_graph_cutoff,
                    on_isolated_atoms=args.on_isolated_atoms,
                )
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        # Save system mapping
        mapping_path = os.path.join(args.output, "system_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(system_mapping, f, indent=2)
        print(f"\nSystem mapping saved to: {mapping_path}")


if __name__ == "__main__":
    main()

