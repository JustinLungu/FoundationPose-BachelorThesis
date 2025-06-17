#!/usr/bin/env python3
"""
view_raw_overlap.py

Quick check: load two .ply meshes (the Linemod GT and the GenAI‐produced PLY)
and display them in object‐space side‐by‐side (green vs. red).  No ICP, no
camera‐pose, no scaling—just raw‐mesh overlay.  If they already occupy the
same object‐space, you will see them exactly overlapping.  Otherwise, you’ll
see a mismatch immediately.

Usage:
    cd Pose_Evaluation
    python view_raw_overlap.py
"""

import os
import sys
import open3d as o3d

# ----------------------------------------------------------------------
#  CONFIGURATION: replace these paths if your files live elsewhere
# ----------------------------------------------------------------------
GT_PLY = "3d_models/original_models/obj_09.ply"
GENAI_PLY = "3d_models/dreamfusion/obj_09.ply"
# ----------------------------------------------------------------------

def load_and_color(ply_path: str, color: list) -> o3d.geometry.TriangleMesh:
    """
    Load a PLY (or other mesh) via Open3D, compute vertex normals if missing,
    paint it the given RGB triplet, then return it.
    """
    if not os.path.exists(ply_path):
        print(f"[ERROR] Cannot find PLY at: {ply_path}")
        sys.exit(1)

    mesh = o3d.io.read_triangle_mesh(ply_path)
    if mesh.is_empty():
        print(f"[ERROR] Loaded mesh is empty: {ply_path}")
        sys.exit(1)

    # Make sure normals exist (so lighting/shading looks correct)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    mesh.paint_uniform_color(color)
    return mesh


def main():
    # Load GT mesh (paint it green)
    mesh_gt = load_and_color(GT_PLY, [0.0, 1.0, 0.0])

    # Load GenAI mesh (paint it red)
    mesh_gen = load_and_color(GENAI_PLY, [1.0, 0.0, 0.0])

    # (Optional) Center both meshes at the origin if you suspect they have different centers.
    # Uncomment these lines if you want them visually centered rather than left in “raw” coordinates.
    #
    # center_gt = mesh_gt.get_center()
    # center_gen = mesh_gen.get_center()
    # mesh_gt.translate(-center_gt)
    # mesh_gen.translate(-center_gen)

    # Display the two meshes together
    o3d.visualization.draw_geometries(
        [mesh_gt, mesh_gen],
        window_name="RAW MESH OVERLAY: GT (green) vs GenAI (red)",
        width=800,
        height=600,
        mesh_show_back_face=False
    )


if __name__ == "__main__":
    main()
