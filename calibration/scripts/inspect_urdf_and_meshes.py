from __future__ import annotations

import argparse
import json
import math
import struct
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def parse_vector(text: str | None, length: int, default: list[float]) -> list[float]:
    if text is None:
        return default.copy()
    values = [float(value) for value in text.split()]
    if len(values) != length:
        raise ValueError(f"Expected {length} values, got {text!r}")
    return values


def parse_origin(element: ET.Element | None) -> dict:
    if element is None:
        return {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]}
    return {
        "xyz_m": parse_vector(element.get("xyz"), 3, [0.0, 0.0, 0.0]),
        "rpy_rad": parse_vector(element.get("rpy"), 3, [0.0, 0.0, 0.0]),
    }


def resolve_mesh_path(urdf_path: Path, filename: str) -> Path:
    if filename.startswith("package://"):
        package_relative = filename[len("package://") :]
        parts = package_relative.split("/", 1)
        if len(parts) != 2:
            return urdf_path.parent.parent / package_relative
        return urdf_path.parent.parent / parts[1]
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    return urdf_path.parent / candidate


def inspect_binary_stl(path: Path) -> dict | None:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        header = stream.read(80)
        triangle_count_bytes = stream.read(4)
        if len(triangle_count_bytes) != 4:
            return None
        triangle_count = struct.unpack("<I", triangle_count_bytes)[0]
        expected_size = 84 + triangle_count * 50
        if expected_size != file_size:
            return None

        minimum = [math.inf, math.inf, math.inf]
        maximum = [-math.inf, -math.inf, -math.inf]
        degenerate_triangles = 0
        for _ in range(triangle_count):
            record = stream.read(50)
            if len(record) != 50:
                raise ValueError(f"Truncated binary STL: {path}")
            values = struct.unpack("<12fH", record)
            vertices = [values[3:6], values[6:9], values[9:12]]
            for vertex in vertices:
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], float(vertex[axis]))
                    maximum[axis] = max(maximum[axis], float(vertex[axis]))
            a, b, c = vertices
            ab = [b[i] - a[i] for i in range(3)]
            ac = [c[i] - a[i] for i in range(3)]
            cross = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ]
            if sum(value * value for value in cross) < 1e-20:
                degenerate_triangles += 1
    return {
        "stl_encoding": "binary",
        "header_ascii": header.decode("ascii", errors="replace").rstrip("\x00 "),
        "triangle_count": triangle_count,
        "degenerate_triangle_count": degenerate_triangles,
        "bbox_min_raw": minimum,
        "bbox_max_raw": maximum,
    }


def inspect_ascii_stl(path: Path) -> dict:
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    vertex_count = 0
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                vertex = [float(value) for value in parts[1:]]
                vertex_count += 1
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], vertex[axis])
                    maximum[axis] = max(maximum[axis], vertex[axis])
    if vertex_count == 0:
        raise ValueError(f"No STL vertices found: {path}")
    return {
        "stl_encoding": "ascii",
        "triangle_count": vertex_count // 3,
        "degenerate_triangle_count": None,
        "bbox_min_raw": minimum,
        "bbox_max_raw": maximum,
    }


def unit_assessment(max_extent: float) -> dict:
    if 0.001 <= max_extent <= 5.0:
        return {
            "likely_unit": "meter",
            "confidence": "high",
            "reason": "Raw mesh extent is within a plausible robot-link range in meters.",
        }
    if 10.0 <= max_extent <= 5000.0:
        return {
            "likely_unit": "millimeter",
            "confidence": "medium",
            "reason": "Raw mesh extent is too large for meters and plausible for millimeters.",
        }
    return {
        "likely_unit": "uncertain",
        "confidence": "low",
        "reason": "Extent is outside the normal heuristic ranges; manual confirmation is required.",
    }


def inspect_mesh_file(path: Path) -> dict:
    result = {
        "path": str(path.resolve()),
        "exists": path.is_file(),
        "extension": path.suffix.lower(),
    }
    if not path.is_file():
        return result
    result["file_size_bytes"] = path.stat().st_size
    if path.suffix.lower() != ".stl":
        result["inspection_error"] = "Only STL geometry inspection is implemented."
        return result
    geometry = inspect_binary_stl(path)
    if geometry is None:
        geometry = inspect_ascii_stl(path)
    result.update(geometry)
    extents = [
        geometry["bbox_max_raw"][axis] - geometry["bbox_min_raw"][axis]
        for axis in range(3)
    ]
    result["bbox_extent_raw"] = extents
    result["max_extent_raw"] = max(extents)
    result["unit_assessment"] = unit_assessment(result["max_extent_raw"])
    result["size_assessment"] = (
        "suspiciously_small_placeholder_geometry"
        if result["max_extent_raw"] < 1e-4
        else "plausible_robot_geometry"
        if result["max_extent_raw"] <= 5.0
        else "suspiciously_large_geometry"
    )
    return result


def make_tree_text(root_link: str, joints: list[dict], orphan_links: list[str]) -> str:
    children: dict[str, list[dict]] = defaultdict(list)
    for joint in joints:
        children[joint["parent_link"]].append(joint)
    for entries in children.values():
        entries.sort(key=lambda item: item["name"])

    lines = [
        "URDF joint tree",
        "Direction convention:",
        "  origin defines T_parent_link_child_link at q=0.",
        "  xyz is meters; rpy and joint limits are radians.",
        "",
    ]

    def visit(link_name: str, prefix: str, active_path: set[str]) -> None:
        lines.append(f"{prefix}{link_name}")
        if link_name in active_path:
            lines.append(f"{prefix}  [cycle detected]")
            return
        next_path = active_path | {link_name}
        entries = children.get(link_name, [])
        for index, joint in enumerate(entries):
            is_last = index == len(entries) - 1
            branch = "`-- " if is_last else "|-- "
            continuation = "    " if is_last else "|   "
            origin = joint["origin"]
            lines.append(
                f"{prefix}{branch}{joint['name']} [{joint['type']}] "
                f"xyz_m={origin['xyz_m']} rpy_rad={origin['rpy_rad']}"
            )
            visit(joint["child_link"], prefix + continuation, next_path)

    visit(root_link, "", set())
    if orphan_links:
        lines.extend(["", "Links not reachable from the selected root:"])
        lines.extend(f"  {name}" for name in orphan_links)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    urdf_path = args.urdf.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    robot = ET.parse(urdf_path).getroot()

    links = []
    mesh_references = []
    for link_element in robot.findall("link"):
        link_name = link_element.get("name", "")
        visual_entries = []
        collision_entries = []
        for kind, target in (("visual", visual_entries), ("collision", collision_entries)):
            for geometry_parent in link_element.findall(kind):
                mesh_element = geometry_parent.find("geometry/mesh")
                if mesh_element is None:
                    continue
                filename = mesh_element.get("filename", "")
                resolved_path = resolve_mesh_path(urdf_path, filename)
                reference = {
                    "link_name": link_name,
                    "kind": kind,
                    "uri": filename,
                    "resolved_path": str(resolved_path.resolve()),
                    "origin": parse_origin(geometry_parent.find("origin")),
                    "scale": parse_vector(
                        mesh_element.get("scale"), 3, [1.0, 1.0, 1.0]
                    ),
                }
                target.append(reference)
                mesh_references.append(reference)
        links.append(
            {
                "name": link_name,
                "visual_meshes": visual_entries,
                "collision_meshes": collision_entries,
            }
        )

    joints = []
    for joint_element in robot.findall("joint"):
        limit_element = joint_element.find("limit")
        mimic_element = joint_element.find("mimic")
        axis_element = joint_element.find("axis")
        joint = {
            "name": joint_element.get("name", ""),
            "type": joint_element.get("type", ""),
            "parent_link": joint_element.find("parent").get("link", ""),
            "child_link": joint_element.find("child").get("link", ""),
            "axis": parse_vector(
                axis_element.get("xyz") if axis_element is not None else None,
                3,
                [1.0, 0.0, 0.0],
            ),
            "origin": parse_origin(joint_element.find("origin")),
            "limit": None,
            "mimic": None,
        }
        if limit_element is not None:
            joint["limit"] = {
                key: float(limit_element.get(key)) if limit_element.get(key) is not None else None
                for key in ("lower", "upper", "effort", "velocity")
            }
        if mimic_element is not None:
            joint["mimic"] = {
                "joint": mimic_element.get("joint"),
                "multiplier": float(mimic_element.get("multiplier", "1")),
                "offset": float(mimic_element.get("offset", "0")),
            }
        joints.append(joint)

    link_names = {link["name"] for link in links}
    child_links = {joint["child_link"] for joint in joints}
    root_links = sorted(link_names - child_links)
    if len(root_links) != 1:
        raise ValueError(f"Expected one URDF root link, found {root_links}")
    root_link = root_links[0]

    reachable = {root_link}
    changed = True
    while changed:
        changed = False
        for joint in joints:
            if joint["parent_link"] in reachable and joint["child_link"] not in reachable:
                reachable.add(joint["child_link"])
                changed = True
    orphan_links = sorted(link_names - reachable)

    left_links = sorted(name for name in link_names if name.endswith("_L"))
    right_links = sorted(name for name in link_names if name.endswith("_R"))
    fixed_joints = [joint["name"] for joint in joints if joint["type"] == "fixed"]
    mimic_joints = [joint["name"] for joint in joints if joint["mimic"] is not None]

    urdf_report = {
        "status": "pass" if not orphan_links else "warning",
        "urdf_path": str(urdf_path),
        "robot_name": robot.get("name"),
        "coordinate_convention": {
            "homogeneous_matrix_shape": [4, 4],
            "origin_direction": "T_parent_link_child_link at q=0",
            "length_unit": "meter",
            "angle_unit": "radian",
        },
        "root_link": root_link,
        "root_link_candidates": root_links,
        "link_count": len(links),
        "joint_count": len(joints),
        "links": links,
        "joints": joints,
        "fixed_joint_count": len(fixed_joints),
        "fixed_joints": fixed_joints,
        "mimic_joint_count": len(mimic_joints),
        "mimic_joints": mimic_joints,
        "left_and_right_arms_in_same_urdf": bool(left_links and right_links),
        "left_arm_links": left_links,
        "right_arm_links": right_links,
        "orphan_links": orphan_links,
    }
    (output_dir / "urdf_inspection.json").write_text(
        json.dumps(urdf_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "urdf_joint_tree.txt").write_text(
        make_tree_text(root_link, joints, orphan_links), encoding="utf-8"
    )

    unique_mesh_paths = sorted({Path(item["resolved_path"]) for item in mesh_references})
    inspected_meshes = {str(path): inspect_mesh_file(path) for path in unique_mesh_paths}
    for mesh in inspected_meshes.values():
        references = [
            item for item in mesh_references if item["resolved_path"] == mesh["path"]
        ]
        mesh["references"] = references
        effective_extents = []
        if "bbox_extent_raw" in mesh:
            for reference in references:
                effective_extents.append(
                    [
                        mesh["bbox_extent_raw"][axis] * abs(reference["scale"][axis])
                        for axis in range(3)
                    ]
                )
        mesh["effective_bbox_extents_m_if_urdf_units_are_correct"] = effective_extents
        mesh["abnormal_scale_references"] = [
            reference
            for reference in references
            if any(value <= 0.0 or value < 1e-4 or value > 1e4 for value in reference["scale"])
        ]
        mesh["visual_origin_assessment"] = [
            {
                "link_name": reference["link_name"],
                "origin": reference["origin"],
                "assessment": (
                    "Mesh is used directly in the link frame."
                    if reference["origin"]
                    == {"xyz_m": [0.0, 0.0, 0.0], "rpy_rad": [0.0, 0.0, 0.0]}
                    else "Rendering must apply the explicit URDF visual origin."
                ),
            }
            for reference in references
            if reference["kind"] == "visual"
        ]

    missing_files = sorted(
        mesh["path"] for mesh in inspected_meshes.values() if not mesh["exists"]
    )
    links_missing_visual_mesh = sorted(
        link["name"] for link in links if not link["visual_meshes"]
    )
    links_missing_collision_mesh = sorted(
        link["name"] for link in links if not link["collision_meshes"]
    )
    visual_collision_comparison = []
    for link in links:
        visual_paths = {entry["resolved_path"] for entry in link["visual_meshes"]}
        collision_paths = {entry["resolved_path"] for entry in link["collision_meshes"]}
        visual_collision_comparison.append(
            {
                "link_name": link["name"],
                "has_visual_mesh": bool(visual_paths),
                "has_collision_mesh": bool(collision_paths),
                "same_mesh_files": bool(visual_paths) and visual_paths == collision_paths,
                "visual_mesh_paths": sorted(visual_paths),
                "collision_mesh_paths": sorted(collision_paths),
            }
        )

    unit_votes = defaultdict(int)
    for mesh in inspected_meshes.values():
        if "unit_assessment" in mesh:
            unit_votes[mesh["unit_assessment"]["likely_unit"]] += 1
    suspicious_size_meshes = sorted(
        mesh["path"]
        for mesh in inspected_meshes.values()
        if mesh.get("size_assessment") != "plausible_robot_geometry"
    )
    abnormal_scale_references = [
        reference
        for mesh in inspected_meshes.values()
        for reference in mesh.get("abnormal_scale_references", [])
    ]
    mesh_has_warning = bool(
        missing_files or suspicious_size_meshes or abnormal_scale_references
    )
    mesh_report = {
        "status": "warning" if mesh_has_warning else "pass",
        "urdf_path": str(urdf_path),
        "mesh_file_count": len(inspected_meshes),
        "mesh_reference_count": len(mesh_references),
        "format_counts": {
            extension: sum(
                mesh.get("extension") == extension for mesh in inspected_meshes.values()
            )
            for extension in sorted(
                {mesh.get("extension") for mesh in inspected_meshes.values()}
            )
        },
        "unit_vote_counts": dict(unit_votes),
        "unit_conclusion": (
            "URDF and mesh magnitudes are consistent with meters."
            if unit_votes["meter"] == len(inspected_meshes)
            else "Mixed or uncertain mesh units; inspect per-file assessments."
        ),
        "missing_mesh_files": missing_files,
        "suspicious_size_meshes": suspicious_size_meshes,
        "abnormal_scale_references": abnormal_scale_references,
        "links_missing_visual_mesh": links_missing_visual_mesh,
        "links_missing_collision_mesh": links_missing_collision_mesh,
        "visual_collision_comparison": visual_collision_comparison,
        "meshes": list(inspected_meshes.values()),
        "occlusion_scope_note": (
            "Triangle rasterization can resolve robot self-occlusion. Without scene depth, "
            "it cannot resolve occlusion by tables or manipulated objects."
        ),
    }
    (output_dir / "mesh_inspection.json").write_text(
        json.dumps(mesh_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "urdf_status": urdf_report["status"],
        "mesh_status": mesh_report["status"],
        "root_link": root_link,
        "link_count": len(links),
        "joint_count": len(joints),
        "fixed_joint_count": len(fixed_joints),
        "mimic_joint_count": len(mimic_joints),
        "mesh_file_count": len(inspected_meshes),
        "missing_mesh_file_count": len(missing_files),
        "both_arms_in_same_urdf": urdf_report["left_and_right_arms_in_same_urdf"],
        "outputs": {
            "urdf": str(output_dir / "urdf_inspection.json"),
            "tree": str(output_dir / "urdf_joint_tree.txt"),
            "mesh": str(output_dir / "mesh_inspection.json"),
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
