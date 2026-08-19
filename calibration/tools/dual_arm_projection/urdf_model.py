from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

from .transforms import (
    homogeneous_from_rotation_translation,
    homogeneous_from_xyz_rpy,
    rotation_about_axis,
    transform_points,
)


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    axis: np.ndarray
    T_parent_child_at_zero: np.ndarray


@dataclass(frozen=True)
class RenderLink:
    side: str
    link_name: str
    link_id: int
    vertices_link_m: np.ndarray
    faces: np.ndarray
    source_face_count: int
    rendered_face_count: int
    mesh_path: Path
    visual_scale: np.ndarray
    T_link_visual: np.ndarray
    geometry_mode: str
    proxy_metadata: dict | None


@dataclass(frozen=True)
class ArmModel:
    side: str
    base_link: str
    render_links: tuple[RenderLink, ...]
    chain_joints: tuple[UrdfJoint, ...]
    excluded_links: tuple[str, ...]

    def forward_kinematics(self, joint_angles_rad: dict[str, float]) -> dict[str, np.ndarray]:
        T_base_link_by_name: dict[str, np.ndarray] = {
            self.base_link: np.eye(4, dtype=np.float64)
        }
        for urdf_joint in self.chain_joints:
            if urdf_joint.parent_link not in T_base_link_by_name:
                raise KeyError(f"missing FK parent link {urdf_joint.parent_link}")
            T_parent_child = urdf_joint.T_parent_child_at_zero.copy()
            if urdf_joint.joint_type in {"revolute", "continuous"}:
                joint_angle_rad = float(joint_angles_rad[urdf_joint.name])
                R_joint_motion = rotation_about_axis(urdf_joint.axis, joint_angle_rad)
                T_joint_motion = homogeneous_from_rotation_translation(
                    R_joint_motion, [0.0, 0.0, 0.0]
                )
                T_parent_child = T_parent_child @ T_joint_motion
            T_base_parent = T_base_link_by_name[urdf_joint.parent_link]
            T_base_link_by_name[urdf_joint.child_link] = T_base_parent @ T_parent_child
        return T_base_link_by_name


def _float_vector(text: str | None, length: int, default: float = 0.0) -> np.ndarray:
    if text is None:
        return np.full(length, default, dtype=np.float64)
    values = [float(value) for value in text.split()]
    if len(values) != length:
        raise ValueError(f"expected {length} values, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _origin_matrix(element: ET.Element | None) -> np.ndarray:
    if element is None:
        return np.eye(4, dtype=np.float64)
    return homogeneous_from_xyz_rpy(
        _float_vector(element.get("xyz"), 3).tolist(),
        _float_vector(element.get("rpy"), 3).tolist(),
    )


def _resolve_mesh_path(urdf_path: Path, mesh_filename: str) -> Path:
    mesh_name = Path(mesh_filename.replace("\\", "/")).name
    candidates = [
        urdf_path.parent / mesh_filename,
        urdf_path.parent.parent / "meshes" / mesh_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    matches = list(urdf_path.parent.parent.rglob(mesh_name))
    if len(matches) == 1:
        return matches[0].resolve()
    raise FileNotFoundError(f"cannot resolve mesh {mesh_filename!r} from {urdf_path}")


def _load_render_link(
    urdf_path: Path,
    link_element: ET.Element,
    side: str,
    link_id: int,
    max_faces_per_link: int,
    geometry_mode: str,
    capsule_radius_margin_m: float,
    capsule_sections: int,
    sphere_subdivisions: int,
    terminal_axis_end_link_m: np.ndarray | None,
    terminal_link_radius_m: float,
    terminal_distal_sphere_scale: float,
) -> RenderLink:
    visual_element = link_element.find("visual")
    if visual_element is None:
        raise ValueError(f"link {link_element.get('name')} has no visual element")
    mesh_element = visual_element.find("geometry/mesh")
    if mesh_element is None:
        raise ValueError(f"link {link_element.get('name')} has no visual mesh")
    mesh_path = _resolve_mesh_path(urdf_path, mesh_element.get("filename", ""))
    visual_scale = _float_vector(mesh_element.get("scale"), 3, default=1.0)
    T_link_visual = _origin_matrix(visual_element.find("origin"))

    triangle_mesh = trimesh.load_mesh(mesh_path, process=True)
    if isinstance(triangle_mesh, trimesh.Scene):
        triangle_mesh = triangle_mesh.to_geometry()
    source_face_count = int(len(triangle_mesh.faces))
    vertices_visual_m = np.asarray(triangle_mesh.vertices, dtype=np.float64) * visual_scale
    source_vertices_link_m = transform_points(T_link_visual, vertices_visual_m)
    proxy_metadata = None
    if geometry_mode == "capsule_proxy":
        triangle_mesh, proxy_metadata = _fit_capsule_proxy(
            source_vertices_link_m,
            radius_margin_m=capsule_radius_margin_m,
            cylinder_sections=capsule_sections,
            sphere_subdivisions=sphere_subdivisions,
            terminal_axis_end_link_m=terminal_axis_end_link_m,
            terminal_link_radius_m=terminal_link_radius_m,
            terminal_distal_sphere_scale=terminal_distal_sphere_scale,
        )
        vertices_link_m = np.asarray(triangle_mesh.vertices, dtype=np.float64)
    elif geometry_mode == "visual_mesh":
        if max_faces_per_link > 0 and source_face_count > max_faces_per_link:
            triangle_mesh = triangle_mesh.simplify_quadric_decimation(
                face_count=max_faces_per_link
            )
        vertices_visual_m = (
            np.asarray(triangle_mesh.vertices, dtype=np.float64) * visual_scale
        )
        vertices_link_m = transform_points(T_link_visual, vertices_visual_m)
    else:
        raise ValueError(f"unsupported geometry_mode: {geometry_mode}")
    faces = np.asarray(triangle_mesh.faces, dtype=np.int32)
    return RenderLink(
        side=side,
        link_name=str(link_element.get("name")),
        link_id=link_id,
        vertices_link_m=vertices_link_m,
        faces=faces,
        source_face_count=source_face_count,
        rendered_face_count=int(len(faces)),
        mesh_path=mesh_path,
        visual_scale=visual_scale,
        T_link_visual=T_link_visual,
        geometry_mode=geometry_mode,
        proxy_metadata=proxy_metadata,
    )


def _fit_capsule_proxy(
    vertices_link_m: np.ndarray,
    radius_margin_m: float,
    cylinder_sections: int,
    sphere_subdivisions: int,
    terminal_axis_end_link_m: np.ndarray | None = None,
    terminal_link_radius_m: float = 0.04,
    terminal_distal_sphere_scale: float = 0.25,
) -> tuple[trimesh.Trimesh, dict]:
    if len(vertices_link_m) < 3:
        raise ValueError("at least three vertices are required to fit a capsule")
    if terminal_axis_end_link_m is not None:
        return _make_terminal_link_proxy(
            terminal_axis_end_link_m,
            radius_m=terminal_link_radius_m,
            distal_sphere_scale=terminal_distal_sphere_scale,
            cylinder_sections=cylinder_sections,
            sphere_subdivisions=sphere_subdivisions,
        )
    center_m = vertices_link_m.mean(axis=0)
    centered = vertices_link_m - center_m
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axis_link = eigenvectors[:, int(np.argmax(eigenvalues))]
    axis_link = axis_link / np.linalg.norm(axis_link)

    axial_coordinates_m = centered @ axis_link
    radial_vectors_m = centered - np.outer(axial_coordinates_m, axis_link)
    radial_distances_m = np.linalg.norm(radial_vectors_m, axis=1)
    axial_min_m = float(np.quantile(axial_coordinates_m, 0.005))
    axial_max_m = float(np.quantile(axial_coordinates_m, 0.995))
    radius_m = max(
        0.004,
        float(np.quantile(radial_distances_m, 0.98)) + radius_margin_m,
    )
    axial_span_m = axial_max_m - axial_min_m
    axial_mid_m = 0.5 * (axial_min_m + axial_max_m)

    parts: list[trimesh.Trimesh] = []
    if axial_span_m <= 2.0 * radius_m:
        radius_m = max(radius_m, 0.5 * axial_span_m + radius_margin_m)
        sphere_center_m = center_m + axis_link * axial_mid_m
        sphere = trimesh.creation.icosphere(
            subdivisions=sphere_subdivisions, radius=radius_m
        )
        sphere.apply_translation(sphere_center_m)
        parts.append(sphere)
        endpoint_centers_m = [sphere_center_m, sphere_center_m]
        cylinder_length_m = 0.0
    else:
        start_center_m = center_m + axis_link * (axial_min_m + radius_m)
        end_center_m = center_m + axis_link * (axial_max_m - radius_m)
        cylinder_length_m = float(np.linalg.norm(end_center_m - start_center_m))
        cylinder = trimesh.creation.cylinder(
            radius=radius_m,
            height=cylinder_length_m,
            sections=cylinder_sections,
        )
        T_link_cylinder = trimesh.geometry.align_vectors(
            [0.0, 0.0, 1.0], axis_link
        )
        T_link_cylinder[:3, 3] = 0.5 * (start_center_m + end_center_m)
        cylinder.apply_transform(T_link_cylinder)
        parts.append(cylinder)
        for sphere_center_m in (start_center_m, end_center_m):
            sphere = trimesh.creation.icosphere(
                subdivisions=sphere_subdivisions, radius=radius_m
            )
            sphere.apply_translation(sphere_center_m)
            parts.append(sphere)
        endpoint_centers_m = [start_center_m, end_center_m]

    capsule = trimesh.util.concatenate(parts)
    metadata = {
        "axis_link": axis_link.tolist(),
        "radius_m": radius_m,
        "cylinder_length_m": cylinder_length_m,
        "endpoint_centers_link_m": [value.tolist() for value in endpoint_centers_m],
        "source_axial_span_m": axial_span_m,
        "radius_margin_m": radius_margin_m,
        "cylinder_sections": cylinder_sections,
        "sphere_subdivisions": sphere_subdivisions,
    }
    return capsule, metadata


def _make_terminal_link_proxy(
    axis_end_link_m: np.ndarray,
    radius_m: float,
    distal_sphere_scale: float,
    cylinder_sections: int,
    sphere_subdivisions: int,
) -> tuple[trimesh.Trimesh, dict]:
    axis_end_link_m = np.asarray(axis_end_link_m, dtype=np.float64)
    axis_length_m = float(np.linalg.norm(axis_end_link_m))
    if axis_length_m <= 0.0:
        raise ValueError("terminal axis must have non-zero length")
    if radius_m <= 0.0:
        raise ValueError("terminal link radius must be positive")
    if not 0.0 < distal_sphere_scale <= 1.0:
        raise ValueError("terminal distal sphere scale must be in (0, 1]")

    axis_link = axis_end_link_m / axis_length_m
    proximal_center_m = np.zeros(3, dtype=np.float64)
    distal_radius_m = radius_m * distal_sphere_scale
    distal_center_m = axis_end_link_m - axis_link * distal_radius_m
    cylinder_length_m = float(np.linalg.norm(distal_center_m - proximal_center_m))

    cylinder = trimesh.creation.cylinder(
        radius=radius_m,
        height=cylinder_length_m,
        sections=cylinder_sections,
    )
    T_link_cylinder = trimesh.geometry.align_vectors([0.0, 0.0, 1.0], axis_link)
    T_link_cylinder[:3, 3] = 0.5 * (proximal_center_m + distal_center_m)
    cylinder.apply_transform(T_link_cylinder)

    proximal_sphere = trimesh.creation.icosphere(
        subdivisions=sphere_subdivisions, radius=radius_m * 1.15
    )
    proximal_sphere.apply_translation(proximal_center_m)
    distal_sphere = trimesh.creation.icosphere(
        subdivisions=sphere_subdivisions, radius=distal_radius_m
    )
    distal_sphere.apply_translation(distal_center_m)
    proxy = trimesh.util.concatenate([cylinder, proximal_sphere, distal_sphere])
    metadata = {
        "proxy_kind": "terminal_tapered_capsule",
        "axis_link": axis_link.tolist(),
        "radius_m": radius_m,
        "distal_radius_m": distal_radius_m,
        "distal_sphere_scale": distal_sphere_scale,
        "cylinder_length_m": cylinder_length_m,
        "endpoint_centers_link_m": [
            proximal_center_m.tolist(),
            distal_center_m.tolist(),
        ],
        "tcp_end_link_m": axis_end_link_m.tolist(),
        "cylinder_sections": cylinder_sections,
        "sphere_subdivisions": sphere_subdivisions,
    }
    return proxy, metadata


def load_arm_model(
    urdf_path: Path,
    side: str,
    max_faces_per_link: int,
    geometry_mode: str = "visual_mesh",
    capsule_radius_margin_m: float = 0.002,
    capsule_sections: int = 16,
    sphere_subdivisions: int = 2,
    terminal_link_radius_m: float = 0.04,
    terminal_distal_sphere_scale: float = 0.25,
) -> ArmModel:
    side = side.upper()
    if side not in {"L", "R"}:
        raise ValueError("side must be L or R")
    root = ET.parse(urdf_path).getroot()
    link_elements = {element.get("name"): element for element in root.findall("link")}
    joint_elements = {element.get("name"): element for element in root.findall("joint")}
    base_link = f"Base_{side}"
    included_links = [base_link] + [f"Link{index}_{side}" for index in range(1, 8)]
    excluded_links = (f"TCP_Link_{side}",)

    render_links = []
    id_offset = 0 if side == "L" else 100
    for local_id, link_name in enumerate(included_links, start=1):
        terminal_axis_end_link_m = None
        if link_name == f"Link7_{side}":
            terminal_joint = joint_elements[f"JointTCP_{side}"]
            terminal_axis_end_link_m = _origin_matrix(
                terminal_joint.find("origin")
            )[:3, 3]
        render_links.append(
            _load_render_link(
                urdf_path,
                link_elements[link_name],
                side,
                id_offset + local_id,
                max_faces_per_link,
                geometry_mode,
                capsule_radius_margin_m,
                capsule_sections,
                sphere_subdivisions,
                terminal_axis_end_link_m,
                terminal_link_radius_m,
                terminal_distal_sphere_scale,
            )
        )

    chain_joints = []
    for index in range(1, 8):
        joint_name = f"Joint{index}_{side}"
        element = joint_elements[joint_name]
        axis_element = element.find("axis")
        chain_joints.append(
            UrdfJoint(
                name=joint_name,
                joint_type=str(element.get("type")),
                parent_link=str(element.find("parent").get("link")),
                child_link=str(element.find("child").get("link")),
                axis=_float_vector(
                    axis_element.get("xyz") if axis_element is not None else None, 3
                ),
                T_parent_child_at_zero=_origin_matrix(element.find("origin")),
            )
        )
    return ArmModel(
        side=side,
        base_link=base_link,
        render_links=tuple(render_links),
        chain_joints=tuple(chain_joints),
        excluded_links=excluded_links,
    )
