"""Straight-sided P2 tetrahedra and consistently integrated surface operators."""

from itertools import combinations

import numpy as np
from scipy import sparse

TET_EDGES = np.asarray(list(combinations(range(4), 2)))
TRI_EDGES = np.asarray(list(combinations(range(3), 2)))


def shape_values(barycentric):
    b = np.asarray(barycentric)
    edges = TET_EDGES if b.shape[-1] == 4 else TRI_EDGES
    return np.concatenate([b*(2*b-1), 4*b[..., edges[:, 0]]*b[..., edges[:, 1]]], axis=-1)


def shape_gradients(barycentric, gradients):
    """Barycentric values (..., 4), constant physical gradients (..., 4, 3)."""
    b, g = np.asarray(barycentric), np.asarray(gradients)
    a, c = TET_EDGES.T
    return np.concatenate([(4*b-1)[..., None]*g,
        4*(b[..., a, None]*g[..., c, :]+b[..., c, None]*g[..., a, :])], axis=-2)


def triangle_rule():
    # Positive, permutation-symmetric degree-four rule; weights sum to one.
    rows, weights = [], []
    for a, w in [(0.445948490915965, 0.223381589678011),
                 (0.091576213509771, 0.109951743655322)]:
        for i in range(3):
            row = np.full(3, a)
            row[i] = 1-2*a
            rows.append(row)
            weights.append(w)
    return np.asarray(rows), np.asarray(weights)


def strain_matrix(gradient):
    b = np.zeros((*gradient.shape[:-2], 6, 3*gradient.shape[-2]))
    for a in range(gradient.shape[-2]):
        gx, gy, gz = np.moveaxis(gradient[..., a, :], -1, 0)
        b[..., 0, 3*a], b[..., 1, 3*a+1], b[..., 2, 3*a+2] = gx, gy, gz
        b[..., 3, 3*a], b[..., 3, 3*a+1] = gy, gx
        b[..., 4, 3*a+1], b[..., 4, 3*a+2] = gz, gy
        b[..., 5, 3*a], b[..., 5, 3*a+2] = gz, gx
    return b


def quadratic_kinematics(inverse):
    a, b = (5+3*np.sqrt(5))/20, (5-np.sqrt(5))/20
    bary = np.full((4, 4), b)
    np.fill_diagonal(bary, a)
    gradient = shape_gradients(bary[None], inverse[:, None, 1:, :].swapaxes(-1, -2))
    return strain_matrix(gradient)


def surface_matrix(conn, local, ndof):
    dofs = (3*conn[..., None]+np.arange(3)).reshape(len(conn), -1)
    return sparse.coo_matrix((local.ravel(),
        (np.broadcast_to(dofs[:, :, None], local.shape).ravel(),
         np.broadcast_to(dofs[:, None, :], local.shape).ravel())), shape=(ndof, ndof)).tocsr()


def bottom_faces(mesh, body, level):
    tetra = mesh["tetra"][mesh["body"] == body, :4]
    faces = np.concatenate([tetra[:, face] for face in combinations(range(4), 3)])
    return faces[np.all(np.isclose(mesh["xyz"][faces, 2], level), axis=1)]


def upgrade_mesh(mesh, settings):
    """Add shared edge nodes within each body, retaining the original geometry."""
    result = dict(mesh)
    edges = np.unique(np.sort(mesh["tetra"][:, TET_EDGES].reshape(-1, 2), axis=1), axis=0)
    first = len(mesh["xyz"])
    edge_ids = {tuple(edge): first+i for i, edge in enumerate(edges)}
    result["xyz"] = np.concatenate([mesh["xyz"], mesh["xyz"][edges].mean(axis=1)])

    def upgrade(conn):
        pairs = TET_EDGES if conn.shape[1] == 4 else TRI_EDGES
        mid = np.asarray([[edge_ids[tuple(sorted(edge))] for edge in row[pairs]] for row in conn])
        return np.concatenate([conn, mid], axis=1)

    for key in ["tetra", "interface_upper", "interface_lower", "wall_cone", "wall_substrate"]:
        result[key] = upgrade(mesh[key])
    for key in ["fixed_dofs", "driven_dofs"]:
        old = set(mesh[key])
        extra = [3*(first+i)+d for i, (a, b) in enumerate(edges) for d in range(3)
                 if 3*a+d in old and 3*b+d in old]
        result[key] = np.sort(np.r_[mesh[key], extra]).astype(int)
    cfg = settings["geometry"]
    bary, weights = triangle_rule()
    shape = shape_values(bary)
    guide, trestle_faces, trestle_areas = [], [], []
    for body in (1, 2):
        faces = bottom_faces(mesh, body, cfg["bottom_levels_m"][-1])
        v = mesh["xyz"][faces]
        area = np.linalg.norm(np.cross(v[:, 1]-v[:, 0], v[:, 2]-v[:, 0]), axis=1)/2
        conn = upgrade(faces)
        xy = np.einsum("qa,fac->fqc", bary, v[..., :2])
        radius = np.sqrt(np.einsum("fq,q,f->", (xy*xy).sum(axis=-1), weights, area)/area.sum())
        rows = np.zeros((3, len(result["xyz"]), 3))
        mean = area[:, None]*np.einsum("q,qa->a", weights, shape)[None]/area.sum()
        np.add.at(rows[0, :, 0], conn.ravel(), mean.ravel())
        np.add.at(rows[1, :, 1], conn.ravel(), mean.ravel())
        rotation = np.einsum("fqc,q,qa,f->fac", xy, weights, shape, area)/(area.sum()*radius)
        np.add.at(rows[2, :, 0], conn.ravel(), -rotation[..., 1].ravel())
        np.add.at(rows[2, :, 1], conn.ravel(), rotation[..., 0].ravel())
        guide.extend(rows.reshape(3, -1))
        if body == 1:
            centre = v.mean(axis=1)
            selected = np.any(np.abs(centre[:, 0, None]-cfg["trestle_x_m"]) < cfg["trestle_width_m"]/2, axis=1)
            trestle_faces, trestle_areas = conn[selected], area[selected]
    result["lateral_guide_operator"] = np.asarray(guide)
    result["trestle_faces"], result["trestle_face_area"] = trestle_faces, trestle_areas
    result["trestle_area"] = np.zeros(len(result["xyz"]))
    result["bolt_nodes"] = np.unique(result["fixed_dofs"]//3)
    result["trestle_nodes"] = np.unique(trestle_faces)
    return result
