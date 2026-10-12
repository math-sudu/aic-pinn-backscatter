"""Conforming three-body mesh for the declared interface-history pilot.

Gmsh meshes the common plan view. Each triangle is extruded and split into
three tetrahedra with a consistent diagonal. Body-specific node copies retain
the two cohesive interfaces and the unbonded tapered cone sides.
"""

import json
from pathlib import Path

import gmsh
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def planar_mesh(config, quadrant=False):
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("General.NumThreads", 1)
        gmsh.model.add("lining_interface_pilot")
        occ = gmsh.model.occ
        half = config["across_flats_m"] / 2
        cut = half * (np.sqrt(2) - 1)
        vertices = [(half,cut),(cut,half),(-cut,half),(-half,cut),
                    (-half,-cut),(-cut,-half),(cut,-half),(half,-cut)]
        if quadrant:
            vertices = [(0,0),(half,0),(half,cut),(cut,half),(0,half)]
        points = [occ.addPoint(x, y, 0) for x, y in vertices]
        lines = [occ.addLine(points[i], points[(i+1) % len(points)]) for i in range(len(points))]
        surface = occ.addPlaneSurface([occ.addCurveLoop(lines)])
        disks = [occ.addDisk(0, 0, 0, config["cone_top_radius_m"],
                            config["cone_top_radius_m"])]
        if "jack_patch_radius_m" in config:
            # The extrusion scales cone radii from the interface to its base.
            radius = config["jack_patch_radius_m"] * config["cone_top_radius_m"] / config["cone_bottom_radius_m"]
            disks.append(occ.addDisk(0, 0, 0, radius, radius))
        disks.extend(occ.addDisk(x, y, 0, config["bolt_patch_radius_m"],
                                 config["bolt_patch_radius_m"])
                     for x, y in config["bolt_centres_m"] if not quadrant or (x > 0 and y > 0))
        tools = [(2, disk) for disk in disks]
        if quadrant:
            # Clip the cone and fit the trestle boundaries in the quadrant.
            central = 2 if "jack_patch_radius_m" in config else 1
            clipped = []
            for disk in tools[:central]:
                clipped.extend(occ.intersect([disk], occ.copy([(2, surface)]))[0])
            tools[:central] = clipped
            for x in config["trestle_x_m"]:
                if x > 0:
                    patch = occ.addRectangle(x-config["trestle_width_m"]/2, 0, 0,
                                             config["trestle_width_m"], half)
                    tools.extend(occ.intersect([(2, patch)], occ.copy([(2, surface)]))[0])
        occ.fragment([(2, surface)], tools)
        occ.synchronize()
        gmsh.model.addPhysicalGroup(2, [s for _, s in gmsh.model.getEntities(2)], name="plan")
        size = config["mesh_size_m"]
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MinimumCirclePoints", 12)
        if "readout_resolution" in config:
            local = config["readout_resolution"]
            fine, extent, transition = [local[k] for k in ("size_m", "extent_m", "transition_m")]
            # Distance to the two measured axis segments and cone transfer edge.
            axis = f"Max(Min(Abs(x),Abs(y)),Max(Abs(x),Abs(y))-{extent},0)"
            circle = f"Abs(Sqrt(x*x+y*y)-{config['cone_top_radius_m']})"
            distance = f"Min({axis},{circle})"
            field = gmsh.model.mesh.field.add("MathEval")
            gmsh.model.mesh.field.setString(field, "F", f"{fine}+{size-fine}*Min({distance}/{transition},1)")
            gmsh.model.mesh.field.setAsBackgroundMesh(field)
            gmsh.option.setNumber("Mesh.MeshSizeMin", fine)
            gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.model.mesh.generate(2)
        tags, coordinates, _ = gmsh.model.mesh.getNodes()
        order = np.argsort(tags)
        tags, xy = tags[order], coordinates.reshape(-1, 3)[order, :2]
        _, nodes = gmsh.model.mesh.getElementsByType(2)
        triangles = np.searchsorted(tags, nodes).reshape(-1, 3)
        triangles.sort(axis=1)
        return xy, triangles
    finally:
        gmsh.finalize()


def build_mesh(settings, reflection_symmetric=False):
    cfg = settings["geometry"]
    xy, triangles = planar_mesh(cfg, quadrant=reflection_symmetric)
    radius = cfg["cone_top_radius_m"]
    centres = xy[triangles].mean(axis=1)
    cone = np.linalg.norm(centres, axis=1) < radius
    points, tets, bodies, maps = [], [], [], []
    count = 0
    for body, selected, levels in [(0,np.ones(len(triangles),bool),cfg["top_levels_m"]),
                                   (1,~cone,cfg["bottom_levels_m"]),
                                   (2,cone,cfg["bottom_levels_m"])]:
        tri = triangles[selected]
        used = np.unique(tri)
        mapping = np.full((len(levels), len(xy)), -1, dtype=int)
        for layer, z in enumerate(levels):
            coords = xy[used].copy()
            if body:
                radial = np.linalg.norm(coords, axis=1)
                bottom_delta = (cfg["cone_bottom_radius_m"]-radius) * z / levels[-1]
                if body == 2:
                    factor = np.full(len(used), 1 + bottom_delta / radius)
                else:
                    # Fade the cone-wall displacement to zero at the octagon.
                    half = cfg["across_flats_m"] / 2
                    extent = np.maximum.reduce([np.abs(coords[:,0])/half,
                        np.abs(coords[:,1])/half,
                        (np.abs(coords[:,0])+np.abs(coords[:,1]))/(half*np.sqrt(2))])
                    outer_radius = radial / extent
                    fade = (outer_radius-radial)/(outer_radius-radius)
                    factor = 1 + bottom_delta * fade / radial
                coords *= factor[:,None]
            mapping[layer,used] = np.arange(count,count+len(used))
            points.append(np.column_stack([coords,np.full(len(used),z)]))
            count += len(used)
        for layer in range(len(levels)-1):
            a,b,c = mapping[layer,tri].T
            d,e,f = mapping[layer+1,tri].T
            prism = np.concatenate([np.stack([a,b,c,f],axis=1),
                np.stack([a,b,e,f],axis=1),np.stack([a,d,e,f],axis=1)])
            tets.append(prism)
            bodies.extend([body]*len(prism))
        maps.append(mapping)
    xyz, tetra = np.concatenate(points), np.concatenate(tets)
    upper = maps[0][0,triangles]
    lower = np.where(cone[:,None], maps[2][0,triangles], maps[1][0,triangles])
    area = np.abs(np.linalg.det(xy[triangles[:,1:]]-xy[triangles[:,0,None]])) / 2
    edge_counts = {}
    for tri in triangles[cone]:
        for a,b in [(tri[0],tri[1]),(tri[1],tri[2]),(tri[0],tri[2])]:
            edge_counts[(a,b)] = edge_counts.get((a,b),0)+1
    wall_edges = np.array([edge for edge,n in edge_counts.items() if n == 1])
    if reflection_symmetric:
        # The two symmetry-plane boundaries are not physical cone walls.
        wall_edges = wall_edges[np.all(np.isclose(np.linalg.norm(xy[wall_edges], axis=-1), radius), axis=1)]
    wall_cone,wall_substrate = [],[]
    for layer in range(len(cfg["bottom_levels_m"])-1):
        for mapping,out in [(maps[2],wall_cone),(maps[1],wall_substrate)]:
            a,b = mapping[layer,wall_edges].T
            c,d = mapping[layer+1,wall_edges].T
            out.extend([np.stack([a,b,d],axis=1),np.stack([a,c,d],axis=1)])
    wall_cone,wall_substrate = np.concatenate(wall_cone),np.concatenate(wall_substrate)
    vertices = xyz[wall_cone]
    cross = np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0])
    wall_area = np.linalg.norm(cross,axis=1)/2
    wall_normal = cross/(2*wall_area[:,None])
    inward = np.sum(wall_normal[:,:2]*vertices.mean(axis=1)[:,:2],axis=1) < 0
    wall_normal[inward] *= -1
    bolt_patches = []
    for centre in cfg["bolt_centres_m"]:
        selected = np.linalg.norm(centres-centre,axis=1) < cfg["bolt_patch_radius_m"]
        bolt_patches.append(maps[0][-1,np.unique(triangles[selected])])
    # Trestles act through unilateral contact, not bilateral displacement fixes.
    trestle = np.zeros(len(triangles),bool)
    for x in cfg["trestle_x_m"]:
        trestle |= np.abs(centres[:,0]-x) < cfg["trestle_width_m"]/2
    trestle_nodes = maps[1][-1,np.unique(triangles[trestle & ~cone])]
    trestle_area = np.zeros(count)
    selected = trestle & ~cone
    np.add.at(trestle_area, maps[1][-1,triangles[selected]].ravel(),
              np.repeat(area[selected]/3,3))
    fixed_z = np.unique(np.concatenate(bolt_patches))
    pins = [trestle_nodes[np.argmin(np.linalg.norm(xyz[trestle_nodes,:2]-target,axis=1))]
            for target in [(-0.7,-0.4),(0.7,-0.4)]]
    # Each independent cone has free lateral rigid modes before friction/bond
    # acts. The jack centre fixes x/y; a second point fixes in-plane rotation.
    cone_bottom = maps[2][-1]
    cone_bottom = cone_bottom[cone_bottom >= 0]
    cp = cone_bottom[np.argmin(np.linalg.norm(xyz[cone_bottom,:2],axis=1))]
    cr = cone_bottom[np.argmax(xyz[cone_bottom,0])]
    fixed = np.unique(np.concatenate([3*fixed_z+2,[3*pins[0],3*pins[0]+1,3*pins[1]+1,
                                                  3*cp,3*cp+1,3*cr+1]]))
    driven = 3*cone_bottom+2
    mesh = dict(xyz=xyz,tetra=tetra,body=np.asarray(bodies),interface_upper=upper,
                interface_lower=lower,interface_area=area,interface_body=np.where(cone,2,1),
                fixed_dofs=fixed,driven_dofs=driven,
                bolt_nodes=np.unique(np.concatenate(bolt_patches)),trestle_nodes=trestle_nodes,
                trestle_area=trestle_area,wall_cone=wall_cone,wall_substrate=wall_substrate,
                wall_area=wall_area,wall_normal=wall_normal)
    return reflect_quadrant(mesh, settings) if reflection_symmetric else mesh


def reflect_quadrant(mesh, settings):
    """Reflect complete tetrahedra, preserving diagonals and shared axis faces."""
    node_body = np.empty(len(mesh["xyz"]), dtype=int)
    for body in range(3):
        node_body[np.unique(mesh["tetra"][mesh["body"] == body])] = body
    points, lookup, maps = [], {}, []
    signs = [(1,1,1),(-1,1,1),(1,-1,1),(-1,-1,1)]
    for sign in signs:
        mapping = []
        for body, point in zip(node_body, mesh["xyz"]*sign):
            key = (int(body), *np.round(point, 12))
            if key not in lookup:
                lookup[key] = len(points)
                points.append(point)
            mapping.append(lookup[key])
        maps.append(np.asarray(mapping))
    result = {"xyz": np.asarray(points)}
    for key in ["tetra", "interface_upper", "interface_lower", "wall_cone", "wall_substrate"]:
        result[key] = np.concatenate([mapping[mesh[key]] for mapping in maps])
    for key in ["body", "interface_area", "interface_body", "wall_area"]:
        result[key] = np.tile(mesh[key], 4)
    result["wall_normal"] = np.concatenate([mesh["wall_normal"]*sign for sign in signs])
    for key in ["bolt_nodes", "trestle_nodes"]:
        result[key] = np.unique(np.concatenate([mapping[mesh[key]] for mapping in maps]))
    result["trestle_area"] = np.zeros(len(points))
    for mapping in maps:
        np.add.at(result["trestle_area"], mapping, mesh["trestle_area"])
    result["fixed_dofs"] = 3*result["bolt_nodes"]+2
    result["driven_dofs"] = 3*np.unique(np.concatenate([
        mapping[mesh["driven_dofs"]//3] for mapping in maps]))+2
    # Distributed lateral guides replace arbitrary off-axis point pins. Each
    # bottom face has only mean x/y translation and mean in-plane rotation
    # stiffness, leaving its other lateral deformation modes unconstrained.
    guide = []
    for body in [1, 2]:
        tetra = result["tetra"][result["body"] == body]
        bottom = settings["geometry"]["bottom_levels_m"][-1]
        weights = np.zeros(len(points))
        for local in [(0,1,2),(0,1,3),(0,2,3),(1,2,3)]:
            faces = tetra[:, local]
            faces = faces[np.all(np.isclose(result["xyz"][faces, 2], bottom), axis=1)]
            v = result["xyz"][faces]
            area = np.linalg.norm(np.cross(v[:,1]-v[:,0], v[:,2]-v[:,0]), axis=1)/2
            np.add.at(weights, faces.ravel(), np.repeat(area/3, 3))
        weights /= weights.sum()
        xy = result["xyz"][:, :2]
        radius = np.sqrt(np.sum(weights*np.sum(xy*xy, axis=1)))
        rows = np.zeros((3, len(points), 3))
        rows[0,:,0], rows[1,:,1] = weights, weights
        rows[2,:,0], rows[2,:,1] = -weights*xy[:,1]/radius, weights*xy[:,0]/radius
        guide.extend(rows.reshape(3, -1))
    result["lateral_guide_operator"] = np.asarray(guide)
    return result


def main():
    settings = json.loads((ROOT/"inputs/stateful_pilot.json").read_text(encoding="utf-8"))
    mesh = build_mesh(settings)
    dest = ROOT/"data/processed/stateful_pilot"
    dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/"mesh.npz",**mesh)
    print(json.dumps({"nodes":len(mesh["xyz"]),"tetrahedra":len(mesh["tetra"]),
                      "interface_triangles":len(mesh["interface_area"]),
                      "interface_area_m2":float(mesh["interface_area"].sum())}))


if __name__ == "__main__":
    main()
