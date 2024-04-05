
import numpy as np
import mujoco

def get_state_summary(data):

    geom1_id = data.geom("gcube_0").id
    geom2_id = data.geom("gball_0").id

    case_a = data.geom("gcase_a").id
    case_b = data.geom("gcase_b").id
    case_c = data.geom("gcase_c").id
    case_d = data.geom("gcase_d").id

    case_ids = [case_a, case_b, case_c, case_d]

    # get contact status
    objects_in_contact = False
    contact_wall_cube = False
    for i in range(data.ncon):
        con = data.contact[i]
        # help(con)
        if (con.geom1 == geom1_id and con.geom2 == geom2_id) or (
            con.geom2 == geom1_id and con.geom1 == geom2_id
        ):
            contact_pos = con.pos
            objects_in_contact = True
            # break

        if con.geom1 in case_ids and con.geom2 == geom1_id:
            contact_wall_cube = True
        if con.geom2 in case_ids and con.geom1 == geom1_id:
            contact_wall_cube = True

    # print("objects_in_contact", objects_in_contact)

    D = {
        "q": np.array(data.qpos),
        "qvel": np.array(data.qvel),
        "time": data.time,
        "cube": np.array(data.body("cube").xpos),
        "ball": np.array(data.body("ball").xpos),
        "cube_vel": np.array(data.body("cube").cvel[3:]),
        "ball_vel": np.array(data.body("ball").cvel[3:]),
        "cube_r": np.array(data.body("cube").xquat),
        "ball_r": np.array(data.body("ball").xquat),
        "cube_w": np.array(data.body("cube").cvel[:3]),
        "ball_w": np.array(data.body("ball").cvel[:3]),
        "contact_ball_cube": np.array(objects_in_contact),
        "contact_cube_wall": np.array(contact_wall_cube),
        "u": np.array(data.ctrl),
    }

    return D


def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        raise ValueError("scene is full")
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )

