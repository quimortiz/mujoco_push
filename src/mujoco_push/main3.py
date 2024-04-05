import mujoco 
import cv2
import numpy as np


# load the model.xml
xml_path = 'model.xml'



model = mujoco.MjModel.from_xml_path(xml_path)


data = mujoco.MjData(model)

# Make renderer, render and show the pixels
renderer = mujoco.Renderer(model)

mujoco.mj_forward(model, data)
renderer.update_scene(data)

while True:
    # cv2 uses BGR, mujoco uses RGB
    cv2.imshow('image', cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
    # close the window when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cv2.destroyAllWindows()

duration = 3.8  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)  # Reset state and time.
while data.time < duration:
  mujoco.mj_step(model, data)
  if len(frames) < data.time * framerate:
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)

import time

for frame in frames:
    cv2.imshow('image', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(1/framerate)

cv2.destroyAllWindows()
# mujoco modern viewer

import mujoco.viewer

m = model
d = data


print('positions', data.qpos)
import sys


print( "geoms", [model.geom(i).name for i in range(model.ngeom)])
print( "bodies", [model.body(i).name for i in range(model.nbody)])

print( "geoms", [model.geom(i).id for i in range(model.ngeom)])
print( "bodies", [model.body(i).id for i in range(model.nbody)])



names = [model.geom(i).name for i in range(model.ngeom)]


for name in names:
   if name != "":
      print(name, data.geom(name).xpos)



# p_lb = np.array([-1,-1,.05])
# p_ub = np.array([1,1,.05])

p_lb = np.array([-.3,-.3,.05])
p_ub = np.array([.3,.3,.05])

p_lb_small = p_lb + .1 * ( p_ub - p_lb)
p_ub_small = p_ub - .1 * ( p_ub - p_lb)

p_lb_small[2] = p_lb[2]
p_ub_small[2] = p_ub[2]

def add_visual_capsule(scene, point1, point2, radius, rgba):
  """Adds one capsule to an mjvScene."""
  if scene.ngeom >= scene.maxgeom:
    raise ValueError("scene is full")
  scene.ngeom += 1  # increment ngeom
  # initialise a new capsule, add it to the scene using mjv_makeConnector
  mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                      mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                      np.zeros(3), np.zeros(9), rgba.astype(np.float32))
  mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                           mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                           point1[0], point1[1], point1[2],
                           point2[0], point2[1], point2[2])

center = np.array([0,0,0])
class Controller:
    """
    """
    def __init__(self):
        """
        """
        self.mode = "to_rand" # to_rand
        self.goal_tolerance = 0.05
        self.penetration  = .5
        self.goal_tolerance = self.goal_tolerance
        self.error = np.zeros(3)
        self.diff_weights = np.array([1,1,0])
        self.next_goals = []
        self.max_steps_per_mode = 4000
        self.steps_in_mode = 0

        if self.mode == "to_obj":
            # get position of the e
            cube = np.array( data.body('cube').xpos)
            ball = np.array( data.body('ball').xpos)
            cube_to_ball = self.diff_weights * np.array(cube - ball)
            self.goal = cube + self.penetration *  cube_to_ball / np.linalg.norm(cube_to_ball)
            self.goal = np.clip(self.goal, p_lb, p_ub)
        elif self.mode == "to_rand":
            r = np.random.rand(3) 
            print("r",r)
            print("p_lb",p_lb)
            print("p_ub",p_ub)
            self.goal = p_lb +   np.random.rand(3)  * (p_ub - p_lb)

        print("initial goal", self.goal)
            
        # add_visual_capsule(scene, self.goal, self.goal + np.array([0,0,.1]) , .05, np.array([1, 0, 0, .1]))
        #


    def policy(self, data):
        """
        """
        # Check if I reached the current goal


        self.steps_in_mode += 1


        cube = np.array( data.body('cube').xpos)
        ball = np.array( data.body('ball').xpos)
        vel_ball = np.array( data.body('ball').cvel)
        cube_to_ball =  self.diff_weights * np.array(cube - ball)
        dist_goal = np.linalg.norm(self.diff_weights * (self.goal - ball))
        print("ball", ball)
        print("goal", self.goal)
        print("dist_goal", dist_goal)

        geom1_id = data.geom("gcube_0").id
        geom2_id = data.geom("gball_0").id
        # get contact status
        objects_in_contact = False
        for i in range(data.ncon):
            con = data.contact[i]
            if (con.geom1 == geom1_id and con.geom2 == geom2_id) or (con.geom2 == geom1_id and con.geom1 == geom2_id):
                contact_pos = con.pos
                objects_in_contact = True
                # break
        print("objects_in_contact", objects_in_contact)

        # check if cube is outside of small bounds
        index_outside = -1
        outside = False
        outside_min = False # or True to indicate it it goes outside min or max


        # check if cube outside of bounds
        is_outside  = np.any(cube[:2] < p_lb_small[:2]) or np.any(cube[:2] > p_ub_small[:2])
        print(f"mode {self.mode} goal {self.goal}")
        if is_outside and not self.mode.startswith("recovery_twds"):
            print("cube is outside!")
            # i have to put the ball outside and push inside
            rel_cube = cube - center

            # I will generate 3 goal

            goal_1 = np.zeros(3)

            goal_1[0] = rel_cube[1]
            goal_1[1] = -1 * rel_cube[0]

            # lets do a rotation of 45 degrees

            a = 2*np.pi/3
            R = np.array( [[ np.cos(a) , -np.sin( a) ] , [ np.sin(a) , np.cos(a) ]] )
            goal_1[:2] = .8 *  R @ goal_1[:2]

            goal_2 = 1.8 * rel_cube 
            goal_3 = center

            # take an orthogonal_direction:
            self.goals = [goal_1, goal_2, goal_3]
            self.mode = "recovery_twds_1"
            self.goal = goal_1
            add_visual_capsule(scene, self.goal, self.goal + np.array([0,0,.1]) , .05, np.array([0, 0, 1, .1]))

            data.ctrl[0] =  0
            data.ctrl[1] =  0 

            return

        if self.steps_in_mode > self.max_steps_per_mode:
            self.goal = p_lb + np.random.rand(3) *  (p_ub - p_lb)
            self.mode = "to_rand"
            self.error = np.zeros(3)
            add_visual_capsule(scene, self.goal, self.goal + np.array([0,0,.1]) , .05, np.array([1, 1, 0, .1]))
            self.steps_in_mode  = 0

            data.ctrl[0] =  0
            data.ctrl[1] =  0 

            return



        if dist_goal < self.goal_tolerance:
            self.steps_in_mode = 0

            if self.mode == "to_obj" or self.mode == "recovery_twds_3" :
                print("old mode", self.mode)
                self.goal = p_lb + np.random.rand(3) *  (p_ub - p_lb)
                self.mode = "to_rand"
                self.error = np.zeros(3)
                add_visual_capsule(scene, self.goal, self.goal + np.array([0,0,.1]) , .05, np.array([1, 0, 0, .1]))
            elif self.mode == "to_rand" :
                    print("old mode", self.mode)
                    cube = np.array( data.body('cube').xpos)
                    ball = np.array( data.body('ball').xpos)
                    cube_to_ball = self.diff_weights *  np.array(cube - ball)
                    self.goal = cube + self.penetration *  cube_to_ball / np.linalg.norm(cube_to_ball)
                    self.ray = self.goal - ball

                    # compute a scaling factor so that the cube does not go outside of small bounds
                    # self.ray = self.goal - ball
                    # r = 
                    # for i in range(2):
                    #     if self.goal[i] < p_lb:
                            
                    alpha = 1.
                    while np.any( ball[:2] + alpha * self.ray[:2] < p_lb[:2] ) or np.any(ball[:2] + alpha * self.ray[:2] > p_ub[:2]):
                        alpha *= .9

                    self.goal = ball + alpha * self.ray

                    # self.goal = np.clip(self.goal, p_lb_small, p_ub_small)
                    self.error = np.zeros(3)
                    self.mode = "to_obj"
                    add_visual_capsule(scene, self.goal, self.goal + np.array([0,0,.1]) , .05, np.array([0, 1, 0, .1]))
            elif self.mode == "recovery_twds_1":
                self.goal = self.goals[1]
                self.mode = "recovery_twds_2"
                add_visual_capsule(scene, self.goal, self.goal + np.array([0,0,.1]) , .05, np.array([0, 0, 1, .1]))
            elif self.mode == "recovery_twds_2":
                self.goal = self.goals[2]
                self.mode = "recovery_twds_3"
                add_visual_capsule(scene, self.goal, self.goal + np.array([0,0,.1]) , .05, np.array([0, 0, 1, .1]))

            data.ctrl[0] =  0
            data.ctrl[1] =  0 

            return

        
        kp = 10
        kv = 10
        ki = 0.001

        if objects_in_contact:
            kp = 10

        self.error  += self.goal - ball
        data.ctrl[0] = kp * (self.goal[0] - ball[0]) - kv * vel_ball[3+0] + ki * self.error[0]
        data.ctrl[1] = kp * (self.goal[1] - ball[1]) - kv * vel_ball[3+1] + ki * self.error[1]

        u_max = 10 * np.array([1,1])
        u_min = -10 * np.array([1,1])

        if not objects_in_contact:
            u_max = 1 * np.array([1,1])
            u_min = -1 * np.array([1,1])

        vel_limit = 1.
        print("vel_ball", vel_ball)
        if vel_ball[3] > vel_limit:
            u_max[0] = 0

        if vel_ball[4] > vel_limit:
            u_max[1] = 0

        if vel_ball[3] < -vel_limit:
            u_min[0] = 0
        if vel_ball[4] < -vel_limit:
            u_min[1] = 0

        data.ctrl[0] = np.clip(data.ctrl[0], u_min[0], u_max[0])
        data.ctrl[1] = np.clip(data.ctrl[1], u_min[1], u_max[1])


        print("data.ctrl", data.ctrl)


# sys.exit(0)


# body = m.geom('ball_geom')
# print(body)

# sys.exit(0)


id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
vel = np.zeros(6)

mujoco.mj_forward(model, data)
data.time = 0
sim_time = 0

simulation_time = 100
pos_array = []
times_array = []
controls_array = []
des_array = []



def add_visual_sphere(scene, point1, radius, rgba):
    """
    """

    if scene.ngeom >= scene.maxgeom:
        raise ValueError("scene is full")
    scene.ngeom += 1  # increment ngeom
    size = radius * np.ones(9)
    # size[0] = radius
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_SPHERE,  # Change geometry type to SPHERE
                        point1,  # Position
                        np.zeros(3),  # Orientation (not needed for spheres, but kept for API compatibility)
                        size,  # Size parameters, will be set in mjv_makeConnector
                        rgba.astype(np.float32))  # Color

    # Make connector for a sphere (mainly to set its radius)
    # For spheres, we can just use the center position (`point1` in this case) and ignore `point2`
    # mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
    #                          mujoco.mjtGeom.mjGEOM_SPHERE,  # Change geometry type to SPHERE
    #                          radius,
    #                          point1[0], point1[1], point1[2],
    #                          0, 0, 0)  # `point2` is not needed for spheres, setting to 0




 # traces of time, position and speed
times = []
positions = []
speeds = []
offset = model.jnt_axis[0]/16  # offset along the joint axis

# def modify_scene(scn):
#   """Draw position trace, speed modifies width and colors."""
#   if len(positions) > 1:
#     for i in range(len(positions)-1):
#       rgba=np.array((np.clip(speeds[i]/10, 0, 1),
#                      np.clip(1-speeds[i]/10, 0, 1),
#                      .5, 1.))
#       radius=.003*(1+speeds[i])
#       point1 = positions[i] + offset*times[i]
#       point2 = positions[i+1] + offset*times[i+1]
#       add_visual_capsule(scn, point1, point2, radius, rgba)


point1 = np.array([1, 0, 1])
point2 = np.array([0, 1, 1])

controller = Controller()


def get_state_summary(data):


    geom1_id = data.geom("gcube_0").id
    geom2_id = data.geom("gball_0").id

    wall_id = data.geom("gwall_0").id

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
        if (con.geom1 == geom1_id and con.geom2 == geom2_id) or (con.geom2 == geom1_id and con.geom1 == geom2_id):
            contact_pos = con.pos
            objects_in_contact = True
            # break

        if con.geom1 in case_ids and con.geom2 == geom2_id:
            contact_wall_cube  = True
        if con.geom2 in case_ids and con.geom1 == geom2_id:
            contact_wall_cube  = True

    print("objects_in_contact", objects_in_contact)



    D = {
        "cube" : data.body('cube').xpos, 
        "ball" : data.body('ball').xpos,
        "cube_vel" : data.body('cube').xpos, 
        "ball_vel" : data.body('ball').xpos,
        "cube_r" : data.body("cube_r").q, 
        "ball_r" : data.body("ball_r").q, 
        "cube_w" : data.body("cube_r").q, 
        "ball_w" : data.body("ball_r").q, 
        "contact_ball_cube" : objects_in_contact,
        "contact_cube_wall" : contact_wall_cube,
    }

    return D


with mujoco.viewer.launch_passive(m, d,
    show_left_ui = False,
    show_right_ui = False) as viewer:

  scene = viewer._user_scn
  goal = controller.goal
  add_visual_capsule(scene, controller.goal, controller.goal + np.array([0,0,.1]) , .05, np.array([1, 0, 0, .1]))
  # add_visual_sphere(scene, point1,  .1, np.array([1, 0, 0, 1]))

  # while True:
  #       time.sleep(0.1)

  # viewer.add_marker(type=mujoco.mjtGeom.mjGEOM_SPHERE,
  #                 pos=np.array([.5, .5, 1]),
  #                 label=str("quim"))

  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and data.time < simulation_time:
    step_start = time.time()

    # Get current desired position
    # xx  = np.interp(sim_time, t_des, x_des)
    # yy = np.interp(sim_time, t_des, y_des)
    #
    # print("xx", xx)
    # print("xx(t)", np.cos(2 * np.pi / T * sim_time))
    #
    # vx = np.interp(sim_time, t_des, vx_des)
    # vy = np.interp(sim_time, t_des, vy_des)
    #
    # print("vx", vx)
    # print("vx(t)", 2 * np.pi / T * -np.sin(2 * np.pi / T * sim_time))
    # vel_ball = data.geom("ball_geom").vel
    # xpos_cube = data.geom("cube_geom").xpos

    # id = model.body_name2id("ball")
    # body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'body_name')






    # mujoco.mj_forward(model, data)

    # mujoco.mj_step2(model, data)
    # mujoco.mj_step1(model, data)


    # mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, id, vel, True)
    # pos = np.array(data.body('ball').xpos)
    # pos_array.append(pos)
    # des_array.append([xx,yy])
    # vel1 = np.array(data.sensor('sensor_ball').data)
    #
    # print("data.qvel", data.qvel )
    # print("pos: ", pos)
    # print("vel: ", vel)
    # print("vel1: ", vel1)

    #
    # d.ctrl[0] = 0
    # d.ctrl[1] = 0 

    max_u = np.array([1,1])
    min_u = np.array([-1,-1])

    # axx = np.interp(sim_time, t_des, axx_des)
    # ayy = np.interp(sim_time, t_des, ayy_des)
    # print("axx", axx)
    # print("axx(t)", - ( 2 * np.pi / T ) ** 2 * np.cos(2 * np.pi / T * sim_time))
    # d.ctrl[0] = axx
    # d.ctrl[1] = ayy


    # d.ctrl[0] = np.clip(d.ctrl[0], min_u[0], max_u[0])
    # d.ctrl[1] = np.clip(d.ctrl[1], min_u[1], max_u[1])

    # controls_array.append(np.copy(d.ctrl))

    controller.policy(data)
    mujoco.mj_step(model, data)

    sim_time += m.opt.timestep
    times_array.append(data.time)

    # mujoco.mj_forward(model,data)

    # Example modification of a viewer option: toggle contact points every two seconds.
    # with viewer.lock():
    #   viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)

import matplotlib.pyplot as plt
plt.plot(np.array(pos_array)[:,0], np.array(pos_array)[:,1], label='trajectory')
# plt.plot(x_des, y_des, label='desired trajectory')
plt.plot(np.array(des_array)[:,0], np.array(des_array)[:,1], label='desired trajectory')
plt.legend()
plt.show()


plt.plot(times_array, np.array(pos_array)[:,0], label="x")
plt.plot(times_array, np.array(pos_array)[:,1], label="y")
# plt.plot(t_des, x_des, label="x-des")
# plt.plot(t_des, y_des, label="y-des")

plt.plot(times_array, np.array(des_array)[:,0], label="x-des")
plt.plot(times_array, np.array(des_array)[:,1], label="y-des")

plt.plot(times_array, np.array(controls_array)[:,0], label="u_x")
plt.plot(times_array, np.array(controls_array)[:,1], label="u_y")
plt.legend()

plt.show()
