
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


print( [model.geom(i).name for i in range(model.ngeom)])
print( [model.body(i).name for i in range(model.ngeom)])



names = [model.geom(i).name for i in range(model.ngeom)]


for name in names:
   if name != "":
      print(name, data.geom(name).xpos)



T = 10 # in seconds
t_des = np.linspace(0, T, T * 1000)

# x = cos( 2 * pi / T * t ) ; where t is in seconds
# vx = 2 * pi / 1000  * -sin( 2 * pi / 1000 * t ) ; where t is in seconds

# from zero to 2pi in T seconds
# w = 

x_des =  np.cos(np.linspace(0, 2*np.pi, T * 1000))
y_des =  np.sin(np.linspace(0, 2*np.pi, T * 1000))

# get desired velocity using finite differences
vx_des = - 2 * np.pi / T * np.sin(np.linspace(0, 2*np.pi, T * 1000))
vy_des =  2 * np.pi / T * np.cos(np.linspace(0, 2*np.pi, T* 1000))

axx_des = - ( 2 * np.pi / T ) ** 2 * np.cos(np.linspace(0, 2*np.pi, T * 1000))
ayy_des = - ( 2 * np.pi / T ) ** 2 * np.sin(np.linspace(0, 2*np.pi, T * 1000))

# plot desired trajectory

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.plot(t_des, x_des, label="x")
plt.plot(t_des, y_des, label="y")
plt.plot(t_des, vx_des, label="vx")
plt.plot(t_des, vy_des, label="vy")
plt.plot(t_des, axx_des, label="ax")
plt.plot(t_des, ayy_des, label="ay")
plt.legend()
plt.show()



# i could even compute the real acceleration and try!




# print("time step", m.opt.timestep)
# sys.exit(0)
kp = 3
kv = .00001
kvv = .3




print(data.qvel)

print('Total number of DoFs in the model:', model.nv)
print('Generalized positions:', data.qpos)
print('Generalized velocities:', data.qvel)


data.qvel[-2] = vy_des[0]
data.qvel[-3] = vx_des[0]

# sys.exit(0)


# body = m.geom('ball_geom')
# print(body)

# sys.exit(0)

id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'ball')
vel = np.zeros(6)

mujoco.mj_forward(model, data)
data.time = 0
sim_time = 0

simulation_time = 10
pos_array = []
times_array = []
controls_array = []
des_array = []
with mujoco.viewer.launch_passive(m, d,
    show_left_ui = False,
    show_right_ui = False) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and data.time < simulation_time:
    step_start = time.time()

    # Get current desired position
    xx  = np.interp(sim_time, t_des, x_des)
    yy = np.interp(sim_time, t_des, y_des)

    print("xx", xx)
    print("xx(t)", np.cos(2 * np.pi / T * sim_time))

    vx = np.interp(sim_time, t_des, vx_des)
    vy = np.interp(sim_time, t_des, vy_des)

    print("vx", vx)
    print("vx(t)", 2 * np.pi / T * -np.sin(2 * np.pi / T * sim_time))
    # vel_ball = data.geom("ball_geom").vel
    # xpos_cube = data.geom("cube_geom").xpos

    # id = model.body_name2id("ball")
    # body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'body_name')






    # mujoco.mj_forward(model, data)

    # mujoco.mj_step2(model, data)
    # mujoco.mj_step1(model, data)


    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, id, vel, True)
    pos = np.array(data.body('ball').xpos)
    pos_array.append(pos)
    des_array.append([xx,yy])
    vel1 = np.array(data.sensor('sensor_ball').data)

    print("data.qvel", data.qvel )
    print("pos: ", pos)
    print("vel: ", vel)
    print("vel1: ", vel1)


    d.ctrl[0] = kp * (xx - pos[0]) - kv * vel1[0]  +  kvv * (vx - vel1[0])
    d.ctrl[1] = kp * (yy - pos[1]) - kv * vel1[1]  +  kvv * (vy - vel1[1])

    max_u = np.array([1,1])
    min_u = np.array([-1,-1])

    axx = np.interp(sim_time, t_des, axx_des)
    ayy = np.interp(sim_time, t_des, ayy_des)
    print("axx", axx)
    print("axx(t)", - ( 2 * np.pi / T ) ** 2 * np.cos(2 * np.pi / T * sim_time))
    d.ctrl[0] = axx
    d.ctrl[1] = ayy


    # d.ctrl[0] = np.clip(d.ctrl[0], min_u[0], max_u[0])
    # d.ctrl[1] = np.clip(d.ctrl[1], min_u[1], max_u[1])

    controls_array.append(np.copy(d.ctrl))

    mujoco.mj_step(model, data)

    


    sim_time += m.opt.timestep
    print("sim time", sim_time)
    print("data time", data.time)
    times_array.append(data.time)



    # mujoco.mj_forward(model,data)



    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

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


