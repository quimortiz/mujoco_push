import mujoco
import cv2
import numpy as np
import yaml
import pickle
import pathlib
import pickle
import mujoco.viewer
import time
import utils
import controllers
import robot_push


import argparse

argsp = argparse.ArgumentParser()
argsp.add_argument("--simulate", action="store_true")
argsp.add_argument("--time", type=float, default=5)
argsp.add_argument("--data_file", type=str, default="")
argsp.add_argument("--vis", type=int, default=0)


# load the model.xml
xml_path = "model.xml"

model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

mujoco.mj_forward(model, data)
renderer.update_scene(data)


args = argsp.parse_args()
add_capsules = args.vis > 0

if not args.simulate and args.data_file == "":
    raise ValueError("You need to use --simulate or do provide a --data_file")

if args.vis > 1:
    while True:
        # cv2 uses BGR, mujoco uses RGB
        cv2.imshow("image", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        # close the window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

duration = 5  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
if args.vis > 1:
    frames = []
    mujoco.mj_resetData(model, data)  # Reset state and time.
    while data.time < duration:
        mujoco.mj_step(model, data)
        if len(frames) < data.time * framerate:
            renderer.update_scene(data)
            pixels = renderer.render()
            frames.append(pixels)

    for frame in frames:
        cv2.imshow("image", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(1 / framerate)

    cv2.destroyAllWindows()


m = model
d = data


print("qpos", data.qpos)
print("geoms", [model.geom(i).name for i in range(model.ngeom)])
print("bodies", [model.body(i).name for i in range(model.nbody)])



names = [model.geom(i).name for i in range(model.ngeom)]


for name in names:
    if name != "":
        print(name, data.geom(name).xpos)




center = np.array([0, 0, 0])


# sys.exit(0)


# body = m.geom('ball_geom')
# print(body)

# sys.exit(0)
id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
vel = np.zeros(6)

mujoco.mj_forward(model, data)
data.time = 0

simulation_time = args.time
pos_array = []
times_array = []
controls_array = []
des_array = []


# traces of time, position and speed
times = []
positions = []
speeds = []
offset = model.jnt_axis[0] / 16  # offset along the joint axis

point1 = np.array([1, 0, 1])
point2 = np.array([0, 1, 1])




all_data = []

import pathlib

if args.vis > 1:
    while True:
        # cv2 uses BGR, mujoco uses RGB
        cv2.imshow("image", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))
        # close the window when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


time_str = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
image_id = 0

if args.simulate:

    controller = controllers.Position_Planner(data, model, add_capsules)
    goal = controller.goal
    start = time.time()

    sim_time = [0]

    def step():

        # controller.policy(data)
        controller.policy(data)
        mujoco.mj_step(model, data)

        _data = utils.get_state_summary(data)
        _data_controller = controller.get_data()
        _data.update(_data_controller)

        # add dicts _data and _data_controller
        all_data.append(_data)

        sim_time[0] += m.opt.timestep
        times_array.append(data.time)

    if args.vis >= 1:

        with mujoco.viewer.launch_passive(
            m, d, show_left_ui=False, show_right_ui=False
        ) as viewer:

            controller.scene = viewer._user_scn
            last_render = time.time()
            while viewer.is_running() and data.time < simulation_time:

                step_start = time.time()
                step()
                viewer.sync()
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)

                current_time = time.time()

                # We can save some images  to the disk
                # if current_time - last_render > 1/24:
                #     last_render = current_time
                #     renderer.update_scene(data)
                #     frame = renderer.render()
                #     import matplotlib.pyplot as plt
                #
                #     # while True:
                #     #     # cv2 uses BGR, mujoco uses RGB
                #     #     cv2.imshow(
                #     #         "image", cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
                #     #     )
                #     #     # close the window when 'q' is pressed
                #     #     if cv2.waitKey(1) & 0xFF == ord("q"):
                #     #         break
                #     # cv2.destroyAllWindows()
                #
                #     file_out = f"images/frame_{time_str}/{image_id:05d}.png"
                #     image_id += 1
                #     pathlib.Path(file_out).parent.mkdir(parents=True, exist_ok=True)
                #     print("saving image to ", file_out)
                #     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                #     cv2.imwrite(file_out, frame)

    else:
        while data.time < simulation_time:
            step()

    now = time.time()

    data_out = f"data/main2_data_{time_str}.dat"
    pathlib.Path(data_out).parent.mkdir(parents=True, exist_ok=True)
    print("saving data to ", data_out)
    with open(data_out, "wb") as f:
        pickle.dump(all_data, f)

    # create another viewer
    replay_visualization = False
    if replay_visualization:
        print("replaying the simulation")

        data.ctrl[0] = 0.0
        data.ctrl[1] = 0.0

        all_data = all_data[int(525 / 0.002) :]
        # main2_data_2024-03-28--09-55-25.dat

        data.qpos = all_data[0]["q"]
        data.qvel = all_data[0]["qvel"]

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            last_time = 0
            for i in range(len(all_data)):
                if all_data[i]["time"] - last_time > 1 / 30:
                    last_time = all_data[i]["time"]
                    data.qpos = all_data[i]["q"]
                    data.qvel = all_data[i]["qvel"]
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(1 / 30)


if args.data_file != "":

    print("loading data from ", args.data_file)
    if args.data_file.endswith(".yaml"):
        with open(args.data_file, "r") as f:
            all_data = yaml.load(f, Loader=yaml.CLoader)
    elif args.data_file.endswith(".dat"):
        with open(args.data_file, "rb") as f:

            all_data = pickle.load(f)

    replay_visualization = True
    if replay_visualization:
        print("replaying the simulation")

        data.ctrl[0] = 0.0
        data.ctrl[1] = 0.0

        if False:
            pass
            all_data = all_data[int(525 / 0.002) :]
            # main2_data_2024-03-28--09-55-25.dat

            data.qpos = all_data[0]["q"]
            data.qvel = all_data[0]["qvel"]

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            with mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            ) as viewer:
                last_time = all_data[0]["time"]
                for i in range(len(all_data)):
                    if all_data[i]["time"] - last_time > 1 / 30:
                        last_time = all_data[i]["time"]
                        data.qpos = all_data[i]["q"]
                        data.qvel = all_data[i]["qvel"]
                        mujoco.mj_step(model, data)
                        mujoco.mj_forward(model, data)
                        viewer.sync()
                        time.sleep(1 / 30)


# print the trajectory of the ball and the cube
import matplotlib.pyplot as plt

print("len(all_data)", len(all_data))

Xball = [x["ball"][0] for x in all_data]
Yball = [x["ball"][1] for x in all_data]

Xcube = [x["cube"][0] for x in all_data]
Ycube = [x["cube"][1] for x in all_data]

plt.plot(Xball, Yball, label="ball")
plt.plot(Xcube, Ycube, label="cube")
plt.legend()

# save fig
fileout = f"figs/all_traj_{time_str}.png"
pathlib.Path(fileout).parent.mkdir(parents=True, exist_ok=True)
print("saving fig to ", fileout)
plt.savefig(fileout)

if args.vis >= 1:
    plt.show()
plt.close()

# plt.plot(np.array(pos_array)[:,0], np.array(pos_array)[:,1], label='trajectory')
# # plt.plot(x_des, y_des, label='desired trajectory')
# plt.plot(np.array(des_array)[:,0], np.array(des_array)[:,1], label='desired trajectory')
# plt.legend()
# plt.show()
#
#
# plt.plot(times_array, np.array(pos_array)[:,0], label="x")
# plt.plot(times_array, np.array(pos_array)[:,1], label="y")
# # plt.plot(t_des, x_des, label="x-des")
# # plt.plot(t_des, y_des, label="y-des")
#
# plt.plot(times_array, np.array(des_array)[:,0], label="x-des")
# plt.plot(times_array, np.array(des_array)[:,1], label="y-des")
#
# plt.plot(times_array, np.array(controls_array)[:,0], label="u_x")
# plt.plot(times_array, np.array(controls_array)[:,1], label="u_y")
# plt.legend()


times = [data["time"] for data in all_data]

fig, axs = plt.subplots(4, 1, sharex=True)

ax = axs[0]
ax.plot(times, Xball, label="x-ball")
ax.plot(times, Yball, label="y-ball")
ax.plot(times, Xcube, label="x-cube")
ax.plot(times, Ycube, label="y-cube")
ax = axs[1]
contact = [data["contact_ball_cube"] for data in all_data]
contact_wall = [data["contact_cube_wall"] for data in all_data]

U0 = [data["u"][0] for data in all_data]
U1 = [data["u"][1] for data in all_data]

ax.plot(times, contact, label="contact")
ax.plot(times, contact_wall, label="contact_wall")
ax.plot(times, U0, label="u0")
ax.plot(times, U1, label="u1")
ax.legend()

ax = axs[2]

Xdes = [data["xdes"][0] for data in all_data]
Ydes = [data["xdes"][1] for data in all_data]

ax.plot(times, Xdes, label="xdes-ball")
ax.plot(times, Ydes, label="ydes-ball")

ax.plot(times, Xball, label="x-ball", linestyle="--")
ax.plot(times, Yball, label="y-ball", linestyle="--")

ax.legend()

ax = axs[3]

# lets plot the mode

Dmode2int = {"to_rand": 0, "to_obj": 1, "recovery": 2}
modes = [Dmode2int[data["mode"]] for data in all_data]
ax.plot(times, modes, label="mode")


fileout = f"figs/all_traj_time{time_str}.png"
print("saving fig to ", fileout)
plt.savefig(fileout)

if args.vis >= 1:
    plt.show()
plt.close()


# lets replay the trajectories at this level!

# lest take

if  args.vis >= 1:

    index_start = 0
    index_end = 1000

    # set the initial state

    data.qpos = all_data[index_start]["q"]
    nvel = data.qvel.shape[0]
    data.qvel = np.zeros(nvel)
    data.ctrl[0] = 0
    data.ctrl[1] = 0

    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        last_time = 0
        time_sleep = 0.01
        data_recorded = []
        for i in range(index_start, index_end):
            tic = time.time()
            data_recorded.append(utils.get_state_summary(data))
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0
            data.qpos = all_data[i]["q"]
            data.qvel = np.zeros(8)
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()
            toc = time.time()
            time.sleep( max( 0, m.opt.timestep - (toc - tic)))
    print("lets do a rollout!")

    data.qpos = all_data[index_start]["q"]
    nvel = data.qvel.shape[0]
    data.qvel = np.zeros(nvel)
    data.ctrl[0] = 0.0
    data.ctrl[1] = 0.0

    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        last_time = 0
        time_sleep = 0.01
        data_rollout = []
        for i in range(index_start, index_end):
            tic = time.time()
            data_rollout.append(utils.get_state_summary(data))
            data.ctrl[0] = all_data[i]["u"][0]
            data.ctrl[1] = all_data[i]["u"][1]
            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()
            toc = time.time()
            time.sleep( max( 0, m.opt.timestep - (toc - tic)))

    print("lets do a rollout with highlevel abstraction!")
    data.qpos = all_data[index_start]["q"]
    nvel = data.qvel.shape[0]
    data.qvel = np.zeros(nvel)
    data.ctrl[0] = 0.0
    data.ctrl[1] = 0.0

    mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        last_time = 0
        time_sleep = 0.002

        low_level_force_control = controllers.Low_level_Force_Control()

        # lowlvel_controller = Position_Planner(data)
        data_rollout_high_level = []

        for i in range(index_start, index_end):

            data_rollout_high_level.append(utils.get_state_summary(data))
            low_level_force_control.xdes = all_data[i]["xdes"]
            # low_level_force_control.vdes = all_data[i]["vdes"]
            low_level_force_control.vdes = np.zeros(3)
            # all_data[i]["vdes"]
            low_level_force_control.policy(data)

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(time_sleep)

    # Plot all the data!

    fig, ax = plt.subplots(1, 1)
    # plot recorded data with solid line
    ax.plot(
        [d["ball"][0] for d in data_recorded],
        [d["ball"][1] for d in data_recorded],
        label="cube",
        linestyle="-",
    )
    ax.plot(
        [d["ball"][0] for d in data_rollout],
        [d["ball"][1] for d in data_rollout],
        label="cube",
        linestyle="--",
    )
    ax.plot(
        [d["ball"][0] for d in data_rollout_high_level],
        [d["ball"][1] for d in data_rollout_high_level],
        label="cube",
        linestyle="-.",
    )
    ax.legend()
    plt.show()





# N = 100
# Xdes_smooth = np.convolve(Xdes, np.ones(N)/N, mode='same')
# Ydes_smooth = np.convolve(Ydes, np.ones(N)/N, mode='same')
#
# ax.plot(times, Xdes_smooth, label="xdes-ball-s", linestyle="dotted")
# ax.plot(times, Ydes_smooth, label="ydes-ball-s", linestyle="dotted")

# lets sample at 0.1!

# save_times = np.linspace(0, times[-1], 1000)


# plot the controls


# plt.legend()


times_i = np.arange(0, times[-1], 0.1)

Xcube_i = np.interp(times_i, np.array(times), np.array(Xcube))
Ycube_i = np.interp(times_i, np.array(times), np.array(Ycube))
Xball_i = np.interp(times_i, np.array(times), np.array(Xball))
Yball_i = np.interp(times_i, np.array(times), np.array(Yball))
XballDes_i = np.interp(times_i, np.array(times), np.array(Xdes))
YballDes_i = np.interp(times_i, np.array(times), np.array(Ydes))
contact_i = np.interp(
    times_i, np.array(times), [float(data["contact_ball_cube"]) for data in all_data]
)
contact_wall_i = np.interp(
    times_i, np.array(times), [float(data["contact_cube_wall"]) for data in all_data]
)


# Q = np.interp(times_i, np.array(times), np.array([data["q"] for data in all_data]))
# Qvel = np.interp(times_i, np.array(times), np.array([data["qvel"] for data in all_data]))

nq = len(all_data[0]["q"])
nvel = len(all_data[0]["qvel"])

Q_i = np.array(
    [
        np.interp(
            times_i, np.array(times), np.array([data["q"][i] for data in all_data])
        )
        for i in range(nq)
    ]
).transpose()
Q_vel = np.array(
    [
        np.interp(
            times_i, np.array(times), np.array([data["qvel"][i] for data in all_data])
        )
        for i in range(nvel)
    ]
).transpose()

# Plot the interp data!

print("Q_i", Q_i[0])
print("ball", Xball_i[0], Yball_i[0])
print("cube", Xcube_i[0], Ycube_i[0])


# sys.exit(0)

fig, axs = plt.subplots(2, 1)

ax = axs[0]
ax.plot(times_i, Xball_i, label="x-ball")
ax.plot(times_i, Yball_i, label="y-ball")
ax.plot(times_i, XballDes_i, label="x-des")
ax.plot(times_i, YballDes_i, label="y-des")

ax = axs[1]

Xdif_ball = XballDes_i - Xball_i
Ydif_ball = YballDes_i - Yball_i

ax.plot(times_i, Xdif_ball, label="ux-ball")
ax.plot(times_i, Ydif_ball, label="uy-ball")

plt.legend()
plt.show()

number_of_cuts = 10000

trajs = []
skipped_trajs = []

print("len times", len(times_i))
len_traj = 16
for i in range(number_of_cuts):
    # choose start index at random:
    # rand_start = np.random.randint(0, len(times_i) - len_traj)
    rand_start = i * len_traj
    if rand_start > len(times_i) - len_traj:
        break
    print("rand_start", rand_start)
    Xs = []
    Us = []
    low_level_force = []
    contact_cube_wall = False
    for index in range(len_traj):
        X = Q_i[rand_start + index]
        U = [XballDes_i[rand_start + index], YballDes_i[rand_start + index]]
        # U = [ Xdif_ball[rand_start+index], Ydif_ball[rand_start+index] ]
        Xs.append(X)
        Us.append(U)
        if contact_wall_i[rand_start + index] > 0.0:
            contact_cube_wall = True
    traj = {"X": Xs, "U": Us}
    if contact_cube_wall:
        print(rand_start, "contact with the wall")
        print("skipping traj because of contact with the wall")
        skipped_trajs.append(traj)
    else:
        trajs.append(traj)

print("len(trajs)", len(trajs))
print("len(skipped_trajs)", len(skipped_trajs))

# lets try to simulate the trajs

print("lets simulate the trajs")


# lets store the trajs with pickle


traj_out = f"data/main2_trajs_{time_str}.dat"
print("saving data to ", traj_out)
with open(traj_out, "wb") as f:
    pickle.dump(trajs, f)

if args.vis >= 1:

    if True:
        for traj in trajs:

            data.qpos = traj["X"][0]
            nvel = data.qvel.shape[0]
            data.qvel = np.zeros(nvel)
            data.ctrl[0] = 0
            data.ctrl[1] = 0

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            time_sleep = 0.002

            with mujoco.viewer.launch_passive(
                model, data, show_left_ui=False, show_right_ui=False
            ) as viewer:
                last_time = 0
                for x in traj["X"]:
                    data.ctrl[0] = 0.0
                    data.ctrl[1] = 0.0
                    print("x", x)
                    data.qpos = x
                    data.qvel = np.zeros(8)
                    mujoco.mj_step(model, data)
                    mujoco.mj_forward(model, data)
                    time.sleep(time_sleep)
            print("done!")

            data.qpos = traj["X"][0]
            print("data.qpos", data.qpos)
            nvel = data.qvel.shape[0]
            data.qvel = np.zeros(nvel)
            data.ctrl[0] = 0.0
            data.ctrl[1] = 0.0

            mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)

            # with mujoco.viewer.launch_passive(
            #     model, data, show_left_ui=False, show_right_ui=False
            # ) as viewer:
            #
            #     time_sleep = 0.1
            #     low_level_force_control = controllers.Low_level_Force_Control()
            #     datas = []
            #     for i, u in enumerate(traj["U"]):
            #         # the u  is constant of
            #         u_number_steps = 0.1 / m.opt.timestep
            #         print("u", u)
            #         print("x", data.body("ball").xpos)
            #         print("xdes", np.array(data.body("ball").xpos)[:2] + np.array(u))
            #         # low_level_force_control.xdes =  np.array(data.body("ball").xpos)[:2] + np.array([u[0], u[1]])
            #         low_level_force_control.xdes = np.array(u)
            #         low_level_force_control.kv = 0
            #         low_level_force_control.vdes = np.zeros(2)
            #         for i in range(int(u_number_steps)):
            #             low_level_force_control.policy(data)
            #             # print("low level control", data.ctrl)
            #             datas.append(utils.get_state_summary(data))
            #             mujoco.mj_step(model, data)
            #             mujoco.mj_forward(model, data)
            #             viewer.sync()
            #         time.sleep(time_sleep)
            
            robot = robot_push.Robot_push()
            # x = robot.x0
            datas = []
            with mujoco.viewer.launch_passive(
                robot.model, robot.data, show_left_ui=False, show_right_ui=False
            ) as viewer:
                x = traj["X"][0]
                robot.reset(x)
                for i, u in enumerate(traj["U"]):
                    x = robot.step(x,u)
                    datas.append(utils.get_state_summary(robot.data))
                    mujoco.mj_forward(robot.model, robot.data)
                    viewer.sync()
                    time.sleep(0.1)
            # def step(self,x,u):


            # lets plot first the trajectory, then the training
            fig, axs = plt.subplots(4, 1)
            Xball = [x["ball"][0] for x in datas]
            Yball = [x["ball"][1] for x in datas]

            Xcube = [x["cube"][0] for x in datas]
            Ycube = [x["cube"][1] for x in datas]

            axs[0].set_title("Simulation -- Traj")
            axs[0].plot(Xball, Yball, color="b", label="ball")
            axs[0].plot(Xcube, Ycube, color="r", label="cube")
            axs[0].legend()

            # Q_i = np.array([np.interp(times_i, np.array(times), np.array([data["q"][i] for data in all_data])) for i in range(nq)]).transpose()

            print("traj X")

            DXball = [x[7] + 0.3 for x in traj["X"]]
            DYball = [x[8] for x in traj["X"]]
            DXcube = [x[0] for x in traj["X"]]
            DYcube = [x[1] for x in traj["X"]]

            axs[1].set_title("Recorded Data -- Traj")
            axs[1].plot(DXball, DYball, color="b", label="ball")
            axs[1].plot(DXcube, DYcube, color="r", label="cube")
            axs[1].legend()

            axs[2].set_title("Simulation -- time")
            axs[2].plot(Xball, label="x-ball")
            axs[2].plot(Yball, label="y-ball")
            axs[2].plot(Xcube, label="x-cube")
            axs[2].plot(Ycube, label="y-cube")
            axs[2].legend()

            axs[3].set_title("Recorded Data -- time")
            axs[3].plot(DXball, label="x-ball")
            axs[3].plot(DYball, label="y-ball")
            axs[3].plot(DXcube, label="x-cube")
            axs[3].plot(DYcube, label="y-cube")
            axs[3].legend()

            print("done!")
            plt.show()

    print("skipped trajs")
    for traj in skipped_trajs:
        input("Press Enter to continue...")
        time_sleep = 0.1

        data.qpos = traj["X"][0]
        nvel = data.qvel.shape[0]
        data.qvel = np.zeros(nvel)

        mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)

        with mujoco.viewer.launch_passive(
            model, data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            last_time = 0
            for x in traj["X"]:
                data.ctrl[0] = 0.0
                data.ctrl[1] = 0.0
                print("x", x)
                data.qpos = x
                data.qvel = np.zeros(8)
                mujoco.mj_step(model, data)
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(time_sleep)
        print("done!")


# lets tr
