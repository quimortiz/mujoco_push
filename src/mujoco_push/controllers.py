import numpy as np
try:
    from . import utils
except:
    import utils

class Low_level_Force_Control:

    def __init__(self):
        self.kp = 10
        # self.kv = 5
        # self.damping = .1
        self.kv = 0  # lets try kv to zero
        self.damping = 5.0
        self.xdes = np.array([0, 0, 0])
        self.vdes = np.array([0, 0, 0])
        self.u0 = np.array([0, 0])
        self.umax = np.array([1, 1])
        self.umin = np.array([-1, -1])

    def policy(self, data):
        """ """
        x = data.body("ball").xpos
        v = data.body("ball").cvel[3:]

        # print("x", x)
        # print("v", v)
        # print("xdes", self.xdes)
        # print("vdes", self.vdes)

        u = (
            self.kp * (self.xdes[:2] - x[:2])
            + self.kv * (self.vdes[:2] - v[:2])
            - self.damping * v[:2]
        )
        u = np.clip(u, self.umin, self.umax)
        data.ctrl[0] = u[0]
        data.ctrl[1] = u[1]

    def control_u0(self, data):
        """ """
        data.ctrl[0] = self.u0[0]
        data.ctrl[1] = self.u0[1]


class Position_Planner:

    def __init__(self, data, model, add_capsules):
        self.low_level_force_control = Low_level_Force_Control()

        self.low_level_force_control.xdes = np.array(data.body("ball").xpos[:3])
        self.low_level_force_control.vdes = np.zeros(3)

        self.plan = []  # sequence of positions and velocities
        self.reference_max_vel = 0.2
        self.mode = "to_rand"
        self.p_lb = np.array([-0.3, -0.3, 0.05])
        self.p_ub = np.array([0.3, 0.3, 0.05])
        self.goal = self.p_lb + np.random.rand(3) * (self.p_ub - self.p_lb)
        self.start = data.body("ball").xpos
        self.scene = None  # used to add visual capsules

        self.p_lb_small = self.p_lb + 0.1 * (self.p_ub - self.p_lb)
        self.p_ub_small = self.p_ub - 0.1 * (self.p_ub - self.p_lb)
        self.last_call_time = 0

        self.dif = self.goal - self.start
        time = np.linalg.norm(self.dif) / self.reference_max_vel
        # The goal is to reach self.goal from self.start in time time

        # ratio_high_level_low_level   = 50
        # Low level at 1 / m.opt.timestep ( 1 / 0.002 = 500 )
        # High level at 1 / m.opt.timestep / ratio_high_level_low_level, e.g. 1 / 0.002 / 50 = 10
        # num_steps = int(time / m.opt.timestep)

        self.hl_time = 0.1  # in seconds
        self.ll_time = model.opt.timestep
        times = np.linspace(0, time, int(time / self.hl_time))
        self.plan = [self.start + t * self.dif / time for t in times]
        self.times = times
        self.add_capsules = add_capsules
        self.center = np.array([0, 0, 0])


        # if add_capsules:
        #     pass
            # add_visual_capsule(
            #     self.scene,
            #     self.goal,
            #     self.goal + np.array([0, 0, 0.1]),
            #     0.05,
            #     np.array([0, 0, 1, 0.1]),
            # )
            #
            # if len(self.plan) > 1000:
            #     for i in range(0, len(self.plan), 1000):
            #         add_visual_capsule(
            #             self.scene,
            #             self.plan[i],
            #             self.plan[i] + np.array([0, 0, 0.1]),
            #             0.05,
            #             np.array([0, 0, 1, 0.1]),
            #         )

        self.counter = 0

    def get_data(self):
        return {
            "xdes": self.low_level_force_control.xdes,
            "vdes": self.low_level_force_control.vdes,
            "mode": self.mode,
        }

    # first control is at 0.102 -- then i can take all the control from that

    def policy(self, data):

        # genereate a xdes and vdes
        if data.time - self.last_call_time < self.hl_time:
            # i don't do anything, only call again the lowe level force control
            self.low_level_force_control.policy(data)

        else:
            self.last_call_time = data.time
            if self.counter < len(self.plan):
                xdes = self.plan[self.counter]
                vdes = (
                    np.zeros(3)
                    if self.counter == len(self.plan) - 1
                    else (self.plan[self.counter + 1] - self.plan[self.counter])
                    / self.hl_time
                )
                self.counter += 1
                # print("counter is ", self.counter)
                # print("updating xdes and vdes")
                # print("xdes", xdes)
                # print("vdes",vdes)
                if (
                    np.linalg.norm(self.low_level_force_control.xdes[:2] - xdes[:2])
                    > 0.1
                ):
                    print("Warning: very big change in xdes")
                self.low_level_force_control.xdes = np.copy(xdes)
                self.low_level_force_control.vdes = np.copy(vdes)

                self.dist_fail_controller = 0.2

                if (
                    np.linalg.norm(data.body("ball").xpos - xdes)
                    > self.dist_fail_controller
                ):
                    print("PRINT: failed to reach the desired position!")
                    print("lets do a random motion instead")

                    self.mode = "to_rand"
                    self.goal = self.p_lb + np.random.rand(3) * (self.p_ub - self.p_lb)

                    if self.add_capsules:
                        utils.add_visual_capsule(
                            self.scene,
                            self.goal,
                            self.goal + np.array([0, 0, 0.1]),
                            0.05,
                            np.array([0, 0, 1, 0.1]),
                        )

                    self.start = data.body("ball").xpos
                    self.dif = self.goal - self.start
                    time = np.linalg.norm(self.dif) / self.reference_max_vel
                    steps = int(time / self.hl_time)
                    self.plan = [
                        self.start + t * self.dif / steps for t in range(steps)
                    ]
                    print(f"current plan ({len(self.plan)}) ) is ", self.plan)
                    self.counter = 0
                    self.mode = "to_rand"

                    if self.add_capsules:
                        if len(self.plan) > 1000:
                            for i in range(0, len(self.plan), 1000):
                                utils.add_visual_capsule(
                                    self.scene,
                                    self.plan[i],
                                    self.plan[i] + np.array([0, 0, 0.1]),
                                    0.05,
                                    np.array([0, 0, 1, 0.1]),
                                )

                else:
                    self.low_level_force_control.policy(data)
            else:
                self.counter = 0

                xdes = data.body("ball").xpos[:2]
                vdes = np.zeros(2)

                if np.linalg.norm(self.low_level_force_control.xdes[:2] - xdes) > 0.1:
                    print("Warning: very big change in xdes")
                self.low_level_force_control.xdes = np.copy(xdes)
                self.low_level_force_control.vdes = np.copy(vdes)
                # self.low_level_force_control.control_u0(data)
                # generate a new plan
                # change the mode
                # compute distance between cube and ball

                self.threshold_rand_vs_obj = 0.1

                cube = np.array(data.body("cube").xpos)
                if np.any(cube[:2] < self.p_lb_small[:2]) or np.any(
                    cube[:2] > self.p_ub_small[:2]
                ):

                    print("cube is outside!")
                    rel_cube = cube - self.center

                    # I will generate 3 goal

                    goal_1 = np.zeros(3)

                    goal_1[0] = rel_cube[1]
                    goal_1[1] = -1 * rel_cube[0]

                    a = 2 * np.pi / 3  # lets do "a" degrees
                    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                    goal_1[:2] = 0.8 * R @ goal_1[:2]

                    goal_2 = 1.8 * rel_cube
                    goal_3 = self.center

                    # take an orthogonal_direction:
                    self.big_goals = [goal_1, goal_2, goal_3]

                    if self.add_capsules:
                        for g in self.big_goals:
                            utils.add_visual_capsule(
                                self.scene,
                                g,
                                g + np.array([0, 0, 0.1]),
                                0.05,
                                np.array([1, 1, 0, 0.1]),
                            )

                    self.plan = []
                    for i in range(len(self.big_goals)):

                        if i == 0:
                            start = data.body("ball").xpos
                        else:
                            start = self.big_goals[i - 1]
                        dif = self.big_goals[i] - start

                        time = np.linalg.norm(self.dif) / self.reference_max_vel

                        num_steps = int(time / self.hl_time)
                        print(f"num_steps {i}", num_steps)
                        self.plan += [
                            start + t * dif / num_steps for t in range(num_steps)
                        ]

                    self.mode = "recovery"
                    print("new high level mode", self.mode)
                    print("len(self.plan)", len(self.plan))

                    if self.add_capsules:
                        if len(self.plan) > 1000:
                            for i in range(0, len(self.plan), 1000):
                                utils.add_visual_capsule(
                                    self.scene,
                                    self.plan[i],
                                    self.plan[i] + np.array([0, 0, 0.1]),
                                    0.05,
                                    np.array([1, 1, 0, 0.1]),
                                )

                else:
                    if (
                        np.linalg.norm(data.body("cube").xpos - data.body("ball").xpos)
                        < self.threshold_rand_vs_obj
                    ):
                        self.mode = "to_rand"
                    else:
                        self.mode = "to_obj"

                    print("new high level mode", self.mode)
                    if self.mode == "to_rand":
                        self.goal = self.p_lb + np.random.rand(3) * (
                            self.p_ub - self.p_lb
                        )

                        if self.add_capsules:
                            utils.add_visual_capsule(
                                self.scene,
                                self.goal,
                                self.goal + np.array([0, 0, 0.1]),
                                0.05,
                                np.array([0, 0, 1, 0.1]),
                            )

                    elif self.mode == "to_obj":
                        cube = np.array(data.body("cube").xpos)
                        ball = np.array(data.body("ball").xpos)
                        diff_weights = np.array([1, 1, 0])
                        # self.penetration = 0.2
                        self.penetration = 0.4
                        cube_to_ball = diff_weights * np.array(cube - ball)
                        self.goal = (
                            cube
                            + self.penetration
                            * cube_to_ball
                            / np.linalg.norm(cube_to_ball)
                        )
                        # self.goal = np.clip(self.goal, self.p_lb, self.p_ub)

                        if self.add_capsules:
                            utils.add_visual_capsule(
                                self.scene,
                                self.goal,
                                self.goal + np.array([0, 0, 0.1]),
                                0.05,
                                np.array([1, 0, 0, 0.1]),
                            )

                    self.start = data.body("ball").xpos

                    self.dif = self.goal - self.start

                    time = np.linalg.norm(self.dif) / self.reference_max_vel
                    times = np.linspace(0, time, int(time / self.hl_time))

                    self.plan = [self.start + t * self.dif / time for t in times]

                    if self.add_capsules:
                        if len(self.plan) > 1000:
                            for i in range(0, len(self.plan), 1000):
                                utils.add_visual_capsule(
                                    self.scene,
                                    self.plan[i],
                                    self.plan[i] + np.array([0, 0, 0.1]),
                                    0.05,
                                    np.array([1, 0, 0, 0.1]),
                                )

        # generate a plan of q


class Controller:
    """ """

    def __init__(self):
        """ """
        self.mode = "to_rand"  # to_rand
        self.goal_tolerance = 0.05
        self.penetration = 0.5
        self.goal_tolerance = self.goal_tolerance
        self.error = np.zeros(3)
        self.diff_weights = np.array([1, 1, 0])
        self.next_goals = []
        self.max_steps_per_mode = 4000
        self.steps_in_mode = 0
        self.scene = None

        self.p_lb = np.array([-0.3, -0.3, 0.05])
        self.p_ub = np.array([0.3, 0.3, 0.05])

        self.p_lb_small = self.p_lb + 0.1 * (self.p_ub - self.p_lb)
        self.p_ub_small = self.p_ub - 0.1 * (self.p_ub - self.p_lb)

        self.p_lb_small[2] = self.p_lb[2]
        self.p_ub_small[2] = self.p_ub[2]

        if self.mode == "to_obj":
            # get position of the e
            cube = np.array(data.body("cube").xpos)
            ball = np.array(data.body("ball").xpos)
            cube_to_ball = self.diff_weights * np.array(cube - ball)
            self.goal = cube + self.penetration * cube_to_ball / np.linalg.norm(
                cube_to_ball
            )
            self.goal = np.clip(self.goal, self.p_lb, self.p_ub)
        elif self.mode == "to_rand":
            r = np.random.rand(3)
            print("r", r)
            print("self.p_lb", self.p_lb)
            print("self.p_ub", self.p_ub)
            self.goal = self.p_lb + np.random.rand(3) * (self.p_ub - self.p_lb)

        print("initial goal", self.goal)

    def policy(self, data):
        """ """
        # Check if I reached the current goal

        self.steps_in_mode += 1

        cube = np.array(data.body("cube").xpos)
        ball = np.array(data.body("ball").xpos)
        vel_ball = np.array(data.body("ball").cvel)
        cube_to_ball = self.diff_weights * np.array(cube - ball)
        dist_goal = np.linalg.norm(self.diff_weights * (self.goal - ball))
        # print("ball", ball)
        # print("goal", self.goal)
        # print("dist_goal", dist_goal)

        geom1_id = data.geom("gcube_0").id
        geom2_id = data.geom("gball_0").id
        # get contact status
        objects_in_contact = False
        for i in range(data.ncon):
            con = data.contact[i]
            if (con.geom1 == geom1_id and con.geom2 == geom2_id) or (
                con.geom2 == geom1_id and con.geom1 == geom2_id
            ):
                contact_pos = con.pos
                objects_in_contact = True
                # break
        # print("objects_in_contact", objects_in_contact)

        # check if cube is outside of small bounds
        index_outside = -1
        outside = False
        outside_min = False  # or True to indicate it it goes outside min or max

        # check if cube outside of bounds
        is_outside = np.any(cube[:2] < self.p_lb_small[:2]) or np.any(
            cube[:2] > self.p_ub_small[:2]
        )
        print(f"mode {self.mode} goal {self.goal}")
        if is_outside and not self.mode.startswith("recovery_twds"):
            print("cube is outside!")
            # i have to put the ball outside and push inside
            rel_cube = cube - self.center

            # I will generate 3 goal

            goal_1 = np.zeros(3)

            goal_1[0] = rel_cube[1]
            goal_1[1] = -1 * rel_cube[0]

            # lets do a rotation of 45 degrees

            a = 2 * np.pi / 3
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            goal_1[:2] = 0.8 * R @ goal_1[:2]

            goal_2 = 1.8 * rel_cube
            goal_3 = self.center

            # take an orthogonal_direction:
            self.goals = [goal_1, goal_2, goal_3]
            self.mode = "recovery_twds_1"
            self.goal = goal_1
            if self.add_capsules:
                utils.add_visual_capsule(
                    self.scene,
                    self.goal,
                    self.goal + np.array([0, 0, 0.1]),
                    0.05,
                    np.array([0, 0, 1, 0.1]),
                )

            data.ctrl[0] = 0
            data.ctrl[1] = 0

            return

        if self.steps_in_mode > self.max_steps_per_mode:
            self.goal = self.p_lb + np.random.rand(3) * (self.p_ub - self.p_lb)
            self.mode = "to_rand"
            self.error = np.zeros(3)
            if self.add_capsules:
                utils.add_visual_capsule(
                    self.scene,
                    self.goal,
                    self.goal + np.array([0, 0, 0.1]),
                    0.05,
                    np.array([1, 1, 0, 0.1]),
                )
            self.steps_in_mode = 0

            data.ctrl[0] = 0
            data.ctrl[1] = 0

            return

        if dist_goal < self.goal_tolerance:
            self.steps_in_mode = 0

            if self.mode == "to_obj" or self.mode == "recovery_twds_3":
                print("old mode", self.mode)
                self.goal = self.p_lb + np.random.rand(3) * (self.p_ub - self.p_lb)
                self.mode = "to_rand"
                self.error = np.zeros(3)
                if self.add_capsules:
                    utils.add_visual_capsule(
                        self.scene,
                        self.goal,
                        self.goal + np.array([0, 0, 0.1]),
                        0.05,
                        np.array([1, 0, 0, 0.1]),
                    )
            elif self.mode == "to_rand":
                print("old mode", self.mode)
                cube = np.array(data.body("cube").xpos)
                ball = np.array(data.body("ball").xpos)
                cube_to_ball = self.diff_weights * np.array(cube - ball)
                self.goal = cube + self.penetration * cube_to_ball / np.linalg.norm(
                    cube_to_ball
                )
                self.ray = self.goal - ball

                # compute a scaling factor so that the cube does not go outside of small bounds
                # self.ray = self.goal - ball
                # r =
                # for i in range(2):
                #     if self.goal[i] < p_lb:

                alpha = 1.0
                while np.any(ball[:2] + alpha * self.ray[:2] < self.p_lb[:2]) or np.any(
                    ball[:2] + alpha * self.ray[:2] > self.p_ub[:2]
                ):
                    alpha *= 0.9

                self.goal = ball + alpha * self.ray

                # self.goal = np.clip(self.goal, p_lb_small, p_ub_small)
                self.error = np.zeros(3)
                self.mode = "to_obj"
                if self.add_capsules:
                    utils.add_visual_capsule(
                        self.scene,
                        self.goal,
                        self.goal + np.array([0, 0, 0.1]),
                        0.05,
                        np.array([0, 1, 0, 0.1]),
                    )
            elif self.mode == "recovery_twds_1":
                self.goal = self.goals[1]
                self.mode = "recovery_twds_2"
                if self.add_capsules:
                    utils.add_visual_capsule(
                        self.scene,
                        self.goal,
                        self.goal + np.array([0, 0, 0.1]),
                        0.05,
                        np.array([0, 0, 1, 0.1]),
                    )
            elif self.mode == "recovery_twds_2":
                self.goal = self.goals[2]
                self.mode = "recovery_twds_3"
                if self.add_capsules:
                    utils.add_visual_capsule(
                        self.scene,
                        self.goal,
                        self.goal + np.array([0, 0, 0.1]),
                        0.05,
                        np.array([0, 0, 1, 0.1]),
                    )

            data.ctrl[0] = 0
            data.ctrl[1] = 0

            return

        kp = 10
        kv = 10
        ki = 0.001

        if objects_in_contact:
            kp = 10

        self.error += self.goal - ball
        data.ctrl[0] = (
            kp * (self.goal[0] - ball[0]) - kv * vel_ball[3 + 0] + ki * self.error[0]
        )
        data.ctrl[1] = (
            kp * (self.goal[1] - ball[1]) - kv * vel_ball[3 + 1] + ki * self.error[1]
        )

        u_max = 10 * np.array([1, 1])
        u_min = -10 * np.array([1, 1])

        if not objects_in_contact:
            u_max = 1 * np.array([1, 1])
            u_min = -1 * np.array([1, 1])

        vel_limit = 1.0
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
