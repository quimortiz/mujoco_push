
# exposes the step function
# U  -->  Controller --> Torque 


import mujoco

try:
    from . import controllers
except:
    import controllers

import numpy as np
import os
from scipy.spatial.transform import Rotation as R

try:
    from . import MUJOCO_PUSH_DATADIR
except:
    MUJOCO_PUSH_DATADIR = "xml_models"

class Robot_push():


    def __init__(self):
        """
        """
        # self.controller = controllers.Position_Planner(data)
        xml_path = os.path.join( MUJOCO_PUSH_DATADIR,  'model.xml')
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        # self.renderer = mujoco.Renderer(self.model)
        # self.renderer.update_scene(self.data)
        self.x0 = self.data.qpos

        mujoco.mj_forward(self.model, self.data)

        self.low_level_force_control = controllers.Low_level_Force_Control()
        self.dt = .1
        self.ball_offset_x = .3
        self.height_cube  = 0.05

    def _physic_step(self, data,model):

        self.low_level_force_control.policy(data)
        mujoco.mj_step(model, data)


    def from_reduced_x(self,_x):
        r = R.from_euler('xyz', [0., 0., _x[2]])
        _q = r.as_quat()
        # mujoco uses quaterion with w,x,y,z
        # scipy uses x,y,z,w
        q = np.array([_q[3], _q[0], _q[1], _q[2]])
        x = np.concatenate([_x[:2], [ self.height_cube],q, [_x[3]  - self.ball_offset_x],[ _x[4]]])
        return x


    def to_reduced_x(self,x):
        """
        """
        _q = R.from_quat([x[4], x[5], x[6], x[3]])
        euler  = _q.as_euler('xyz')
        yaw = euler[2]
        out = np.concatenate([x[:2], [yaw], [x[7] + self.ball_offset_x], [x[8]]])
        return out


    def reset(self,_x,reduced_x=False):

        if reduced_x:
            x = self.from_reduced_x(_x)
        else:
            x = _x

        self.data.qpos = x
        self.data.qvel = np.zeros(8) # lets do zero velocity
        self.data.ctrl = np.zeros(2)
        mujoco.mj_forward(self.model, self.data) # is this necessary?



    def step(self,_x,u, reduced_x=False):
        """
        """

        print(f"u {u}")
        print(f"_x {_x}")

        # TODO: lets fix the offset!!

        if reduced_x:
            x= self.from_reduced_x(_x)

        else:
            x = _x

        print(f"x {x}")
        self.low_level_force_control.xdes = u
        self.data.qpos = x
        # self.data.qvel = np.zeros(8) # lets do zero velocity
        elapsed_time = 0
        while elapsed_time < self.dt:
            self._physic_step(self.data,self.model)
            elapsed_time += self.model.opt.timestep
        print("out")
        print("q pos is", self.data.qpos)

        if reduced_x:
            out = self.to_reduced_x(self.data.qpos)
        else: 
            out = np.copy(self.data.qpos)

        print("out ", out)

        return out



    def cost(self,x,u):
        """
        """
        return self.dt


if __name__ == "__main__" :

    robot = Robot_push()

    x = robot.x0
    u = np.zeros(2)

    for i in range(100):
        x = robot.step(x,u)
        print(f"x {x}")


