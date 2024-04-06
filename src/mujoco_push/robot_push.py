
# exposes the step function
# U  -->  Controller --> Torque 


import mujoco
from . import controllers
import numpy as np
import os

from . import MUJOCO_PUSH_DATADIR
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


    def _physic_step(self, data,model):

        self.low_level_force_control.policy(data)
        mujoco.mj_step(model, data)

    def reset(self,x):
        self.data.qpos = x
        self.data.qvel = np.zeros(8) # lets do zero velocity
        self.data.ctrl = np.zeros(2)
        mujoco.mj_forward(self.model, self.data) # is this necessary?


    def step(self,x,u):
        """
        """
        self.low_level_force_control.xdes = u
        self.data.qpos = x
        # self.data.qvel = np.zeros(8) # lets do zero velocity
        elapsed_time = 0
        while elapsed_time < self.dt:
            self._physic_step(self.data,self.model)
            elapsed_time += self.model.opt.timestep
        return np.copy(self.data.qpos)

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


