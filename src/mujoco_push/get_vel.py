import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_path(filename='vel.xml')
data = mujoco.MjData(model)
for i in range(5):
    mujoco.mj_step(model, data)

    vel1 = np.array(data.sensor('xvelp').data)
    id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'box')
    vel2 = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, id, vel2, True)
    pos = np.array(data.body('box').xpos)
    vel3 = data.body('box').cvel

    print('time:', data.time, ' sensor vel:', vel1,' mj_objectVelocity:', vel2, 'pos:', pos, 
          'vel3:', vel3)
