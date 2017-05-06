"""Test the SpaceCraft class

"""
import numpy as np

import spacecraft

class TestSpaceCraft():

    sc = spacecraft.SpaceCraft()

    def test_spacecraft_moi_symmetric(self):
        J = self.sc.J
        JT = self.sc.J.T
        np.testing.assert_array_almost_equal(J, JT)

    def test_des_attitude_attitude_so3(self):
        R_des, _ , _ = self.sc.des_attitude(0)
        np.testing.assert_array_almost_equal(R_des.T.dot(R_des), np.eye(3,3))
        np.testing.assert_almost_equal(np.linalg.det(R_des), 1)

    def test_des_attitude_angular_velocity_size(self):
        _, ang_vel_des , _ = self.sc.des_attitude(0)
        np.testing.assert_equal(ang_vel_des.shape, (3,))

    def test_des_attitude_angular_velocity_dot_size(self):
        _, _, ang_vel_dot_des = self.sc.des_attitude(0)
        np.testing.assert_equal(ang_vel_dot_des.shape, (3,))

    def test_integration_state_output_size(self):
        self.sc.integrate(10)
        np.testing.assert_equal(self.sc.state.shape, (1e3, 15)) 
