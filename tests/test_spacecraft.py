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

#    def test_integration_state_output_size(self):
#        self.sc.integrate(10)
#        np.testing.assert_equal(self.sc.state.shape, (1e3, 15)) 

class TestSpaceCraftNoAdaptiveControl():
    """This will test and duplicate the no adaptive control figure 
    from the paper
    """

    sc = spacecraft.SpaceCraft(
            scenario='multiple',
            avoid_switch=True,
            dist_switch=True,
            adaptive_switch=False)

    # known outputs for controller function
    t_in = 0.200340994817447
    state_in = np.array([  -0.539371310072546,
                        -0.839668102466909,
                        -0.063948035394251,
                        0.830978116201076,
                        -0.543015707201983,
                        0.121126699769061,
                        -0.136423515883572,
                        0.012196112655530,
                        0.990576800678766,
                        0.574229463658664,
                        0.382027448558780,
                        0.671356783919187,
                        0.060976084594403,
                        0.057293017520313,
                        0.068605292028059])
    u_m_des = np.array([-0.202628253334923,
                        -0.224410491790734,
                        -0.399208118724602])
    Psi_des = 2.056939224329732
    err_att_des = np.array([0.084720742542126,
                        0.273596815301540,
                        0.501272983471633])
    err_vel_des = np.array([0.574229463658664,
                            0.382027448558780,
                            0.671356783919187])

    (u_f_act, u_m_act, R_des_act, ang_vel_des_act, ang_vel_dot_des_act, Psi_act,
            err_att_act, err_vel_act) = sc.controller(t_in, state_in)

    state_dot_des = np.array([0.610000323296398,
                            -0.369216528604853,
                            -0.297108296161126,
                            0.283772185712330,
                            0.570720244060616,
                            0.611750332346718,
                            -0.683226763391052,
                            -0.008960644516801,
                            -0.093984424645144,
                            -0.733790650128088,
                            -3.996347748551956,
                            -18.962570630142391,
                            0.329475103100395,
                            0.327812131930160,
                            0.586314883695410])
    def test_constraint_vectors_unit_norm(self):
        norm_sum = np.sum(np.linalg.norm(self.sc.con, axis=0))
        np.testing.assert_equal(norm_sum, self.sc.num_con)

    def test_adaptive_control_switch(self):
        np.testing.assert_equal(self.sc.adaptive_switch, False)

    def test_scenario_switch(self):
        np.testing.assert_equal(self.sc.scenario, 'multiple')

    def test_dist_switch(self):
        np.testing.assert_equal(self.sc.dist_switch, True)

    def test_avoid_switch(self):
        np.testing.assert_equal(self.sc.avoid_switch, True)

    def test_constraint_angles(self):
        np.testing.assert_array_almost_equal(self.sc.con_angle, np.array([40, 40, 40, 20])*np.pi/180)

    def test_constraint_vectors_normalized(self):
        normalized_constraints = np.array([[0.183027337860104, 0.371393746768547, -0.853212903182330, -0.121952874319354],
                                           [-0.982457089432973, 0.656531295850099, 0.436108822728600, -0.139945921350078],
                                           [-0.035763962570365, 0.656531295850099, -0.286071383716467, -0.982620290622337]])
        np.testing.assert_array_almost_equal(self.sc.con, normalized_constraints)

    def test_delta_disturbance(self):
        np.testing.assert_array_almost_equal(self.sc.delta, np.array([0.2, 0.2, 0.2]))

    def test_controller_known_uf(self):
        np.testing.assert_array_almost_equal(self.u_f_act, np.zeros(3))

    def test_controller_known_um(self):
        np.testing.assert_array_almost_equal(self.u_m_act, self.u_m_des)

    def test_controller_known_Psi(self):
        np.testing.assert_equal(self.Psi_act, self.Psi_des)

    def test_controller_known_err_att(self):
        np.testing.assert_array_almost_equal(self.err_att_act, self.err_att_des)

    def test_controller_known_err_vel(self):
        np.testing.assert_array_almost_equal(self.err_vel_act, self.err_vel_des)

    def test_dynamics_known_state(self):
        statedot_act = self.sc.dynamics(self.state_in, self.t_in)
        np.testing.assert_array_almost_equal(statedot_act, self.state_dot_des)
