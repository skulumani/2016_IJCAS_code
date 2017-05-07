"""Spacecraft rigid body class

"""
import numpy as np
from scipy import integrate
import pdb
from kinematics import attitude

class SpaceCraft(object):

    def __init__(
            self, 
            scenario='multiple', 
            dist_switch=False, 
            avoid_switch=False, 
            adaptive_switch=False):
        """Initialize the model and it's properties
        """
        self.m_sc = 1
        self.J = np.array( [[55710.50413e-7, 617.6577e-7, -250.2846e-7],
                       [617.6577e-7, 55757.4605e-7, 100.6760e-7],
                       [-250.2846e-7, 100.6760e-7, 105053.7595e-7]])
        
        self.G = np.diag([0.9, 1.0, 1.1])

        self.kp = 0.4
        self.kv = 0.296
        
        self.sen = np.array([1, 0, 0])

        # logic to change the constraints 
        self.scenario = scenario
        self.dist_switch = dist_switch
        self.avoid_switch = avoid_switch
        self.adaptive_switch = adaptive_switch
        if scenario == 'multiple':
            self.con = np.array([[0.174, 0.4, -0.853, -0.122],
                            [-0.934, 0.7071, 0.436, -0.140],
                            [-0.034, 0.7071, -0.286, -0.983]])
            self.con_angle = np.array([40, 40, 40, 20])*np.pi/180
        elif scenario == 'single':
            self.con = np.array([ 1/np.sqrt(2), 1/np.sqrt(2), 0])
            self.con_angle = 12 * np.pi/180

        self.con = self.con / np.linalg.norm(self.con, axis=0)

        self.alpha = 15
        self.num_con = self.con.shape[1]

        # Adaptive control paramters
        self.W = np.eye(3,3)
        self.delta = lambda t: np.array([np.sin(9 * t), np.cos(9 * t), 1/2*(np.sin(9*t) + np.cos(9*t))])
        self.kd = 0.5
        self.c = 1

        # desired/initial conditions
        self.q0 = np.array([-0.188, -0.735, -0.450, -0.471]) 
        self.qd = np.array([0, 0, 0, 1])

        if scenario == 'multiple':
            self.R0 = attitude.rot1(0).dot(attitude.rot3(225 * np.pi/180))
            self.Rd = np.eye(3,3)
        elif scenario == 'single':
            self.R0 = attitude.rot1(0).dot(attitude.rot3(0))
            self.Rd = attitude.rot3(90*np.pi/180)

        # define the initial state
        self.w0 = np.zeros(3)
        self.delta_est0 = np.zeros(3)
        self.initial_state = np.hstack((self.R0.reshape(9), self.w0, self.delta_est0))

        self.tspan = np.linspace(0, 20, 1e3)

    def dynamics(self, state, t):
        """EOMs for the attitude dynamics of a rigid body

        """
        m_sc = self.m_sc
        J = self.J
        kd = self.kd
        W = self.W

        R = state[0:9].reshape((3,3))
        ang_vel = state[9:12]
        delta_est = state[12:15]

        # determine the external force and moment
        (_, m) = self.ext_force_moment(t, state)

        # compute control input
        (_, u_m, _, _, _, _, err_att, err_vel) = self.controller(t, state)

        # differential equations
        R_dot = R.dot(attitude.hat_map(ang_vel))
        ang_vel_dot = np.linalg.inv(J).dot(m + u_m - attitude.hat_map(ang_vel).dot(J.dot(ang_vel)))
        theta_est_dot = kd * W.T.dot(err_vel + self.c * err_att)

        state_dot = np.hstack((R_dot.reshape(9), ang_vel_dot, theta_est_dot))
        
        return state_dot
    def ext_force_moment(self, t, state):
        """External moment on the spacecraft

        """

        R = state[0:9].reshape((3,3))
        ang_vel = state[9:12]

        m_sc = self.m_sc
        J = self.J
        W = self.W
        delta = self.delta 

        # external force
        f = np.zeros(3)

        if self.dist_switch:
            m = W.dot(delta)
        else:
            m = np.zeros(3)

        return (f, m)

    def des_attitude(self, t):
        """Define the desired attitude for the simulation
        """
        a  = 2*np.pi/(20/10)
        b = np.pi/9

        phi = b*np.sin(a*t)
        theta = b*np.cos(a*t)
        psi = 0

        phi_d = b*a*np.cos(a*t)
        theta_d = -b*a*np.sin(a*t)
        psi_d = 0

        phi_dd = -b*a**2*np.sin(a*t)
        theta_dd = -b*a**2*np.cos(a*t)
        psi_dd = 0

        R_des = self.Rd

        ang_vel_des = np.zeros(3)

        ang_vel_dot_des = np.zeros(3)

        return (R_des, ang_vel_des, ang_vel_dot_des)
    def controller(self, t, state):
        """Controller for SO(3) avoidance
        """
        R = state[0:9].reshape((3,3))
        ang_vel = state[9:12]
        delta_est = state[12:15]

        # extract out the object parameters
        J = self.J
        G = self.G
        kp = self.kp
        kv = self.kv
        sen = self.sen
        alpha = self.alpha
        con_angle = self.con_angle
        con = self.con
        W = self.W
        num_con = self.num_con

        (R_des, ang_vel_des, ang_vel_dot_des) = self.des_attitude(t)

        psi_attract = 1/2*np.trace(G.dot(np.eye(3,3) - R_des.T.dot(R)))
        dA = 1/2*attitude.vee_map(G.dot(R_des.T).dot(R) - R.T.dot(R_des).dot(G))

        if self.avoid_switch:
            # use the constraint avoidance term
            sen_inertial = R.dot(sen)

            psi_avoid = np.zeros(self.num_con)
            dB = np.zeros(3,num_con)

            for ii in range(num_con):
                c = np.squeeze(con[:,ii])
                a = con_angle[ii]
                psi_avoid[ii] = -1 / alpha * np.log((np.cos(a)-np.inner(sen_inertial, c))/ (1 + np.cos(a)))
                dB[ii] = 1/alpha/( np.inner(sen_inertial, c) - np.cos(a)*attitude.hat_map(R.T.dot(c)).dot(sen))

            Psi = psi_attract * (np.sum(psi_avoid) + 1)
            err_att = dA * (np.sum(psi_avoid) + 1) + np.sum(dB * psi_attract, axis=1)
        else:
            err_att = dA
            Psi = psi_attract
        
        err_vel = ang_vel - R.T.dot(R_des).dot(ang_vel_des)
        alpha_d = -attitude.hat_map(ang_vel).dot(R.T).dot(R_des).dot(ang_vel_des) + R.T.dot(R_des).dot(ang_vel_dot_des)

        # compute the control input
        u_f = np.zeros(3)
        
        if self.adaptive_switch:
            u_m = -kp * err_att - kv * err_vel + np.cross(ang_vel, J.dot(ang_vel)) - W.dot(delta_est)
        else:
            u_m = -kp * err_att - kv * err_vel + np.cross(ang_vel, J.dot(ang_vel))

        return (u_f, u_m, R_des, ang_vel_des, ang_vel_dot_des, Psi, err_att, err_vel)


    def integrate(self, tf):
        """Simulate the rigid body for the selected period of time

        Also compute and save the control parameters by passing them through the controller
        and saving it to the object

        """
        num_steps = 1e3
        self.time = np.linspace(0, tf, num_steps)
        self.state = integrate.odeint(self.dynamics, self.initial_state, self.time)

        # compute all the controller paramters again
        self.u_f = np.zeros((num_steps, 3))
        self.u_m = np.zeros_like(self.u_f)
        self.R_des = np.zeros((num_steps, 9))
        self.ang_vel_des = np.zeros_like(self.u_f)
        self.ang_vel_dot_des = np.zeros_like(self.u_f)
        self.Psi = np.zeros_like(self.time)
        self.err_att = np.zeros_like(self.u_f)
        self.err_vel = np.zeros_like(self.u_f)

        self.sen_inertial = np.zeros_like(self.u_f)
        self.ang_con = np.zeros((num_steps, self.num_con))

        for ii, (t, state) in enumerate(zip(self.time, self.state)):
            (u_f, u_m, R_des, ang_vel_des, ang_vel_dot_des, Psi, err_att, err_vel) = self.controller(t, state)

            # compute the angle to each constraint
            self.sen_inertial[ii, :] = state[0:9].reshape((3,3)).dot(self.sen).reshape((1,3))
            for jj in range(self.num_con):
                c = self.con[:, jj]
                self.ang_con[ii, jj] = 180/np.pi * np.arccos(np.dot(self.sen_inertial[ii,:], c))

            self.u_f[ii, :] = u_f
            self.u_m[ii, :] = u_m
            self.R_des[ii, :] = R_des.reshape(9)
            self.ang_vel_des[ii, :] = ang_vel_des
            self.ang_vel_dot_des[ii, :] = ang_vel_dot_des
            self.Psi[ii] = Psi
            self.err_att[ii, :] = err_att
            self.err_vel[ii, :] = err_vel

        return 0



