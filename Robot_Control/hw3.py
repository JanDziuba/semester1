import control
import numpy as np
import pybullet as p

LINK_CART = 0
LINK_POLE = 1
JOINT_CART = 0
JOINT_POLE = 1

POLE_FRICTION_COEF = 0.001

cartpole_path = "assets/table-cartpole.urdf"


# This is a stub of your solution
# Add your code in any organized way, but please keep the following signatures unchanged
# Solution1 should optimize for speed, Solution2 for effort. Refer to the assignement specification.


# Keep this signature unchanged for automated testing!
# Returns 2 numpy arrays - matrices A and B
def linearize(
    gravity: float,
    mass_cart: float,
    mass_pole: float,
    length_pole: float,
    mu_pole: float,
):
    # A rows in order - cart position, cart velocity, pole angle. pole angular velocity.
    # Cart position change depends only on cart velocity.
    # Cart velocity change depends on car acceleration, so it depends on pole angle and pole angular velocity.
    # Pole angle change depends only on pole angular velocity.
    # Pole angular velocity change depends on pole acceleration, so it depends on pole angle and pole angular velocity.

    d_cv_d_pa = -mass_pole*gravity/((mass_cart+mass_pole)*(4/3-(mass_pole/(mass_cart+mass_pole))))
    d_cv_d_pav = mu_pole / ((length_pole * (4 / 3 - mass_pole / (mass_cart + mass_pole)))*(mass_cart + mass_pole))

    d_pav_d_pa = gravity/(length_pole*(4/3-mass_pole/(mass_cart+mass_pole)))
    d_pav_d_pav = (-mu_pole/(length_pole*mass_pole)) / (length_pole * (4 / 3 - mass_pole / (mass_cart + mass_pole)))

    A = np.array([[0, 1, 0, 0],
                  [0, 0, d_cv_d_pa, d_cv_d_pav],
                  [0, 0, 0, 1],
                  [0, 0, d_pav_d_pa, d_pav_d_pav]])

    d_cv_d_F = (mass_pole/((mass_cart+mass_pole)*(4/3-mass_pole/(mass_cart+mass_pole)))+1)/(mass_cart+mass_pole)
    d_pav_d_F = -1/((mass_cart+mass_pole)*length_pole*(4/3-mass_pole/(mass_cart+mass_pole)))

    B = np.array([[0, d_cv_d_F, 0, d_pav_d_F]]).reshape((4,1))
    return A, B


def get_u(cartpole, state, Q, R):
    gravity = 9.8
    mass_cart = list(p.getDynamicsInfo(cartpole, LINK_CART))[0]
    mass_pole = list(p.getDynamicsInfo(cartpole, LINK_POLE))[0]
    length_pole = list(list(list(p.getVisualShapeData(cartpole))[LINK_POLE + 1])[3])[1]
    mu_pole = POLE_FRICTION_COEF

    A, B = linearize(gravity, mass_cart, mass_pole, length_pole, mu_pole)

    K, S, E = control.lqr(A, B, Q, R)

    cart_pos = state[0]
    cart_vel = state[1]
    pole_angle = state[2]
    pole_ang_vel = state[3]

    x = np.array([cart_pos, cart_vel, pole_angle, pole_ang_vel])
    u = -K @ x
    return u[0]



class Solution1:
    # Keep this signature unchanged for automated testing!
    # Reminder: implementing arbitrary target_pos is not required, but please try!
    def __init__(self, init_state, target_pos):
        self.cartpole = p.loadURDF(cartpole_path)

    # Keep this signature unchanged for automated testing!
    # Returns one float - a desired force (u)
    def update(self, state):
        # big position value to quickly go to destination point
        # small R value to not punish using big force
        Q = np.array([[7000, 0, 0, 0],
                      [0, 1000, 0, 0],
                      [0, 0, 100, 0],
                      [0, 0, 0, 100]])
        R = np.array([[1]])
        
        return get_u(self.cartpole, state, Q, R)


class Solution2:
    # Keep this signature unchanged for automated testing!
    # Reminder: implementing arbitrary target_pos is not required, but please try!
    def __init__(self, init_state, target_pos):
        self.cartpole = p.loadURDF(cartpole_path)

    # Keep this signature unchanged for automated testing!
    # Returns one float - a desired force (u)
    def update(self, state):
        # big cart velocity value to go slow, so that cart does not overshoot destination point
        # big R value to punish using big force
        Q = np.array([[1000, 0, 0, 0],
                      [0, 10000, 0, 0],
                      [0, 0, 100, 0],
                      [0, 0, 0, 100]])
        R = np.array([[100]])

        return get_u(self.cartpole, state, Q, R)
