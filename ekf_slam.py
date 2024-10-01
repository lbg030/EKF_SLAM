"""
Extended Kalman Filter SLAM example

author: Atsushi Sakai (@Atsushi_twi)
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from angle import angle_mod

# EKF state covariance
# x 공분산, y공분산, yaw 공분산 -> 즉 상태 추정의 공분산
Cx = np.diag([0.5, 0.5, np.deg2rad(30.0)]) ** 2 

#  Simulation parameter
# Q_sim : 입력 데이터에 대한 노이즈 공분산
# R_sim : 센서 관측값에 대한 노이즈 공분산
Q_sim = np.diag([0.2, np.deg2rad(1.0)]) ** 2
R_sim = np.diag([1.0, np.deg2rad(10.0)]) ** 2

DT = 0.1  # time tick [s]
SIM_TIME = 1000.0  # simulation time [s]
MAX_RANGE = 10.0  # maximum observation range
M_DIST_TH = 2.0  # Threshold of Mahalanobis distance for data association.
STATE_SIZE = 3  # State size [x,y,yaw]
LM_SIZE = 2  # LM state size [x,y]

show_animation = True


def ekf_slam(xEst, PEst, u, z):
    # XEst: 상태 추정값, PEst: 상태 추정 공분산, u: 노이즈가 포함된 입력값, z: 관측된 landmark 위치(거리, 각도, 랜드마크 번호)
    
    # Predict
    G, Fx = jacob_motion(xEst, u)
    xEst[0:STATE_SIZE] = motion_model(xEst[0:STATE_SIZE], u)
    PEst = G.T @ PEst @ G + Fx.T @ Cx @ Fx
    initP = np.eye(2)

    # Update ( correction )
    for iz in range(len(z[:, 0])):  # for each observation
        min_id = search_correspond_landmark_id(xEst, PEst, z[iz, 0:2])

        nLM = calc_n_lm(xEst) # 현재까지 몇개의 landmark를 보았는지
        if min_id == nLM:
            print("New LM")
            # Extend state and covariance matrix
            xAug = np.vstack((xEst, calc_landmark_position(xEst, z[iz, :])))
            PAug = np.vstack((np.hstack((PEst, np.zeros((len(xEst), LM_SIZE)))),
                              np.hstack((np.zeros((LM_SIZE, len(xEst))), initP))))
            xEst = xAug
            PEst = PAug
        
        # 기존 랜드마크 업데이트 및 칼만 이득 계산
        lm = get_landmark_position_from_state(xEst, min_id)
        # y: 관측된 값과 예상된 값 사이의 차이
        # S: 공분산
        # H: 관측 모델의 야코비안
        
        y, S, H = calc_innovation(lm, xEst, PEst, z[iz, 0:2], min_id)

        K = (PEst @ H.T) @ np.linalg.inv(S)
        xEst = xEst + (K @ y)
        PEst = (np.eye(len(xEst)) - (K @ H)) @ PEst

    xEst[2] = pi_2_pi(xEst[2])

    return xEst, PEst


def calc_input():
    v = 1.0  # [m/s]
    yaw_rate = 0.1  # [rad/s]
    u = np.array([[v, yaw_rate]]).T
    return u


def observation(xTrue, xd, u, RFID):
    xTrue = motion_model(xTrue, u) # noise 없이 업데이트 하면 GT

    # add noise to gps x-y
    # 로봇이 landmark를 관측한 정보를 저장할 배열
    z = np.zeros((0, 3)) #  z = [d, theta, i] -> d : 거리, theta : 각도, i : 랜드마크 번호

    for i in range(len(RFID[:, 0])):

        dx = RFID[i, 0] - xTrue[0, 0] # 랜드마크와 x축 거리 계산
        dy = RFID[i, 1] - xTrue[1, 0] # 랜드마크와 y축 거리 계산
        d = math.hypot(dx, dy) # 유클리디안 거리
        angle = pi_2_pi(math.atan2(dy, dx) - xTrue[2, 0])
        if d <= MAX_RANGE:
            dn = d + np.random.randn() * Q_sim[0, 0] ** 0.5  # add noise
            angle_n = angle + np.random.randn() * Q_sim[1, 1] ** 0.5  # add noise
            zi = np.array([dn, angle_n, i])
            z = np.vstack((z, zi))

    # add noise to input
    ud = np.array([[
        u[0, 0] + np.random.randn() * R_sim[0, 0] ** 0.5,
        u[1, 0] + np.random.randn() * R_sim[1, 1] ** 0.5]]).T

    # dead reckoning
    xd = motion_model(xd, ud)
    return xTrue, z, xd, ud


def motion_model(x, u):
    F = np.array([[1.0, 0, 0],
                  [0, 1.0, 0],
                  [0, 0, 1.0]])
    
    # DT를 곱하는 이유는, 속도에 시간 간격을 반영하기 위함 1m/s일 때, 0.1초가 지났으면 0.1m만큼 이동하고 회전도 동일한 맥락
    B = np.array([[DT * math.cos(x[2, 0]), 0], 
                  [DT * math.sin(x[2, 0]), 0],
                  [0.0, DT]])

    x = (F @ x) + (B @ u)
    return x


def calc_n_lm(x):
    n = int((len(x) - STATE_SIZE) / LM_SIZE) # -> X_t = [x, y, yaw, lm1_x, lm1_y, lm2_x, lm2_y, ...] 이므로, 랜드마크의 개수는 (전체 길이 - 상태 길이) / 랜드마크 길이
    return n


def jacob_motion(x, u):
    Fx = np.hstack((np.eye(STATE_SIZE), np.zeros(
        (STATE_SIZE, LM_SIZE * calc_n_lm(x)))))

    jF = np.array([[0.0, 0.0, -DT * u[0, 0] * math.sin(x[2, 0])],
                   [0.0, 0.0, DT * u[0, 0] * math.cos(x[2, 0])],
                   [0.0, 0.0, 0.0]], dtype=float)

    G = np.eye(len(x)) + Fx.T @ jF @ Fx

    return G, Fx,


def calc_landmark_position(x, z):
    zp = np.zeros((2, 1))

    zp[0, 0] = x[0, 0] + z[0] * math.cos(x[2, 0] + z[1])
    zp[1, 0] = x[1, 0] + z[0] * math.sin(x[2, 0] + z[1])

    return zp


def get_landmark_position_from_state(x, ind):
    lm = x[STATE_SIZE + LM_SIZE * ind: STATE_SIZE + LM_SIZE * (ind + 1), :]

    return lm


def search_correspond_landmark_id(xAug, PAug, zi):
    """
    Landmark association with Mahalanobis distance
    """

    nLM = calc_n_lm(xAug)

    min_dist = []

    for i in range(nLM):
        lm = get_landmark_position_from_state(xAug, i)
        y, S, H = calc_innovation(lm, xAug, PAug, zi, i)
        min_dist.append(y.T @ np.linalg.inv(S) @ y)

    min_dist.append(M_DIST_TH)  # new landmark

    min_id = min_dist.index(min(min_dist))

    return min_id


def calc_innovation(lm, xEst, PEst, z, LMid):
    delta = lm - xEst[0:2]
    q = (delta.T @ delta)[0, 0]
    z_angle = math.atan2(delta[1, 0], delta[0, 0]) - xEst[2, 0]
    zp = np.array([[math.sqrt(q), pi_2_pi(z_angle)]])
    y = (z - zp).T
    y[1] = pi_2_pi(y[1])
    H = jacob_h(q, delta, xEst, LMid + 1)
    S = H @ PEst @ H.T + Cx[0:2, 0:2]

    return y, S, H


def jacob_h(q, delta, x, i):
    sq = math.sqrt(q)
    G = np.array([[-sq * delta[0, 0], - sq * delta[1, 0], 0, sq * delta[0, 0], sq * delta[1, 0]],
                  [delta[1, 0], - delta[0, 0], - q, - delta[1, 0], delta[0, 0]]])

    G = G / q
    nLM = calc_n_lm(x)
    F1 = np.hstack((np.eye(3), np.zeros((3, 2 * nLM))))
    F2 = np.hstack((np.zeros((2, 3)), np.zeros((2, 2 * (i - 1))),
                    np.eye(2), np.zeros((2, 2 * nLM - 2 * i))))

    F = np.vstack((F1, F2))

    H = G @ F

    return H


def pi_2_pi(angle):
    return angle_mod(angle)


def main():
    print(__file__ + " start!!")

    time = 0.0

    # RFID positions [x, y] -> 랜드 마크 위치
    RFID = np.array([[10.0, -2.0],
                     [15.0, 10.0],
                     [3.0, 15.0],
                     [-5.0, 20.0]])

    # State Vector [x y yaw v]'
    xEst = np.zeros((STATE_SIZE, 1))
    xTrue = np.zeros((STATE_SIZE, 1))
    PEst = np.eye(STATE_SIZE)

    xDR = np.zeros((STATE_SIZE, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue

    while SIM_TIME >= time:
        time += DT
        u = calc_input() # 선속도, 각속도

        xTrue, z, xDR, ud = observation(xTrue, xDR, u, RFID) # xTrue : 실제 위치, z : 각 랜드마크에 대한 거리와 각도, xDR : Dead Reckoning, ud : 노이즈 추가된 입력값

        xEst, PEst = ekf_slam(xEst, PEst, ud, z)

        x_state = xEst[0:STATE_SIZE]

        # store data history
        hxEst = np.hstack((hxEst, x_state))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])

            plt.plot(RFID[:, 0], RFID[:, 1], "*k")
            plt.plot(xEst[0], xEst[1], ".r")

            # plot landmark
            for i in range(calc_n_lm(xEst)):
                plt.plot(xEst[STATE_SIZE + i * 2],
                         xEst[STATE_SIZE + i * 2 + 1], "xg")
            # black : dead reckoning, blue: gt, red: estimation ( z 포함 )
            plt.plot(hxTrue[0, :],
                     hxTrue[1, :], "-b")
            plt.plot(hxDR[0, :],
                     hxDR[1, :], "-k")
            plt.plot(hxEst[0, :],
                     hxEst[1, :], "-r")
            plt.axis("equal")
            plt.grid(True)
                # 현재 time 시각화 추가
            plt.text(0.05, 0.95, f'Time: {time:.2f} s', transform=plt.gca().transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
            
            plt.pause(0.001)


if __name__ == '__main__':
    main()