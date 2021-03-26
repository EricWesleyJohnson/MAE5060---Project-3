'''
Toggle comment ("Ctrl+/" if you're using Pycharm or a few other python IDEs) on lines 43 - 48 to run the different
lay-ups.  Results needed will print in the console, I believe that's all you should need. Hopefully it's pretty clear.
'''

import math
import numpy as np
from numpy import linalg as lg
from numpy import matmul as mm  # Matrix multiplication
from math import sin
from math import cos
from math import tan


def Reverse(lst):
    return [i for i in reversed(lst)]


def main():
    # Independent material properties for Scotchply 1002 in US units
    E11 = 5.6 * (10 ** 6)  # psi
    E22 = 1.2 * (10 ** 6)  # psi
    V12 = 0.26  # unit-less
    V21 = (V12 * E22) / E11  # unit-less
    G12 = 0.6 * (10 ** 6)  # psi

    # Typical strengths of Scotchply 1002 in US units
    SLt = 154 * (10 ** 3)  # psi
    SLc = 88.5 * (10 ** 3)  # psi
    STt = 4.5 * (10 ** 3)  # psi
    STc = 17.1 * (10 ** 3)  # psi
    SLTs = 10.4 * (10 ** 3)  # psi

    # Tsai-Wu Coefficients
    F11 = 1 / (SLt * SLc)
    F22 = 1 / (STt * STc)
    F12 = (-1 / 2) * math.sqrt(F11 * F22)
    F66 = 1 / (SLTs ** 2)
    F1 = (1 / SLt) - (1 / SLc)
    F2 = (1 / STt) - (1 / STc)

    # [Nxx, Nyy, Nxy, Mxx, Myy, Mxy] in lb/in & in-lb/in
    stress_resultant = np.array([[1000], [0], [0], [0], [0], [0]])

    # Enter a desired ply orientation angles in degrees here:
    angle_in_degrees = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, -45, 90, 90, 90, 90, -45, 45, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # angle_in_degrees = [0,0,0,0,0,0,0,0,0,0,0,0,45,-45,45,-45,45,-45,90,90,90,90,-45,45,-45,45,-45,45,0,0,0,0,0,0,0,0,0,0,0,0]
    # angle_in_degrees = [0,0,0,0,0,0,0,0,0,0,45,-45,45,-45,45,-45,45,-45,90,90,90,90,-45,45,-45,45,-45,45,-45,45,0,0,0,0,0,0,0,0,0,0]
    # angle_in_degrees = [0,0,0,0,0,0,0,0,45,-45,45,-45,45,-45,45,-45,45,-45,90,90,90,90,-45,45,-45,45,-45,45,-45,45,-45,45,0,0,0,0,0,0,0,0]
    # angle_in_degrees = [0,0,0,0,0,0,45,-45,45,-45,45,-45,45,-45,45,-45,90,90,90,90,90,90,90,90,-45,45,-45,45,-45,45,-45,45,-45,45,0,0,0,0,0,0]
    # angle_in_degrees = [0,0,45,-45,45,-45,45,-45,45,-45,45,-45,45,-45,45,-45,45,-45,90,90,90,90,-45,45,-45,45,-45,45,-45,45,-45,45,-45,45,-45,45,-45,45,0,0]

    N = len(angle_in_degrees)  # number of plies
    t_ply = 0.005  # ply thickness in m
    h = t_ply * N

    # Number of at each angle
    n_0 = angle_in_degrees.count(0)
    n_45 = 2 * angle_in_degrees.count(45)  # Using symmetry to save on processing resources
    n_90 = angle_in_degrees.count(90)

    # Actual percentages of each ply group
    n_0_percent = n_0 / N
    n_45_percent = n_45 / N
    n_90_percent = n_90 / N

    # Distance from laminate mid-plane to out surfaces of plies)
    z0 = -h / 2
    z = [0] * (N)
    for i in range(N):
        z[i] = (-h / 2) + ((i + 1) * t_ply)

    # Distance from laminate mid-plane to mid-planes of plies
    z_mid_plane = [0] * N
    for i in range(N):
        z_mid_plane[i] = (-h / 2) - (t_ply / 2) + ((i + 1) * t_ply)

    # Ply orientation angle translated to radians to simplify equations below
    angle = [0] * N
    for i in range(N):
        angle[i] = math.radians(angle_in_degrees[i])

    # Stress Transformation (Global to Local), pg 112
    T = [0] * N
    for i in range(N):
        T[i] = np.array([[cos(angle[i]) ** 2, sin(angle[i]) ** 2, 2 * sin(angle[i]) * cos(angle[i])],
                         [sin(angle[i]) ** 2, cos(angle[i]) ** 2, -2 * sin(angle[i]) * cos(angle[i])],
                         [-sin(angle[i]) * cos(angle[i]), sin(angle[i]) * cos(angle[i]),
                          cos(angle[i]) ** 2 - sin(angle[i]) ** 2]])

    # Strain Transformation (Global-to-Local), pg 113
    T_hat = [0] * N
    for i in range(N):
        T_hat[i] = np.array([[cos(angle[i]) ** 2, sin(angle[i]) ** 2, sin(angle[i]) * cos(angle[i])],
                             [sin(angle[i]) ** 2, cos(angle[i]) ** 2, -sin(angle[i]) * cos(angle[i])],
                             [-2 * sin(angle[i]) * cos(angle[i]), 2 * sin(angle[i]) * cos(angle[i]),
                              cos(angle[i]) ** 2 - sin(angle[i]) ** 2]])

    # The local/lamina compliance matrix, pg 110
    S11 = 1 / E11
    S12 = -V21 / E22
    S21 = -V12 / E11
    S22 = 1 / E22
    S33 = 1 / G12
    S = np.array([[S11, S12, 0], [S21, S22, 0], [0, 0, S33]])

    # The local/lamina stiffness matrix, pg 107
    Q_array = lg.inv(S)  # The inverse of the S matrix

    # The global/laminate stiffness and compliance matrices
    Q_bar_array = [0] * N
    for i in range(N):
        Q_bar_array[i] = mm(lg.inv(T[i]), mm(Q_array, T_hat[i]))  # The global/laminate stiffness matrix, pg 114

    A_array = [[0] * 3] * 3
    for i in range(N):
        A_array += Q_bar_array[i] * t_ply

    B_array = [[0] * 3] * 3
    for i in range(N):
        B_array += (1 / 2) * (Q_bar_array[i] * ((z[i] ** 2) - ((z[i] - t_ply) ** 2)))

    D_array = [[0] * 3] * 3
    for i in range(N):
        D_array += (1 / 3) * (Q_bar_array[i] * ((z[i] ** 3) - ((z[i] - t_ply) ** 3)))

    ABD_array = np.array([[A_array[0][0], A_array[0][1], A_array[0][2], B_array[0][0], B_array[0][1], B_array[0][2]],
                          [A_array[1][0], A_array[1][1], A_array[1][2], B_array[1][0], B_array[1][1], B_array[1][2]],
                          [A_array[2][0], A_array[2][1], A_array[2][2], B_array[2][0], B_array[2][1], B_array[2][2]],
                          [B_array[0][0], B_array[0][1], B_array[0][2], D_array[0][0], D_array[0][1], D_array[0][2]],
                          [B_array[1][0], B_array[1][1], B_array[1][2], D_array[1][0], D_array[1][1], D_array[1][2]],
                          [B_array[2][0], B_array[2][1], B_array[2][2], D_array[2][0], D_array[2][1], D_array[2][2]]])

    ABD_inverse_array = lg.inv(ABD_array)

    # Calculating the mid-plane strains and curvatures
    mid_plane_strains_and_curvatures_array = mm(lg.inv(ABD_array), stress_resultant)

    # Transforming numpy array into lists for ease of formatting
    Q = Q_array.tolist()
    Q_bar = [0] * N
    for i in range(N):
        Q_bar[i] = Q_bar_array[i].tolist()
    A = A_array.tolist()
    B = B_array.tolist()
    D = D_array.tolist()
    ABD_inverse = ABD_inverse_array.tolist()
    mid_plane_strains_and_curvatures = mid_plane_strains_and_curvatures_array.tolist()

    # Parsing the Mid-plane strains and curvatures apart
    mid_plane_strains = np.array([[mid_plane_strains_and_curvatures[0][0]], [mid_plane_strains_and_curvatures[1][0]],
                                  [mid_plane_strains_and_curvatures[2][0]]])
    curvatures = np.array([[mid_plane_strains_and_curvatures[3][0]], [mid_plane_strains_and_curvatures[4][0]],
                           [mid_plane_strains_and_curvatures[5][0]]])

    # Global Strains at mid-plane of each ply
    global_strains = [[[0]] * 3] * N
    for i in range(N):
        global_strains[i] = mid_plane_strains + z_mid_plane[i] * curvatures

    # Global Stresses at mid-plane of each ply
    global_stresses = [[[0]] * 3] * N
    for i in range(N):
        global_stresses[i] = mm(Q_bar[i], global_strains[i])

    # Local strains
    local_strains = [[[0]] * 3] * N
    for i in range(N):
        local_strains[i] = mm(T_hat[i], global_strains[i])

    # Local stresses
    local_stresses = [[[0]] * 3] * N
    for i in range(N):
        local_stresses[i] = mm(Q, local_strains[i])

    # Define Tsai-Wu quadratic function coefficients (aR^2 + bR + cc = 0)
    a = [0] * N
    for i in range(N):
        a[i] = (F11 * (local_stresses[i][0] ** 2)) + (2 * F12 * local_stresses[i][0] * local_stresses[i][1]) + (
                    F22 * (local_stresses[i][1] ** 2)) + (F66 * (local_stresses[i][2] ** 2))

    b = [0] * N
    for i in range(N):
        b[i] = (F1 * local_stresses[i][0]) + (F2 * local_stresses[i][1])

    cc = [-1] * N

    # Strength Ratios for Tsai-Wu Criteria
    R_1_array = [0] * N
    for i in range(N):
        R_1_array[i] = (-b[i] + math.sqrt((b[i] ** 2) - 4 * a[i] * cc[i])) / (2 * a[i])

    R_2 = [0] * N
    for i in range(N):
        R_2[i] = (-b[i] - math.sqrt((b[i] ** 2) - 4 * a[i] * cc[i])) / (2 * a[i])

    R_1 = [0] * N
    for i in range(N):
        R_1[i] = R_1_array[i].tolist()
    R_TW = min(R_1)

    # Tsai-Wu critical loads
    N_TW_xxc = float(R_TW * stress_resultant[0])

    # Calculating E_xx
    E_xx = (A[0][0] / h) * (1 - ((A[0][1] ** 2) / (A[0][0] * A[1][1])))

    # Calculating ε_xx and ε_xxc
    e_xx = float((stress_resultant[0]) / (E_xx * h))
    e_xxc = float(e_xx * R_TW[0])

    # Printing Ply Group Percentages
    print('Percent n_0:' + format(n_0_percent, '>9.2f'))
    print('Percent n_45:' + format(n_45_percent, '>8.2f'))
    print('Percent n_90:' + format(n_90_percent, '>8.2f'))

    print("\n# of ply that fails first: " + str(R_1.index(min(R_1)) + 1))

    # Printing the Critical loads
    print("\nThis is the calculated strain for first ply failure under Tsai-Wu:")
    print("ε_xx  = " + format(e_xx, '>8.5f'))

    # Printing the Strength Ratio for Tsai-Wu Failure
    print("\nThis is the Strength Ratio for the first ply failure under Tsai-Wu Failure Criterion:")
    print("R_TW = " + str(np.round(R_TW[0], 3)))

    # Printing the Critical loads
    print("\nThis is the critical strain for first ply failure under Tsai-Wu:")
    print("ε_xxc = " + format(e_xxc, '>8.5f'))


main()