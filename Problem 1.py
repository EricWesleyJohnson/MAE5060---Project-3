'''
Toggle comment ("Ctrl+/" if you're using Pycharm or a few other python IDEs) on lines 43 - 48 to run the different
lay-ups.  Results needed will print in the console, I believe that's all you should need. Hopefully it's pretty clear.
'''

import math
import numpy as np
from numpy import linalg as lg
from numpy import matmul as mm  # Matrix multiplication
from math import sin as s
from math import cos as c


# Calculating the local (Q) and global (Q_bar) stiffness matrices
def calculate_Q_and_Q_bar(E11, E22, G12, V12, V21, N, T, T_hat):
    Q11 = E11 / (1 - V12 * V21)
    Q12 = (V21 * E11) / (1 - V12 * V21)
    Q21 = (V12 * E22) / (1 - V12 * V21)
    Q22 = E22 / (1 - V12 * V21)
    Q = np.array([[Q11, Q12, 0], [Q21, Q22, 0], [0, 0, G12]])
    Q_bar = []
    for i in range(N):
        Q_bar.append(mm(lg.inv(T[i]), mm(Q, T_hat[i])))  # The global/laminate stiffness matrix, pg 114
    return Q, Q_bar


def calculate_A(N, Q_bar, t_ply):
    A = [[0] * 3] * 3
    for i in range(N):
        A += Q_bar[i] * t_ply
    return A


def calculate_B(N, Q_bar, z, t_ply):
    B = [[0] * 3] * 3
    for i in range(N):
        B += (1 / 2) * (Q_bar[i] * ((z[i] ** 2) - ((z[i] - t_ply) ** 2)))
    return B


def calculate_D(N, Q_bar, z, t_ply):
    D = [[0] * 3] * 3
    for i in range(N):
        D += (1 / 3) * (Q_bar[i] * ((z[i] ** 3) - ((z[i] - t_ply) ** 3)))
    return D


def calculate_ABD(A, B, D):
    ABD = np.array([[A[0][0], A[0][1], A[0][2], B[0][0], B[0][1], B[0][2]],
                    [A[1][0], A[1][1], A[1][2], B[1][0], B[1][1], B[1][2]],
                    [A[2][0], A[2][1], A[2][2], B[2][0], B[2][1], B[2][2]],
                    [B[0][0], B[0][1], B[0][2], D[0][0], D[0][1], D[0][2]],
                    [B[1][0], B[1][1], B[1][2], D[1][0], D[1][1], D[1][2]],
                    [B[2][0], B[2][1], B[2][2], D[2][0], D[2][1], D[2][2]]])
    return ABD


# Global Strains at mid-plane of each ply
def calculate_global_strains(N, mid_plane_strains,  z_mid_plane, curvatures):
    global_strains = [[[0]] * 3] * N
    for i in range(N):
        global_strains[i] = mid_plane_strains + z_mid_plane[i] * curvatures
    return global_strains


# Global Stresses at mid-plane of each ply
def calculate_global_stresses(N, Q_bar, global_strains):
    global_stresses = [[[0]] * 3] * N
    for i in range(N):
        global_stresses[i] = mm(Q_bar[i], global_strains[i])
    return global_stresses


# Local strains at mid-plane of each ply
def calculate_local_strains(N, T_hat, global_strains):
    local_strains = [[[0]] * 3] * N
    for i in range(N):
            local_strains[i] = mm(T_hat[i], global_strains[i])
    return local_strains

# Local stresses at mid-plane of each ply
def calculate_local_stresses(N, Q, local_strains):
    local_stresses = [[[0]] * 3] * N
    for i in range(N):
        local_stresses[i] = mm(Q, local_strains[i])
    return local_stresses

# Strength ratios for max stress failure criterion
def R_Max_Stress(N, SLt, STt, SLTs, local_stresses):
    R_sig_11 = []
    for i in range(N):
        R_sig_11.append(SLt / math.fabs(local_stresses[i][0]))
    R_sig_22 = []
    for i in range(N):
        R_sig_22.append(STt / math.fabs(local_stresses[i][1]))
    R_tau_12 = []
    for i in range(N):
        R_tau_12.append(SLTs / math.fabs(local_stresses[i][2]))
    R_MS = []
    for i in range(N):
        R_MS.append(min(R_sig_11[i], R_sig_22[i], R_tau_12[i]))
    return R_MS


# Max stress failure criterion critical loads
def R_MS_Critical_Loads(N, R_MS, stress_resultant):
    N_MScrit = []
    for i in range(N):
        N_MScrit.append(R_MS[i] * max(stress_resultant[0], stress_resultant[1], stress_resultant[2], key=abs))
    M_MScrit = []
    for i in range(N):
        M_MScrit.append(R_MS[i] * max(stress_resultant[3], stress_resultant[4], stress_resultant[5], key=abs))
    return N_MScrit, M_MScrit


# Define Tsai-Wu quadratic function coefficients (aR^2 + bR + cc = 0)
def Tsai_Wu_coeffs(N, SLt, SLc, STt, STc, SLTs, local_stresses):
    # Tsai-Wu Coefficients
    F11 = 1 / (SLt * SLc)
    F22 = 1 / (STt * STc)
    F12 = (-1 / 2) * math.sqrt(F11 * F22)
    F66 = 1 / (SLTs ** 2)
    F1 = (1 / SLt) - (1 / SLc)
    F2 = (1 / STt) - (1 / STc)
    a = []
    for i in range(N):
        a.append((F11 * (local_stresses[i][0] ** 2)) + (2 * F12 * local_stresses[i][0] * local_stresses[i][1]) + (
                    F22 * (local_stresses[i][1] ** 2)) + (F66 * (local_stresses[i][2] ** 2)))
    b = []
    for i in range(N):
        b.append((F1 * local_stresses[i][0]) + (F2 * local_stresses[i][1]))
    cc = [-1] * N
    return a, b, cc


# Strength ratios for Tsai-Wu criterion
def R_Tsai_Wu(N, a, b, cc):
    R_1 = []
    for i in range(N):
        R_1.append((-b[i] + math.sqrt((b[i] ** 2) - 4 * a[i] * cc[i])) / (2 * a[i]))
    R_2 = []
    for i in range(N):
        R_2.append((-b[i] - math.sqrt((b[i] ** 2) - 4 * a[i] * cc[i])) / (2 * a[i]))
    R_TW = []
    for i in range(N):
        R_TW.append(float(R_1[i]))
    return R_TW


def main():
    # Independent material properties for T300/5208 graphite epoxy in SI units
    E11  = 181  * (10**9)  # Pascals
    E22  = 10.3 * (10**9)  # Pascals
    V12  = 0.28            # unit-less
    V21 = (V12*E22)/E11    # unit-less
    G12  = 7.17 * (10**9)  # Pascals

    # # Typical strengths of T300/5208 graphite epoxy in SI units
    SLt  = 1500 * (10**6)  # Pascals
    SLc  = 1500 * (10**6)  # Pascals
    STt  = 40   * (10**6)  # Pascals
    STc  = 246  * (10**6)  # Pascals
    SLTs = 68   * (10**6)  # Pascals

    # [Nxx, Nyy, Nxy, Mxx, Myy, Mxy] N/m and N-m/m
    stress_resultant = np.array([[100], [100], [0], [1], [1], [0]])

    # Enter a desired ply orientation angles in degrees here:
    angle_in_degrees = [0, 0, 45, 45, -45, -45, 90]

    N = len(angle_in_degrees)  # total number of plies
    t_ply = 0.1 * (10**(-3))  # ply thickness in m
    h = t_ply * N

    # Distance from laminate mid-plane to out surfaces of plies)
    z0 = -h / 2
    z = []
    for i in range(N):
        z.append((-h / 2) + ((i + 1) * t_ply))

    # Distance from laminate mid-plane to mid-planes of plies
    z_mid_plane = []
    for i in range(N):
        z_mid_plane.append((-h / 2) - (t_ply / 2) + ((i + 1) * t_ply))

    # Ply orientation angle translated to radians to simplify equations below
    angle = []
    for i in range(N):
        angle.append(math.radians(angle_in_degrees[i]))

    # Stress Transformation (Global to Local)
    T = []
    for i in range(N):
        T.append(np.array([[c(angle[i]) ** 2, s(angle[i]) ** 2, 2 * s(angle[i]) * c(angle[i])],
                         [s(angle[i]) ** 2, c(angle[i]) ** 2, -2 * s(angle[i]) * c(angle[i])],
                         [-s(angle[i]) * c(angle[i]), s(angle[i]) * c(angle[i]), c(angle[i]) ** 2 - s(angle[i]) ** 2]]))

    # Strain Transformation (Global-to-Local)
    T_hat = []
    for i in range(N):
        T_hat.append(np.array([[c(angle[i]) ** 2, s(angle[i]) ** 2, s(angle[i]) * c(angle[i])],
                             [s(angle[i]) ** 2, c(angle[i]) ** 2, -s(angle[i]) * c(angle[i])],
                [-2 * s(angle[i]) * c(angle[i]), 2 * s(angle[i]) * c(angle[i]), c(angle[i]) ** 2 - s(angle[i]) ** 2]]))

    # Calculating the local (Q) and global (Q_bar) stiffness matrices
    Q_array, Q_bar_array = calculate_Q_and_Q_bar(E11, E22, G12, V12, V21, N, T, T_hat)

    A_array = calculate_A(N, Q_bar_array, t_ply)
    B_array = calculate_B(N, Q_bar_array, z, t_ply)
    D_array = calculate_D(N, Q_bar_array, z, t_ply)

    ABD_array = calculate_ABD(A_array, B_array, D_array)
    ABD_inverse_array = lg.inv(ABD_array)

    # Calculating the mid-plane strains and curvatures
    mid_plane_strains_and_curvatures_array = mm(lg.inv(ABD_array), stress_resultant)

    # Transforming numpy array into lists for ease of formatting
    Q = Q_array.tolist()
    Q_bar = []
    for i in range(N):
        Q_bar.append(Q_bar_array[i].tolist())
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
    global_strains = calculate_global_strains(N, mid_plane_strains, z_mid_plane, curvatures)

    # Global Stresses at mid-plane of each ply
    global_stresses = calculate_global_stresses(N, Q_bar, global_strains)

    # Local strains at mid-plane of each ply
    local_strains = calculate_local_strains(N, T_hat, global_strains)

    # Local stresses at mid-plane of each ply
    local_stresses = calculate_local_stresses(N, Q, local_strains)

    # Strength ratios for max stress failure criterion
    R_MS = R_Max_Stress(N, SLt, STt, SLTs, local_stresses)

    # Max stress failure criterion critical loads
    N_MSc, M_MSc = R_MS_Critical_Loads(N, R_MS, stress_resultant)

    # Define Tsai-Wu quadratic function coefficients (aR^2 + bR + cc = 0)
    a, b, cc = Tsai_Wu_coeffs(N, SLt, SLc, STt, STc, SLTs, local_stresses)

    # Strength ratios for Tsai-Wu criterion
    R_TW = R_Tsai_Wu(N, a, b, cc)





    # # Tsai-Wu critical loads
    # N_TW_xxc = float(min(R_TW) * stress_resultant[0])
    #
    # # Calculating E_xx
    # E_xx = (A[0][0] / h) * (1 - ((A[0][1] ** 2) / (A[0][0] * A[1][1])))
    #
    # print(format('Material Properties','^100s'))
    # print('Material:  T300/5208 carbon-epoxy (in SI units)')
    # print(format('E11 = '+format(E11,'^4.2e')+' Pa','^20s')  + format('E22 = '+format(E22,'^4.2e')+' Pa','^20s') +
    #       format('G12 = '+format(G12,'^4.2e')+' Pa','^20s'))
    #
    # print("\n# of ply that fails first: " + str(R_TW.index(min(R_TW)) + 1))
    #
    # # Printing the Strength Ratio for Tsai-Wu Failure
    # print("\nThis is the Strength Ratio for the first ply failure under Tsai-Wu Failure Criterion:")
    # print("R_TW = " + str(np.round(R_TW[0], 3)))


main()