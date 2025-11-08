import numpy as np
import pandas as pd

def distance(i1, v1, i2, v2):
    return np.sqrt((v1-v2)**2 + (i1-i2)**2)

def interpolate_points(v1, i1, v2, i2, num_points):
    return np.linspace(v1, v2, num_points), np.linspace(i1, i2, num_points)

def I_V_image_ver_2(In, Un, I_max_fin, point_per_cycle = 10, N = 16, cycles = 10, speed = 1):
    """_summary_

    Args:
        In (1 dimension array): This is the In array
        Un (1 dimension array): This is the Un array
        I_max_fin (Float): A const that need to import some In value to the output mmatrix
        point_per_cycle (int, optional): Point of 1 full cycle. Defaults to 10.
        N (int, optional): size of image. Defaults to 16.
        cycles (int, optional): how many cycle for 1 image. Defaults to 10.
        speed (int, optional): how fast you want to calculate the image. Defaults to 1.

    Returns:
        _type_: flatten image matrix
    """
    matrix_all = np.zeros((2*N+1, 2*N+1), dtype=float)
    I_max = max(abs(In))
    V_max = max(abs(Un))
    delta_I = I_max/N
    delta_V = V_max/N
    for cycle in range(cycles):
        start_idx = 0 + cycle*point_per_cycle
        end_idx = start_idx + point_per_cycle
        I_t = np.array(In[start_idx: end_idx])
        V_t = np.array(Un[start_idx: end_idx])

        matrix = np.zeros((2*N+1, 2*N+1), dtype=np.int16)
        for k in range(0,max(len(V_t),len(I_t))-speed-1,speed):
            v1, i1 = V_t[k], I_t[k]
            v2, i2 = V_t[k+speed], I_t[k+speed]

            v1 = int(np.round(v1/delta_V)+N)
            v2 = int(np.round(v2/delta_V)+N)
            i1 = int(np.round(i1/delta_I)+N)
            i2 = int(np.round(i2/delta_I)+N)

            Dk = distance(i1, v1, i2, v2)
            if Dk > 1:
                num_interp_points = int(np.ceil(Dk))
                v_interp, i_interp = interpolate_points(v1, i1, v2, i2, num_interp_points)
            else:
                v_interp, i_interp = np.array([v1, v2]), np.array([i1, i2])
                        
            for v, i in zip(v_interp, i_interp):
                v = int(np.round(v))
                i = int(np.round(i))
                matrix[i, v] = 255 - np.round(I_t[k]/I_max_fin*255)
            matrix[i1, v1] = 255 - np.round(I_t[k]/I_max_fin*255)
            matrix[i2, v2] = 255 - np.round(I_t[k+speed]/I_max_fin*255)
        matrix_all+=matrix
    matrix_all = np.round(matrix_all/cycles)
    matrix_all = matrix_all.astype(np.int16)
    matrix_flatten = matrix_all.flatten()
    return matrix_flatten

def I_V_image_ver_1(In, Un, point_per_cycle = 10, N = 16, cycles = 10, speed = 1):
    """_summary_

    Args:
        In (1 dimension array): This is the In array
        Un (1 dimension array): This is the Un array
        point_per_cycle (int, optional): Point of 1 full cycle. Defaults to 10.
        N (int, optional): size of image. Defaults to 16.
        cycles (int, optional): how many cycle for 1 image. Defaults to 10.
        speed (int, optional): how fast you want to calculate the image. Defaults to 1.

    Returns:
        _type_: flatten image matrix
    """
    matrix_all = np.zeros((2*N+1, 2*N+1), dtype=float)
    I_max = max(abs(In))
    V_max = max(abs(Un))
    delta_I = I_max/N
    delta_V = V_max/N
    for cycle in range(cycles):
        start_idx = 0 + cycle*point_per_cycle
        end_idx = start_idx + point_per_cycle
        I_t = np.array(In[start_idx: end_idx])
        V_t = np.array(Un[start_idx: end_idx])

        matrix = np.zeros((2*N+1, 2*N+1), dtype=np.int16)
        for k in range(0,max(len(V_t),len(I_t))-speed,speed):
            v1, i1 = V_t[k], I_t[k]
            v2, i2 = V_t[k+speed], I_t[k+speed]

            v1 = int(np.round(v1/delta_V)+N)
            v2 = int(np.round(v2/delta_V)+N)
            i1 = int(np.round(i1/delta_I)+N) 
            i2 = int(np.round(i2/delta_I)+N) 

            Dk = distance(i1, v1, i2, v2)
            if Dk > 1:
                num_interp_points = int(np.ceil(Dk))
                v_interp, i_interp = interpolate_points(v1, i1, v2, i2, num_interp_points)
            else:
                v_interp, i_interp = np.array([v1, v2]), np.array([i1, i2])
                        
            for v, i in zip(v_interp, i_interp):
                v = int(np.round(v))
                i = int(np.round(i))
                matrix[i, v] = 255
        matrix_all+=matrix
    matrix_all = np.round(matrix_all/cycles)
    matrix_all = matrix_all.astype(np.int16)
    matrix_flatten = matrix_all.flatten()
    return matrix_flatten

def I_V_image_single_cycle(In, Un, N = 16, speed = 1):
    I_t = np.array(In)
    V_t = np.array(Un)
    matrix = np.zeros((2*N+1, 2*N+1), dtype=float)
    I_max = max(abs(I_t))
    V_max = max(abs(V_t))
    delta_I = I_max/N
    delta_V = V_max/N
    for k in range(0,max(len(V_t),len(I_t))-speed,speed):
        v1, i1 = V_t[k], I_t[k]
        v2, i2 = V_t[k+speed], I_t[k+speed]

        v1 = int(np.round(v1/delta_V)+N) if V_max!=0 else 0
        v2 = int(np.round(v2/delta_V)+N) if V_max!=0 else 0
        i1 = int(np.round(i1/delta_I)+N) if I_max!=0 else 0
        i2 = int(np.round(i2/delta_I)+N) if I_max!=0 else 0

        Dk = distance(i1, v1, i2, v2)
        if Dk > 1:
            num_interp_points = int(np.ceil(Dk))
            v_interp, i_interp = interpolate_points(v1, i1, v2, i2, num_interp_points)
        else:
            v_interp, i_interp = np.array([v1, v2]), np.array([i1, i2])
                        
        for v, i in zip(v_interp, i_interp):
            v = int(np.round(v))
            i = int(np.round(i))
            matrix[i, v] = 255
    matrix = matrix.astype(np.int16)
    return matrix

def I_V_image_single_cycle_2(In, Un, N = 16, speed = 1):
    I_t = np.array(In)
    V_t = np.array(Un)
    matrix = np.zeros((2*N+1, 2*N+1), dtype=float)
    I_max = max(abs(I_t))
    V_max = max(abs(V_t))
    delta_I = I_max/N
    delta_V = V_max/N
    for k in range(0,max(len(V_t),len(I_t))-speed,speed):
        v1, i1 = V_t[k], I_t[k]
        v2, i2 = V_t[k+speed], I_t[k+speed]

        v1 = int(np.round(v1/delta_V)+N) if V_max!=0 else 0
        v2 = int(np.round(v2/delta_V)+N) if V_max!=0 else 0
        i1 = int(np.round(i1/delta_I)+N) if I_max!=0 else 0
        i2 = int(np.round(i2/delta_I)+N) if I_max!=0 else 0

        Dk = distance(i1, v1, i2, v2)
        if Dk > 1:
            num_interp_points = int(np.ceil(Dk))
            v_interp, i_interp = interpolate_points(v1, i1, v2, i2, num_interp_points)
        else:
            v_interp, i_interp = np.array([v1, v2]), np.array([i1, i2])
                        
        for v, i in zip(v_interp, i_interp):
            v = int(np.round(v))
            i = int(np.round(i))
            matrix[i, v] = 255
    matrix = matrix.astype(np.int16)
    for e in range(4):
        for i in range(2*N+1):
            for j in range(2*N+1):
                neighrbor = []
                if matrix[i, j] == 0:
                    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                        if 0 <= i + dx < 2*N + 1 and 0 <= j + dy < 2*N + 1:
                            neighrbor.append(matrix[i + dx, j + dy])
                    matrix[i, j] = np.round((2/(e+2) * np.mean(neighrbor) if neighrbor else matrix[i, j]))     
    return matrix

def I_V_image_ver_3(In, Un, I_max_fin, point_per_cycle = 10, N = 16, cycles = 10, speed = 1):
    """_summary_

    Args:
        In (1 dimension array): This is the In array
        Un (1 dimension array): This is the Un array
        I_max_fin (Float): A const that need to import some In value to the output mmatrix
        point_per_cycle (int, optional): Point of 1 full cycle. Defaults to 10.
        N (int, optional): size of image. Defaults to 16.
        cycles (int, optional): how many cycle for 1 image. Defaults to 10.
        speed (int, optional): how fast you want to calculate the image. Defaults to 1.

    Returns:
        _type_: flatten image matrix
    """
    matrix_all = np.zeros((2*N+1, 2*N+1), dtype=float)
    I_max = max(abs(In))
    V_max = max(abs(Un))
    delta_I = I_max/N
    delta_V = V_max/N
    for cycle in range(cycles):
        start_idx = 0 + cycle*point_per_cycle
        end_idx = start_idx + point_per_cycle
        I_t = np.array(In[start_idx: end_idx])
        V_t = np.array(Un[start_idx: end_idx])

        matrix = np.zeros((2*N+1, 2*N+1), dtype=np.int16)
        for k in range(0,max(len(V_t),len(I_t))-speed,speed):
            v1, i1 = V_t[k], I_t[k]
            v2, i2 = V_t[k+speed], I_t[k+speed]

            v1 = int(np.round(v1/delta_V)+N)
            v2 = int(np.round(v2/delta_V)+N)
            i1 = int(np.round(i1/delta_I)+N)
            i2 = int(np.round(i2/delta_I)+N)

            Dk = distance(i1, v1, i2, v2)
            if Dk > 1:
                num_interp_points = int(np.ceil(Dk))
                v_interp, i_interp = interpolate_points(v1, i1, v2, i2, num_interp_points)
            else:
                v_interp, i_interp = np.array([v1, v2]), np.array([i1, i2])
                        
            for v, i in zip(v_interp, i_interp):
                v = int(np.round(v))
                i = int(np.round(i))
                matrix[i, v] = np.round((I_max/I_max_fin)*255)
        matrix_all+=matrix
    matrix_all = np.round(matrix_all/cycles)
    matrix_all = matrix_all.astype(np.int16)
    matrix_flatten = matrix_all.flatten()
    return matrix_flatten, matrix_all


def physical_feature_calculation(In, Un):
    Irms = np.sqrt(np.mean(In**2))
    Urms = np.sqrt(np.mean(Un**2))
    P = np.abs(np.mean(In * Un))
    S = Irms * Urms
    Q = np.sqrt(abs(S**2 - P**2))
    pf = P/S
    return Irms, Urms, P, S, Q, pf
