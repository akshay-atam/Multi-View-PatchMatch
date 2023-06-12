import numpy as np
import cv2
import os
from tqdm import tqdm

def get_values(path, filename):
    with open(os.path.join(path, filename), "r") as f:
        contents = f.read()
        elements = contents.split()
        K = np.array([float(x) for x in elements[0:9]]).reshape((3,3))
        #K[0:2] = np.multiply(K[0:2], scale_factor)
        R = np.array([float(x) for x in elements[12:21]]).reshape((3,3))
        t = np.array([float(x) for x in elements[21:24]]).reshape((3,1))

    return K, R, t

def get_P(path, filename):
    with open(os.path.join(path, filename), "r") as f:
        contents = f.read()
        elements = contents.split()
        P = np.array([float(x) for x in elements[0:12]]).reshape((3,4))
    
    return P

def SAD(ref1, ref2):
    return np.sum(np.absolute(np.subtract(ref1, ref2)))

def SSD(ref1, ref2):
    return np.sum(np.square(ref1 - ref2))

def NCC(ref1, ref2):
    ref1_mean = np.mean(ref1)
    ref2_mean = np.mean(ref2)
    ref1_std = np.std(ref1)
    ref2_std = np.std(ref2)

    if ref1_std == 0 or ref2_std == 0:
        return np.inf

    return np.sum(((ref1 - ref1_mean) * (ref2 - ref2_mean)) / (ref1_std * ref2_std))

def to_3d(pos, z, K, R, T):
    temp_x = np.dot(np.linalg.inv(K), np.array([pos[0], pos[1], 1])).reshape(3,1)*z
    shifted = temp_x - T
    ret = np.dot(R, shifted)
    return np.array([ret[0, 0], ret[1, 0], ret[2, 0], 1])

def to_2d(pos, P):
    ret = np.matmul(P,pos)
    return ret/ret[2]

def project(loc, depth, K0, R0, T0, P1):
    point_3d = to_3d(loc, depth, K0, R0, T0)
    point_reprojected = to_2d(point_3d, P1)
    return np.array([point_reprojected[1], point_reprojected[0]])

def photoconsistency_measure(original_pos, ref_image, other_img, positions, window_size):
    scores = []
    for pos in positions:
        if ((pos is not None) and 
            (int(pos[0]) > (window_size//2)) and 
            (int(pos[1]) > (window_size//2)) and 
            (int(pos[0]) < ref_image.shape[0] - (window_size//2)) and 
            (int(pos[1]) < ref_image.shape[1] - (window_size//2)) and 
            (original_pos[0] > (window_size//2)) and 
            (original_pos[1] > (window_size//2)) and 
            (original_pos[0] < ref_image.shape[0] - (window_size//2)) and 
            (original_pos[1] < ref_image.shape[1] - (window_size//2))):

            l1 = int(pos[0]) - (window_size//2)
            h1 = int(pos[0]) + (window_size//2) + 1
            l2 = int(pos[1]) - (window_size//2)
            h2 = int(pos[1]) + (window_size//2) + 1
            try:
                scores.append(NCC(ref_image[original_pos[0] - (window_size//2):original_pos[0] + 1 + (window_size//2), 
                                            original_pos[1] - (window_size//2):original_pos[1] + 1 + (window_size//2), :], 
                                            other_img[l1:h1, l2:h2, :]))
            except Exception as e:
                print("NCC calculation exception:", e)
                print("Reference position:", original_pos)
                print("Other position:", pos)
        else:
            scores.append(np.inf)

    # if scores list empty
    if not scores:
        return None

    return np.argmin(np.array(scores))

if __name__ == "__main__":
    # fountain 4, 5, 6
    # entry 2, 5, 6
    # herz jesu 5, 6, 7
    # reshape image as well, P matrix will change
    PATH = "data/fountain-P11/"

    # scale factor
    sf = 0.25

    img_ref = cv2.resize(cv2.imread(os.path.join(PATH, "images/0005.png")), None, fx=sf, fy=sf)
    K_ref, R_ref, t_ref = get_values(PATH, "cameras/0005.png.camera")
    K_ref[0:2] = np.multiply(K_ref[0:2], sf)
    P_ref = get_P(PATH, "p/0005.png.P")
    P_ref[0:2] = np.multiply(P_ref[0:2], sf)

    img1 = cv2.resize(cv2.imread(os.path.join(PATH, "images/0004.png")), None, fx=sf, fy=sf)
    K1, R1, t1 = get_values(PATH, "cameras/0004.png.camera")
    K1[0:2] = np.multiply(K1[0:2], sf)
    P1 = get_P(PATH, "p/0004.png.P")
    P1[0:2] = np.multiply(P1[0:2], sf)

    img2 = cv2.resize(cv2.imread(os.path.join(PATH, "images/0006.png")), None, fx=sf, fy=sf)
    K2, R2, t2 = get_values(PATH, "cameras/0006.png.camera")
    K2[0:2] = np.multiply(K2[0:2], sf)
    P2 = get_P(PATH, "p/0006.png.P")
    P2[0:2] = np.multiply(P2[0:2], sf)

    # depth upper and lower limit
    lower = 10000
    upper = 100000

    # generate random depth map
    depthmap = (np.random.rand(img_ref.shape[0], img_ref.shape[1]) + lower) * upper

    # scan
    scan = ['RL', 'LR', 'UD', 'DU']

    # window size
    window = 15

    # number of patchmatch iterations
    num_iters = 10

    for iters in range(num_iters):
        # loop through depth map and check direction
        for i in tqdm(range(depthmap.shape[0])):
            for j in range(depthmap.shape[1]):
                direction = scan[iters % 4]
                
                # find neighbor
                if direction == 'UD':
                    if i != 0:
                        n = depthmap[i-1, j]
                    else:
                        continue
                elif direction == 'DU':
                    if i != depthmap.shape[0] - 1:
                        n = depthmap[i+1, j]
                    else:
                        continue
                elif direction == 'LR':
                    if j != 0:
                        n = depthmap[i, j-1]
                    else:
                        continue
                elif direction == 'RL':
                    if j != depthmap.shape[1] - 1:
                        n = depthmap[i, j+1]
                    else:
                        continue
                else:
                    raise "Invalid scan direction"
                
                rand_val = np.random.uniform(low=lower, high=upper)

                # project into other image 1 with all depth hypothesis
                neighbor_pos = project(np.array([j,i,1]), n, K_ref, R_ref, t_ref, P1)
                rand_pos = project(np.array([j,i,1]), rand_val, K_ref, R_ref, t_ref, P1)
                original_pos = project(np.array([j,i,1]), depthmap[i,j], K_ref, R_ref, t_ref, P1)

                # project into other image 2 with all depth hypothesis
                neighbor_pos2 = project(np.array([j,i,1]), n, K_ref, R_ref, t_ref, P2)
                rand_pos2 = project(np.array([j,i,1]), rand_val, K_ref, R_ref, t_ref, P2)
                original_pos2 = project(np.array([j,i,1]), depthmap[i,j], K_ref, R_ref, t_ref, P2)

                # evaluate photoconsistency
                best_depth = photoconsistency_measure((i,j), img_ref, img1, 
                                                   [neighbor_pos, rand_pos, original_pos, neighbor_pos2, rand_pos2, original_pos2], 
                                                   window)

                # keep best option
                if (best_depth == 0) or (best_depth == 3):
                    depthmap[i,j] = n
                elif (best_depth == 1) or (best_depth == 4):
                    depthmap[i,j] = rand_val

            # randomly propagate depth values in the opposite scan direction
            if iters % 2 == 0:
                depthmap[i, :] = np.roll(depthmap[i, :], np.random.randint(1, depthmap.shape[1]))
            else:
                depthmap[i, :] = np.roll(depthmap[i, :], -np.random.randint(1, depthmap.shape[1]))
                
        np.save("depthmaps/fountain/depth_{}.npy".format(iters), depthmap)
