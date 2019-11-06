from numpy import *
import matplotlib.pyplot as plt


def events(nPhant):  # 0->0, 1->45, 2->90, 3->135
    rn = random.rand(nPhant)
    for i in range(nPhant):
        lor = rn[i]
        if lor < 1/4:
            rn[i] = 0
        elif lor < 2/4:
            rn[i] = 1
        elif lor < 3/4:
            rn[i] = 2
        else:
            rn[i] = 3
    return rn


def lc(vox, lor):
    return (vox == lor).sum()


def scanner_full(voxdata):
    # voxels: right -> left, top -> down
    vox = list()
    for i in range(16):
        vox.append(events(voxdata[i]))
    vox = array(vox)
    # lors numered counterclockwise, starting from up-right
    lors = list()
    i = 0
    while i < 4:
        lors.append(lc(vox[i], 0) + lc(vox[i + 4], 0) + lc(vox[i + 8], 0) + lc(vox[i + 12], 0))
        i += 1
    lors.append(lc(vox[1], 1) + lc(vox[4], 1))
    lors.append(lc(vox[3], 1) + lc(vox[6], 1) + lc(vox[9], 1) + lc(vox[12], 1))
    lors.append(lc(vox[11], 1) + lc(vox[14], 1))
    i = 7
    while i < 11:
        lors.append(lc(vox[i], 2) + lc(vox[i + 1], 2) + lc(vox[i + 2], 2) + lc(vox[i + 3], 2))
        i +=1
    lors.append(lc(vox[2], 3) + lc(vox[7], 3))
    lors.append(lc(vox[0], 3) + lc(vox[5], 3) + lc(vox[10], 3) + lc(vox[15], 3))
    lors.append(lc(vox[8], 3) + lc(vox[13], 3))
    return array(lors)


def scanner_opened(voxdata):
    # voxels: right -> left, top -> down
    vox = list()
    for i in range(16):
        vox.append(events(voxdata[i]))
    vox = array(vox)
    # lors numered counterclockwise, starting from up-right
    lors = list()
    i = 0
    while i < 4:
        lors.append(lc(vox[i], 0) + lc(vox[i + 4], 0) + lc(vox[i + 8], 0) + lc(vox[i + 12], 0))
        i += 1
    lors.append(lc(vox[1], 1) + lc(vox[4], 1))
    lors.append(lc(vox[3], 1) + lc(vox[6], 1) + lc(vox[9], 1) + lc(vox[12], 1))
    lors.append(lc(vox[11], 1) + lc(vox[14], 1))
    i = 7
    while i < 11:
        lors.append(lc(vox[i], 2) + lc(vox[i + 1], 2) + lc(vox[i + 2], 2) + lc(vox[i + 3], 2))
        i += 1
    return array(lors)


def scanner_cross(voxdata):
    # voxels: right -> left, top -> down
    vox = list()
    for i in range(16):
        vox.append(events(voxdata[i]))
    vox = array(vox)
    # lors numered counterclockwise, starting from up-right
    lors = list()
    i = 0
    while i < 4:
        lors.append(lc(vox[i], 0) + lc(vox[i + 4], 0) + lc(vox[i + 8], 0) + lc(vox[i + 12], 0))
        i += 1
    i = 7
    while i < 11:
        lors.append(lc(vox[i], 2) + lc(vox[i + 1], 2) + lc(vox[i + 2], 2) + lc(vox[i + 3], 2))
        i += 1
    return array(lors)


def sim_object(pattern=random.poisson(50, size=16)):
    rn = pattern
    obj = list()
    for i in range(16):
        obj.append(int(rn[i]))
    return array(obj)


def sim_noise(noise, lors):
    rn = random.poisson(noise, size=lors)
    obj = list()
    for i in range(lors):
        obj.append(int(rn[i]))
    return array(obj)


def print_info(title, image, phant):
    print(title)
    print('Counts:\t', image)
    print('Intensity:\t', image / sum(image))
    print('Normed intensity:\t', image / max(image))
    print('Relative voxel error:\t', 100 * abs(image - phant) / sum(phant), '%')
    err = 100 * linalg.norm(image - phant) / sum(phant)
    err = round(err, 1)
    print('RMS Error:\t', err, '%')
    print()


def to_color(img):
    pos = 0.95 * array([[2, 2], [1, 2], [-1, 2], [-2, 2],
                 [2, 1], [1, 1], [-1, 1], [-2, 1],
                 [2, -1], [1, -1], [-1, -1], [-2, -1],
                 [2, -2], [1, -2], [-1, -2], [-2, -2]])
    data = list()
    for i in range(img.shape[0]):
        nPos = int(img[i])
        if nPos < 0:
            nPos = 0
        for j in range(nPos):
            data.append(pos[i])
    data = array(data)
    return data


# pattern selection for simulation and reconstruction
act = 10000  # material rad activity
center_pattern = array([0,0,0,0,0,act,act,0,0,act,act,0,0,0,0,0])
edges_pattern = array([act,0,0,act,0,0,0,0,0,0,0,0,act,0,0,act])
cross_pattern = center_pattern + edges_pattern
line_pattern = array([0,0,0,0,act,act,act,act,0,0,0,0,0,0,0,0])
chess_pattern = array([act,0,act,0,0,act,0,act,act,0,act,0,0,act,0,act])
custom_pattern = array([act,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
phantom = sim_object(pattern=chess_pattern)


data_full = scanner_full(phantom)
data_opened = scanner_opened(phantom)
data_cross = scanner_cross(phantom)
data_full += sim_noise(1, data_full.shape[0])
data_opened += sim_noise(1, data_opened.shape[0])
data_cross += sim_noise(1, data_cross.shape[0])


A_full = 1 / 4 * array([[1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                        [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
                        [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
                        [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
                        [0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],
                        [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
                        [0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0],
                        [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],
                        [0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0]])
Ainv_full = linalg.pinv(A_full)
image_full = dot(Ainv_full, data_full)


A_opened = 1 / 4 * array([[1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                        [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
                        [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
                        [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
                        [0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0],
                        [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]])
Ainv_opened = linalg.pinv(A_opened)
image_opened = dot(Ainv_opened, data_opened)


A_cross = 1 / 4 * array([[1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0],
                        [0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0],
                        [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0],
                        [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
                        [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
                        [0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0],
                        [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0],
                        [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1]])
Ainv_cross = linalg.pinv(A_cross)
image_cross = dot(Ainv_cross, data_cross)


print_info('Phantom', phantom, phantom)
print_info('Image Full PET', image_full, phantom)
print_info('Image Opened PET', image_opened, phantom)
print_info('Image Cross PET', image_cross, phantom)


phantom_image = to_color(phantom)
image_full_plot = to_color(image_full)
image_opened_plot = to_color(image_opened)
image_cross_plot = to_color(image_cross)
plt.hist2d(phantom_image[:, 0], phantom_image[:, 1], bins=[[-2,-1,0,1,2],[-2,-1,0,1,2]])
plt.colorbar()
plt.show()
plt.hist2d(image_full_plot[:, 0], image_full_plot[:, 1], bins=[[-2,-1,0,1,2],[-2,-1,0,1,2]])
plt.colorbar()
plt.show()
plt.hist2d(image_opened_plot[:, 0], image_opened_plot[:, 1], bins=[[-2,-1,0,1,2],[-2,-1,0,1,2]])
plt.colorbar()
plt.show()
plt.hist2d(image_cross_plot[:, 0], image_cross_plot[:, 1], bins=[[-2,-1,0,1,2],[-2,-1,0,1,2]])
plt.colorbar()
plt.show()
