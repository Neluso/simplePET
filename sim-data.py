from numpy import *


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
    # voxels numered counterclockwise
    vox1 = events(voxdata[0])
    vox2 = events(voxdata[1])
    vox3 = events(voxdata[2])
    vox4 = events(voxdata[3])
    # lors numered counterclockwise
    lor1 = lc(vox1, 2) + lc(vox4, 2)
    lor2 = lc(vox2, 2) + lc(vox3, 2)
    lor3 = lc(vox1, 3)
    lor4 = lc(vox2, 3) + lc(vox4, 3)
    lor5 = lc(vox3, 3)
    lor6 = lc(vox1, 0) + lc(vox2, 0)
    lor7 = lc(vox3, 0) + lc(vox4, 0)
    lor8 = lc(vox2, 1)
    lor9 = lc(vox1, 1) + lc(vox3, 1)
    lor10 = lc(vox4, 1)
    return array([lor1, lor2, lor3, lor4, lor5, lor6, lor7, lor8, lor9, lor10])


def scanner_opened(voxdata):
    # voxels numered counterclockwise
    vox1 = events(voxdata[0])
    vox2 = events(voxdata[1])
    vox3 = events(voxdata[2])
    vox4 = events(voxdata[3])
    # lors numered counterclockwise
    lor1 = lc(vox1, 2) + lc(vox4, 2)
    lor2 = lc(vox2, 2) + lc(vox3, 2)
    lor3 = lc(vox1, 3)
    lor4 = lc(vox2, 3) + lc(vox4, 3)
    lor5 = lc(vox3, 3)
    lor6 = lc(vox1, 0) + lc(vox2, 0)
    lor7 = lc(vox3, 0) + lc(vox4, 0)
    return array([lor1, lor2, lor3, lor4, lor5, lor6, lor7])


def scanner_cross(voxdata):
    # voxels numered counterclockwise
    vox1 = events(voxdata[0])
    vox2 = events(voxdata[1])
    vox3 = events(voxdata[2])
    vox4 = events(voxdata[3])
    # lors numered counterclockwise
    lor1 = lc(vox1, 2) + lc(vox4, 2)
    lor2 = lc(vox2, 2) + lc(vox3, 2)
    lor6 = lc(vox1, 0) + lc(vox2, 0)
    lor7 = lc(vox3, 0) + lc(vox4, 0)
    return array([lor1, lor2, lor6, lor7])


def sim_object(pattern = 80 * random.rand(4) + 20*ones(4)):
    rn = pattern
    obj = list()
    for i in range(4):
        obj.append(int(rn[i]))
    return array(obj)


def sim_noise(noise, lors):
    rn = (noise + 1) * random.rand(lors)
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


phantom = sim_object()
data_full = scanner_full(phantom)
data_opened = scanner_opened(phantom)
data_cross = scanner_cross(phantom)
data_full += sim_noise(2, data_full.shape[0])
data_opened += sim_noise(2, data_opened.shape[0])
data_cross += sim_noise(2, data_cross.shape[0])
A_full = 1 / 4 * array([[1, 0, 0, 1],
                  [0, 1, 0, 1],
                  [1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [1, 1, 0, 0],
                  [0, 0, 1, 1],
                  [0, 1, 0, 0],
                  [1, 0, 1, 0],
                  [0, 0, 0, 1]])
Ainv_full = linalg.pinv(A_full)
A_opened = 1 / 4 * array([[1, 0, 0, 1],
                  [0, 1, 0, 1],
                  [1, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 0],
                  [1, 1, 0, 0],
                  [0, 0, 1, 1]])
Ainv_opened = linalg.pinv(A_opened)
A_cross = 1 / 4 * array([[1, 0, 0, 1],
                  [0, 1, 0, 1],
                  [1, 1, 0, 0],
                  [0, 0, 1, 1]])
Ainv_cross = linalg.pinv(A_cross)
image_full = dot(Ainv_full, data_full)
image_opened = dot(Ainv_opened, data_opened)
image_cross = dot(Ainv_cross, data_cross)


print_info('Phantom', phantom, phantom)
print_info('Image Full PET', image_full, phantom)
print_info('Image Opened PET', image_opened, phantom)
print_info('Image Cross PET', image_cross, phantom)


