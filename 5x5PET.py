from numpy import *
import matplotlib.pyplot as plt
from PET_configurations import A_full
from patterns import *


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
    for i in range(25):
        vox.append(events(voxdata[i]))
    vox = array(vox)
    # lors numbered counterclockwise, starting from up-right
    lors = list()
    i = 0
    while i < 5:
        lors.append(lc(vox[i], 2) + lc(vox[i + 5], 2) + lc(vox[i + 10], 2) + lc(vox[i + 15], 2) + lc(vox[i + 20], 2))
        i += 1
    lors.append(lc(vox[2], 1) + lc(vox[6], 1) + lc(vox[10], 1))
    lors.append(lc(vox[3], 1) + lc(vox[7], 1) + lc(vox[11], 1) + lc(vox[15], 1))
    lors.append(lc(vox[4], 1) + lc(vox[8], 1) + lc(vox[12], 1) + lc(vox[16], 1) + lc(vox[20], 1))
    lors.append(lc(vox[9], 1) + lc(vox[13], 1) + lc(vox[17], 1) + lc(vox[21], 1))
    lors.append(lc(vox[14], 1) + lc(vox[18], 1) + lc(vox[21], 1))
    i = 10
    while i < 15:
        lors.append(lc(vox[i], 0) + lc(vox[i + 1], 0) + lc(vox[i + 2], 0) + lc(vox[i + 3], 0) + lc(vox[i + 4], 0))
        i +=1
    lors.append(lc(vox[2], 3) + lc(vox[8], 3) + lc(vox[14], 3))
    lors.append(lc(vox[1], 3) + lc(vox[7], 3) + lc(vox[13], 3) + lc(vox[18], 3))
    lors.append(lc(vox[0], 3) + lc(vox[6], 3) + lc(vox[12], 3) + lc(vox[17], 3) + lc(vox[24], 3))
    lors.append(lc(vox[5], 3) + lc(vox[11], 3) + lc(vox[16], 3) + lc(vox[23], 3))
    lors.append(lc(vox[10], 3) + lc(vox[15], 3) + lc(vox[22], 3))
    i=1
    for lor in lors:
        print("lor ", i, " = ", lor)
        i+=1
    return array(lors)


def sim_object(pattern=random.poisson(50, size=25)):
    rn = pattern
    obj = list()
    for i in range(25):
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
    pos = array([
                        [3, 3], [2, 3], [1, 3], [-1, 3], [-2, 3],
                        [3, 2], [2, 2], [1, 2], [-1, 2], [-2, 2],
                        [3, 1], [2, 1], [1, 1], [-1, 1], [-2, 1],
                        [3, -1], [2, -1], [1, -1], [-1, -1], [-2, -1],
                        [3, -2], [2, -2], [1, -2], [-1, -2], [-2, -2],
                        ])
    data = list()
    for i in range(img.shape[-1]):
        nPos = int(img[i])
        if nPos < 0:
            nPos = 0
        for j in range(nPos):
            data.append(pos[i])
    data = array(data)
    return data


# pattern selection for simulation and reconstruction
act = 1000  # material rad activity
phantom = sim_object(pattern=act*center_55)


data_full = scanner_full(phantom)
data_full += sim_noise(1, data_full.shape[0])


Ainv_full = linalg.pinv(A_full)
image_full = dot(Ainv_full, data_full)


# print_info('Phantom', phantom, phantom)
# print_info('Image Full PET', image_full, phantom)


phantom_image = to_color(phantom)
image_full_plot = to_color(image_full)
plt.hist2d(phantom_image[:, 0], phantom_image[:, 1], bins=[[-2,-1,1,2,3],[-2,-1,1,2,3]])
plt.colorbar()
plt.show()
plt.hist2d(image_full_plot[:, 0], image_full_plot[:, 1], bins=[[-2,-1,1,2,3],[-2,-1,1,2,3]])
plt.colorbar()
plt.show()