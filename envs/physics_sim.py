"""This file is adapted from the 3-body gravitational physics simulation in
Jaques et al., Physics-as-Inverse-Graphics: Joint Unsupervised Learning of
Objects and Physics from Video (https://arxiv.org/abs/1905.11169).

The original code, on which this file is based on is available under:
https://github.com/seuqaj114/paig/blob/master/nn/datasets/generators.py
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape

    bordered = 0.5*np.ones([nindex, height+2, width+2, intensity])
    for i in range(nindex):
        bordered[i,1:-1,1:-1,:] = array[i]

    array = bordered
    nindex, height, width, intensity = array.shape

    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result


def compute_wall_collision(pos, vel, radius, img_size):
    if pos[1] - radius <= 0:
        vel[1] = -vel[1]
        pos[1] = -(pos[1] - radius) + radius
    if pos[1] + radius >= img_size[1]:
        vel[1] = -vel[1]
        pos[1] = img_size[1] - (pos[1] + radius - img_size[1]) - radius
    if pos[0] - radius <= 0:
        vel[0] = -vel[0]
        pos[0] = -(pos[0] - radius) + radius
    if pos[0] + radius >= img_size[0]:
        vel[0] = -vel[0]
        pos[0] = img_size[0] - (pos[0] + radius - img_size[0]) - radius
    return pos, vel


def verify_wall_collision(pos, vel, radius, img_size):
    if pos[1] - radius <= 0:
        return True
    if pos[1] + radius >= img_size[1]:
        return True
    if pos[0] - radius <= 0:
        return True
    if pos[0] + radius >= img_size[0]:
        return True
    return False


def verify_object_collision(poss, radius):
    for pos1, pos2 in combinations(poss, 2):
        if np.linalg.norm(pos1 - pos2) <= radius:
            return True
    return False


def generate_3_body_problem_dataset(dest,
                                    train_set_size,
                                    valid_set_size,
                                    test_set_size,
                                    seq_len,
                                    img_size=None,
                                    radius=3,
                                    dt=0.3,
                                    g=9.8,
                                    m=1.0,
                                    vx0_max=0.0,
                                    vy0_max=0.0,
                                    color=False,
                                    cifar_background=False,
                                    ode_steps=10,
                                    seed=0):
    np.random.seed(seed)

    if cifar_background:
        import tensorflow as tf
        (x_train, y_train), (
        x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    from skimage.draw import circle
    from skimage.transform import resize

    if img_size is None:
        img_size = [32, 32]
    scale = 10
    scaled_img_size = [img_size[0] * scale, img_size[1] * scale]

    def generate_sequence():
        # sample initial position of the center of mass, then sample
        # position of each object relative to that.

        collision = True
        while collision == True:
            seq = []

            cm_pos = np.random.rand(2)
            cm_pos = np.array(img_size) / 2
            angle1 = np.random.rand() * 2 * np.pi
            angle2 = angle1 + 2 * np.pi / 3 + (np.random.rand() - 0.5) / 2
            angle3 = angle1 + 4 * np.pi / 3 + (np.random.rand() - 0.5) / 2

            angles = [angle1, angle2, angle3]
            # calculate position of both objects
            r = (np.random.rand() / 2 + 0.75) * img_size[0] / 4
            poss = [
                [np.cos(angle) * r + cm_pos[0], np.sin(angle) * r + cm_pos[1]]
                for angle in angles]
            poss = np.array(poss)

            # angles = np.random.rand(3)*2*np.pi
            # vels = [[np.cos(angle)*vx0_max, np.sin(angle)*vy0_max] for angle in angles]
            # vels = np.array(vels)
            r = np.random.randint(0, 2) * 2 - 1
            angles = [angle + r * np.pi / 2 for angle in angles]
            noise = np.random.rand(2) - 0.5
            vels = [[np.cos(angle) * vx0_max + noise[0],
                     np.sin(angle) * vy0_max + noise[1]] for angle in angles]
            vels = np.array(vels)

            if cifar_background:
                cifar_img = x_train[np.random.randint(50000)]

            for i in range(seq_len):
                if cifar_background:
                    frame = cifar_img
                    frame = rgb2gray(frame) / 255
                    frame = resize(frame, scaled_img_size)
                    frame = np.clip(frame - 0.2, 0.0, 1.0)  # darken image a bit
                else:
                    if color:
                        frame = np.zeros(scaled_img_size + [3],
                                         dtype=np.float32)
                    else:
                        frame = np.zeros(scaled_img_size + [1],
                                         dtype=np.float32)

                for j, pos in enumerate(poss):
                    rr, cc = circle(int(pos[1] * scale), int(pos[0] * scale),
                                    radius * scale, scaled_img_size)
                    if color:
                        frame[rr, cc, 2 - j] = 1.0
                    else:
                        frame[rr, cc, 0] = 1.0

                frame = resize(frame, img_size, anti_aliasing=True)
                frame = (frame * 255).astype(np.uint8)

                seq.append(frame)

                # rollout physics
                for _ in range(ode_steps):
                    norm01 = np.linalg.norm(poss[0] - poss[1])
                    norm12 = np.linalg.norm(poss[1] - poss[2])
                    norm20 = np.linalg.norm(poss[2] - poss[0])
                    vec01 = (poss[0] - poss[1])
                    vec12 = (poss[1] - poss[2])
                    vec20 = (poss[2] - poss[0])

                    # Compute force vectors
                    F = [vec01 / norm01 ** 3 - vec20 / norm20 ** 3,
                         vec12 / norm12 ** 3 - vec01 / norm01 ** 3,
                         vec20 / norm20 ** 3 - vec12 / norm12 ** 3]
                    F = np.array(F)
                    F = -g * m * m * F

                    vels = vels + dt / ode_steps * F
                    poss = poss + dt / ode_steps * vels

                    collision = any(
                        [verify_wall_collision(pos, vel, radius, img_size) for
                         pos, vel in zip(poss, vels)]) or \
                                verify_object_collision(poss, radius + 1)
                    if collision:
                        break

                if collision:
                    break

        return seq

    sequences = []
    for i in range(train_set_size + valid_set_size + test_set_size):
        if i % 100 == 0:
            print(i)
        sequences.append(generate_sequence())
    sequences = np.array(sequences, dtype=np.uint8)

    np.savez_compressed(dest,
                        train_x=sequences[:train_set_size],
                        valid_x=sequences[
                                train_set_size:train_set_size + valid_set_size],
                        test_x=sequences[train_set_size + valid_set_size:])
    print("Saved to file %s" % dest)

    # Save 10 samples
    result = gallery(np.concatenate(sequences[:10] / 255),
                     ncols=sequences.shape[1])

    norm = plt.Normalize(0.0, 1.0)
    fig, ax = plt.subplots(figsize=(sequences.shape[1], 10))
    ax.imshow(np.squeeze(result), interpolation='nearest', cmap=cm.Greys_r,
              norm=norm)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    fig.savefig(dest.split(".")[0] + "_samples.jpg")
