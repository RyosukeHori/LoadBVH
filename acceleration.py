import os
import sys
import csv
import pprint
import numpy as np
import argparse
import math
from bvh import Bvh
from skeleton import Skeleton
import glob
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import logging

sys.path.append(os.getcwd())


def compute_accel(joints):
    """
    Computes acceleration of 3D joints.
    Args:
        joints (Nx19x3).
    Returns:
        Accelerations (N-2).
    """
    velocities = joints[1:] - joints[:-1]
    acceleration = velocities[1:] - velocities[:-1]
    acceleration_normed = np.linalg.norm(acceleration, axis=2)
    return acceleration_normed

csv_files = glob.glob(os.path.expanduser('CSV/*.csv'))
csv_files.sort()
#file_path = 'CSV/0_6_1,000_1,000,000.csv'
for file_path in csv_files:
    if "accel" in file_path:
        continue
    print(file_path)
    dir_path = file_path[:-4]
    file_name = dir_path[4:]
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    missing_frame = []
    with open(file_path) as f:
        reader = csv.reader(f)
        joints = []
        times = []
        frame = 0
        for row in reader:
            joint_list = row[:-1]
            time = row[-1]
            if all(elem == '0' for elem in joint_list):
                missing_frame.append(frame)
                print('frame', frame, '(time:', time, ') missing!')
            joints.append([float(joint) for joint in joint_list])
            times.append(int(time))
            frame += 1
    joints = np.array(joints).reshape(-1, 18, 3)
    accel = compute_accel(joints)
    for miss in missing_frame:
        accel[miss - 2, :] = 0
        if miss >= len(accel):
            break
        accel[miss - 1, :] = 0
        accel[miss, :] = 0
    left = np.array(times[1:-1])

    with open(dir_path + '/' + file_name + "_accel.csv", 'w') as f:
        writer = csv.writer(f)
        for time, joint in zip(left, accel):
            joint = joint.tolist()
            joint.append(time)
            writer.writerow(joint)

    cm = plt.cm.get_cmap('tab20')
    for j_idx in range(accel.shape[1]):
        fig, ax = plt.subplots()
        ax.get_xaxis().get_major_formatter().set_useOffset(False)
        ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        ax.plot(left, accel[:, j_idx], color = cm.colors[j_idx])
        plt.setp(ax.get_xticklabels(), rotation=-30)
        plt.savefig(dir_path + '/bone_' + str(j_idx) + ".png")

    fig, ax = plt.subplots()
    ax.get_xaxis().get_major_formatter().set_useOffset(False)
    ax.get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    for j_idx in range(accel.shape[1]):
        ax.plot(left, accel[:, j_idx], color = cm.colors[j_idx])
    plt.setp(ax.get_xticklabels(), rotation=-30)
    plt.savefig(dir_path + '/' + file_name + ".png")

