import os
import sys
import numpy as np
import argparse
import math
from bvh import Bvh
from skeleton import Skeleton
import glob
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument('--mocap-fr', type=int, default=50)
parser.add_argument('--dt', type=float, default=1 / 50)
args = parser.parse_args()


def load_bvh_file(fname, skeleton):
    with open(fname) as f:
        mocap = Bvh(f.read())

    # build bone_addr
    bone_addr = dict()
    start_ind = 0
    for bone in skeleton.bones:
        end_ind = start_ind + len(bone.channels)
        bone_addr[bone.name] = (start_ind, end_ind)
        start_ind = end_ind
    dof_num = start_ind

    poses = np.zeros((mocap.nframes, dof_num))
    for i in range(mocap.nframes):
        for bone in skeleton.bones:
            trans = np.array(mocap.frame_joint_channels(i, bone.name, bone.channels))
            start_ind, end_ind = bone_addr[bone.name]
            poses[i, start_ind:end_ind] = trans

    return poses[1:], bone_addr


def main():
    bvh_files = glob.glob(os.path.expanduser('./data/*.bvh'))
    bvh_files.sort()
    print(bvh_files)
    exclude_bones = {'Hand', 'Foot', 'End', 'Toe', 'Head'}
    spec_channels = {'LeftForeArm': ['Xrotation', 'Yrotation'], 'RightForeArm': ['Xrotation', 'Yrotation'],
                     'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation'],
                     'LeftFoot': ['Xrotation', 'Yrotation'], 'RightFoot': ['Xrotation', 'Yrotation']}
    skeleton = Skeleton()
    skeleton.load_from_bvh(bvh_files[0], exclude_bones, spec_channels)

    bone_err_counters = []
    for file in bvh_files:
        data_name = file[7:-4]
        print('extracting trajectory from %s' % file)
        poses, bone_addr = load_bvh_file(file, skeleton)
        bone_err_counter = dict()
        for k in bone_addr.keys():
            bone_err_counter[k] = 0
        error_num = 0
        rot_vals = []
        for frame, pose in enumerate(poses):
            error = False
            rot_val = []
            for bone, addr in bone_addr.items():
                lb = skeleton.name2bone[bone].lb
                ub = skeleton.name2bone[bone].ub
                rot = pose[addr[0]:addr[1]]
                rot_val.append([rot, lb, ub])
                error_rot = False
                for idx, val in enumerate(rot):
                    if val < lb[idx] or val > ub[idx]:
                        error = error_rot = True
                if error_rot:
                    bone_err_counter[bone] += 1
                    #print('  frame:', frame, ' bone:', bone, ' rot:', rot.tolist(), ' lb:', lb, ' ub:', ub)'
            rot_vals.append(rot_val)
            if error:
                error_num += 1

        print(' Error frame num:', error_num, '/', poses.shape[0], ', Percentage:', error_num / poses.shape[0] * 100, '%')
        print(' Error count:', bone_err_counter)
        bone_err_counters.append(bone_err_counter)

        # save histograms
        bone_dict = dict()
        for i, k in enumerate(skeleton.name2bone.keys()):
            bone_dict[i] = k
        rot_vals = np.array(rot_vals)
        dof_ind = {0: 'z', 1: 'x', 2: 'y'}
        print(rot_vals.shape)
        for bone_idx in range(rot_vals.shape[1]):
            for axis in range(3):  # z, x, y
                #fig = plt.figure()
                fig, ax = plt.subplots()
                lb = rot_vals[0, bone_idx, 1, axis]
                ub = rot_vals[0, bone_idx, 2, axis]
                ax.axvspan(lb, ub, color="gray", alpha=0.3)
                ax.hist(rot_vals[:, bone_idx, 0, axis], bins=360, range=(-180, 180))
                plt.title(bone_dict[bone_idx] + '_' + dof_ind[axis])
                plt.xlabel("joint rotation")
                plt.ylabel("frame")
                save_path = "./histogram/" + data_name
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                fig.savefig(save_path + "/" + bone_dict[bone_idx] + '_' + dof_ind[axis] + ".png")
                plt.close()

        if len(bone_err_counters) == 3:
            x = np.array([i for i in range(len(bone_err_counters[0]))])
            labels = list(bone_err_counters[0].keys())
            err_data_1_1_9 = list(bone_err_counters[0].values())
            err_data_1_2_0 = list(bone_err_counters[1].values())
            err_data_orig = list(bone_err_counters[2].values())
            err_data = [err_data_orig, err_data_1_1_9, err_data_1_2_0]

            margin = 0.2  # 0 <margin< 1
            totoal_width = 1 - margin

            fig, ax = plt.subplots(figsize=(12, 8))
            legend = ["orig", "v_1_1_9", "v_1_2_0"]
            for i, h in enumerate(err_data):
                pos = x - totoal_width * (1 - (2 * i + 1) / len(err_data)) / 2
                plt.bar(pos, h, width=totoal_width / len(err_data), label=legend[i], align="center")
            plt.setp(ax.get_xticklabels(), rotation=-30)
            plt.xticks(x, labels)
            plt.title(data_name[:-5])
            ax.legend()
            plt.savefig("./histogram/" + data_name[:-5] + ".png")
            bone_err_counters = []



if __name__ == '__main__':
    main()
