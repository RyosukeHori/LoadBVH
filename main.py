import os
import sys
import numpy as np
import argparse
import math
from bvh import Bvh
from skeleton import Skeleton
import glob

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

    return poses, bone_addr


def main():
    skt_bvh = './test.bvh'
    exclude_bones = {'Hand', 'Foot', 'End', 'Toe', 'Head'}
    spec_channels = {'LeftForeArm': ['Xrotation', 'Yrotation'], 'RightForeArm': ['Xrotation', 'Yrotation'],
                     'LeftLeg': ['Xrotation'], 'RightLeg': ['Xrotation'],
                     'LeftFoot': ['Xrotation', 'Yrotation'], 'RightFoot': ['Xrotation', 'Yrotation']}
    skeleton = Skeleton()
    skeleton.load_from_bvh(skt_bvh, exclude_bones, spec_channels)

    bvh_files = glob.glob(os.path.expanduser('./*.bvh'))
    bvh_files.sort()
    print(bvh_files)
    for file in bvh_files:
        print('extracting trajectory from %s' % file)
        poses, bone_addr = load_bvh_file(file, skeleton)
        error_num = 0
        for frame, pose in enumerate(poses):
            error = False
            for bone, addr in bone_addr.items():
                lb = skeleton.name2bone[bone].lb
                ub = skeleton.name2bone[bone].ub
                rot = pose[addr[0]:addr[1]]
                error_rot = False
                for idx, val in enumerate(rot):
                    if val < lb[idx] or val > ub[idx]:
                        error = error_rot = True
                if error_rot:
                    print(' frame:', frame, ' bone:', bone, ' rot:', rot.tolist(), ' lb:', lb, ' ub:', ub)
            if error:
                error_num += 1
        print(' Error frame num:', error_num, ' Percentage:', error_num / poses.shape[0] * 100, '%')


if __name__ == '__main__':
    main()
