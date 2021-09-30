import math
import re
from bvh import Bvh
import numpy as np


class Bone:

    def __init__(self):
        # original bone info
        self.id = None
        self.name = None
        self.orient = np.identity(3)
        self.dof_index = []
        self.channels = []  # bvh only
        self.lb = []
        self.ub = []
        self.parent = None
        self.child = []

        # bvh specific
        self.offset = np.zeros(3)

        # inferred info
        self.pos = np.zeros(3)
        self.end = np.zeros(3)


class Skeleton:

    def __init__(self):
        self.bones = []
        self.name2bone = {}
        self.mass_scale = 1.0
        self.len_scale = 1.0
        self.dof_name = ['z', 'x', 'y']
        self.root = None

    RoM = {            # [[x_lb, y_lb, z_lb],    [x_ub, y_ub, z_ub]]
        'Hips':          [[-120, -180, -120],    [120, 180, 120]],
        'LeftUpLeg':     [[-100, -40, -40],      [30, 60, 40]],
        'LeftLeg':       [[-20, -10, -10],       [150, 10, 10]],
        'LeftFoot':      [[-20, -10, -1],        [40, 30, 1]],
        'RightUpLeg':    [[-100, -60, -40],      [30, 40, 40]],
        'RightLeg':      [[-20, -10, -10],       [150, 10, 10]],
        'RightFoot':     [[-20, -30, -1],        [40, 10, 1]],
        'Spine':         [[-15, -10, -10],       [25, 10, 10]],
        'Spine1':        [[-10, -10, -10],       [20, 10, 10]],
        'Spine2':        [[-10, -10, -10],       [15, 10, 10]],
        'Spine3':        [[-10, -10, -10],       [15, 10, 10]],
        'Spine4':        [[-10, -10, -10],       [15, 10, 10]],
        'Spine5':        [[-10, -10, -10],       [15, 10, 10]],
        'Spine6':        [[-10, -10, -10],       [15, 10, 10]],
        'LeftShoulder':  [[-20, -20, -25],       [20, 20, 50]],
        'LeftArm':       [[-90, -120, -100],     [120, 60, 40]],
        'LeftForeArm':   [[-5, -150, -5],        [5, 1, 5]],
        'LeftHand':      [[-1, -80, -30],        [1, 60, 20]],
        'RightShoulder': [[-20, -20, -50],       [20, 20, 25]],
        'RightArm':      [[-90, -60, -40],       [120, 120, 100]],
        'RightForeArm':  [[-5, -1, -5],          [5, 150, 5]],
        'RightHand':     [[-1, -60, -20],        [1, 80, 30]],
        'Neck':          [[-45, -30, -30],       [45, 30, 30]]
    }

    def load_from_bvh(self, fname, exclude_bones=None, spec_channels=None):
        if exclude_bones is None:
            exclude_bones = {}
        if spec_channels is None:
            spec_channels = dict()
        with open(fname) as f:
            mocap = Bvh(f.read())

        joint_names = list(filter(lambda x: all([t not in x for t in exclude_bones]), mocap.get_joints_names()))
        dof_ind = {'x': 1, 'y': 2, 'z': 0}
        #self.len_scale = 0.0254
        self.root = Bone()
        self.root.id = 0
        self.root.name = joint_names[0]
        self.root.channels = mocap.joint_channels(self.root.name)
        self.root.lb = [self.RoM[self.root.name][0][2], self.RoM[self.root.name][0][0], self.RoM[self.root.name][0][1]]
        self.root.ub = [self.RoM[self.root.name][1][2], self.RoM[self.root.name][1][0], self.RoM[self.root.name][1][1]]
        self.name2bone[self.root.name] = self.root
        self.bones.append(self.root)
        for i, joint in enumerate(joint_names[1:]):
            bone = Bone()
            bone.id = i + 1
            bone.name = joint

            # pos and rot
            '''
            if joint in spec_channels.keys():
                spec_channel = mocap.joint_channels(joint)[:3]
                spec_channel.extend(spec_channels[joint])
                bone.channels = spec_channel
            else:
                bone.channels = mocap.joint_channels(joint)
            '''
            # rot
            #bone.channels = spec_channels[joint] if joint in spec_channels.keys() else mocap.joint_channels(joint)[3:]
            bone.channels = mocap.joint_channels(joint)[3:]

            bone.dof_index = [dof_ind[x[0].lower()] for x in bone.channels]
            bone.offset = np.array(mocap.joint_offset(joint)) * self.len_scale
            bone.lb = [self.RoM[joint][0][2], self.RoM[joint][0][0], self.RoM[joint][0][1]]  # (z, x, y)
            bone.ub = [self.RoM[joint][1][2], self.RoM[joint][1][0], self.RoM[joint][1][1]]
            self.bones.append(bone)
            self.name2bone[joint] = bone

        for bone in self.bones[1:]:
            parent_name = mocap.joint_parent(bone.name).name
            if parent_name in self.name2bone.keys():
                bone_p = self.name2bone[parent_name]
                bone_p.child.append(bone)
                bone.parent = bone_p

        self.forward_bvh(self.root)
        for bone in self.bones:
            if len(bone.child) == 0:
                bone.end = bone.pos + np.array(
                    [float(x) for x in mocap.get_joint(bone.name).children[-1]['OFFSET']]) * self.len_scale
            else:
                bone.end = sum([bone_c.pos for bone_c in bone.child]) / len(bone.child)

    def forward_bvh(self, bone):
        if bone.parent:
            bone.pos = bone.parent.pos + bone.offset
        else:
            bone.pos = bone.offset
        for bone_c in bone.child:
            self.forward_bvh(bone_c)
