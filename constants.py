import numpy as np

FOCAL_LENGTH = 5000.
IMG_RES = 224
IMG_H = 256
IMG_W = 192
CROP_ASPECT_RATIO = 256 / 192.

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]

"""
We create a superset of joints containing the OpenPose joints together with the ones that each dataset provides.
We keep a superset of 24 joints such that we include all joints from every dataset.
If a dataset doesn't provide annotations for a specific joint, we simply ignore it.
The joints used here are the following:
"""
JOINT_NAMES = [
# 25 OpenPose joints (in the order provided by OpenPose)
'OP Nose',
'OP Neck',
'OP RShoulder',
'OP RElbow',
'OP RWrist',
'OP LShoulder',
'OP LElbow',
'OP LWrist',
'OP MidHip',
'OP RHip',
'OP RKnee',
'OP RAnkle',
'OP LHip',
'OP LKnee',
'OP LAnkle',
'OP REye',
'OP LEye',
'OP REar',
'OP LEar',
'OP LBigToe',
'OP LSmallToe',
'OP LHeel',
'OP RBigToe',
'OP RSmallToe',
'OP RHeel',
# 24 Ground Truth joints (superset of joints from different datasets)
'Right Ankle',
'Right Knee',
'Right Hip',
'Left Hip',
'Left Knee',
'Left Ankle',
'Right Wrist',
'Right Elbow',
'Right Shoulder',
'Left Shoulder',
'Left Elbow',
'Left Wrist',
'Neck (LSP)',
'Top of Head (LSP)',
'Pelvis (MPII)',
'Thorax (MPII)',
'Spine (H36M)',
'Jaw (H36M)',
'Head (H36M)',
'Nose',
'Left Eye',
'Right Eye',
'Left Ear',
'Right Ear'
]

# Dict containing the joints in numerical order
JOINT_IDS = {JOINT_NAMES[i]: i for i in range(len(JOINT_NAMES))}

# Map joints to SMPL joints
JOINT_MAP = {
'OP Nose': 24, 'OP Neck': 12, 'OP RShoulder': 17,
'OP RElbow': 19, 'OP RWrist': 21, 'OP LShoulder': 16,
'OP LElbow': 18, 'OP LWrist': 20, 'OP MidHip': 0,
'OP RHip': 2, 'OP RKnee': 5, 'OP RAnkle': 8,
'OP LHip': 1, 'OP LKnee': 4, 'OP LAnkle': 7,
'OP REye': 25, 'OP LEye': 26, 'OP REar': 27,
'OP LEar': 28, 'OP LBigToe': 29, 'OP LSmallToe': 30,
'OP LHeel': 31, 'OP RBigToe': 32, 'OP RSmallToe': 33, 'OP RHeel': 34,
'Right Ankle': 8, 'Right Knee': 5, 'Right Hip': 45,
'Left Hip': 46, 'Left Knee': 4, 'Left Ankle': 7,
'Right Wrist': 21, 'Right Elbow': 19, 'Right Shoulder': 17,
'Left Shoulder': 16, 'Left Elbow': 18, 'Left Wrist': 20,
'Neck (LSP)': 47, 'Top of Head (LSP)': 48,
'Pelvis (MPII)': 49, 'Thorax (MPII)': 50,
'Spine (H36M)': 51, 'Jaw (H36M)': 52,
'Head (H36M)': 53, 'Nose': 24, 'Left Eye': 26,
'Right Eye': 25, 'Left Ear': 28, 'Right Ear': 27
}

COCO_18 = {
    'Nose':0, 'Neck':1, 'R_Shoulder':2, 'R_Elbow':3, 'R_Wrist':4, 'L_Shoulder':5, 'L_Elbow':6, \
    'L_Wrist':7, 'R_Hip': 8, 'R_Knee':9, 'R_Ankle':10, 'L_Hip':11, 'L_Knee':12, 'L_Ankle':13, \
    'R_Eye':14, 'L_Eye':15, 'R_Ear':16, 'L_Ear':17,
    }
SMPL_22 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21
    }

SMPL_24 = {
    'Pelvis_SMPL':0, 'L_Hip_SMPL':1, 'R_Hip_SMPL':2, 'Spine_SMPL': 3, 'L_Knee':4, 'R_Knee':5, 'Thorax_SMPL': 6, 'L_Ankle':7, 'R_Ankle':8,'Thorax_up_SMPL':9, \
    'L_Toe_SMPL':10, 'R_Toe_SMPL':11, 'Neck': 12, 'L_Collar':13, 'R_Collar':14, 'Jaw':15, 'L_Shoulder':16, 'R_Shoulder':17,\
    'L_Elbow':18, 'R_Elbow':19, 'L_Wrist': 20, 'R_Wrist': 21, 'L_Hand':22, 'R_Hand':23
    }

LSP_JOINT_NAMES = ['Right ankle',
'Right knee',
'Right hip',
'Left hip',
'Left knee',
'Left ankle',
'Right wrist',
'Right elbow',
'Right shoulder',
'Left shoulder',
'Left elbow',
'Left wrist',
'Neck',
'Head']

# Joint selectors
# Indices to get the 14 LSP joints from the 17 H36M joints
# 'Pelvis' 'RHip' 'RKnee' 'RAnkle' 'LHip' 'LKnee' 'LAnkle' 'Spine1' 'Neck' 'Head' 'Site' 'LShoulder' 'LElbow' 'LWrist' 'RShoulder' 'RElbow' 'RWrist
H36M_TO_J17 = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10, 0, 7, 9]
H36M_TO_J14 = H36M_TO_J17[:14]
# Indices to get the 14 LSP joints from the ground truth joints
J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
J24_TO_J14 = J24_TO_J17[:14]

# Permutation of SMPL pose parameters when flipping the shape
SMPL_JOINTS_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22]
SMPL_POSE_FLIP_PERM = []
for i in SMPL_JOINTS_FLIP_PERM:
    SMPL_POSE_FLIP_PERM.append(3*i)
    SMPL_POSE_FLIP_PERM.append(3*i+1)
    SMPL_POSE_FLIP_PERM.append(3*i+2)
# Permutation indices for the 24 ground truth joints
J24_FLIP_PERM = [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6, 12, 13, 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]
# Permutation indices for the full set of 49 joints
J49_FLIP_PERM = [0, 1, 5, 6, 7, 2, 3, 4, 8, 12, 13, 14, 9, 10, 11, 16, 15, 18, 17, 22, 23, 24, 19, 20, 21]\
              + [25+i for i in J24_FLIP_PERM]
J45_FLIP_PERM = [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22, 24, \
                 26, 25, 28, 27, 32, 29, 33, 30, 34, 31, 40, 35, 41, 36, 42, 37, 43, 38, 44, 39]

def joint_mapping(source_format, target_format):
    mapping = np.ones(len(target_format), dtype=np.int) *- 1
    for joint_name in target_format:
        if joint_name in source_format:
            mapping[target_format[joint_name]] = source_format[joint_name]
    return np.array(mapping)

# SMPL-X Joints
"""
0	pelvis',
1	left_hip',
2	right_hip',
3	spine1',
4	left_knee',
5	right_knee',
6	spine2',
7	left_ankle',
8	right_ankle',
9	spine3',
10	left_foot',
11	right_foot',
12	neck',
13	left_collar',
14	right_collar',
15	head',
16	left_shoulder',
17	right_shoulder',
18	left_elbow',
19	right_elbow',
20	left_wrist',
21	right_wrist',
22	jaw',
23	left_eye_smplhf',
24	right_eye_smplhf',
25	left_index1',
26	left_index2',
27	left_index3',
28	left_middle1',
29	left_middle2',
30	left_middle3',
31	left_pinky1',
32	left_pinky2',
33	left_pinky3',
34	left_ring1',
35	left_ring2',
36	left_ring3',
37	left_thumb1',
38	left_thumb2',
39	left_thumb3',
40	right_index1',
41	right_index2',
42	right_index3',
43	right_middle1',
44	right_middle2',
45	right_middle3',
46	right_pinky1',
47	right_pinky2',
48	right_pinky3',
49	right_ring1',
50	right_ring2',
51	right_ring3',
52	right_thumb1',
53	right_thumb2',
54	right_thumb3',
55	nose',
56	right_eye',
57	left_eye',
58	right_ear',
59	left_ear',
60	left_big_toe',
61	left_small_toe',
62	left_heel',
63	right_big_toe',
64	right_small_toe',
65	right_heel',
66	left_thumb',
67	left_index',
68	left_middle',
69	left_ring',
70	left_pinky',
71	right_thumb',
72	right_index',
73	right_middle',
74	right_ring',
75	right_pinky',
76	right_eye_brow1',
77	right_eye_brow2',
78	right_eye_brow3',
79	right_eye_brow4',
80	right_eye_brow5',
81	left_eye_brow5',
82	left_eye_brow4',
83	left_eye_brow3',
84	left_eye_brow2',
85	left_eye_brow1',
86	nose1',
87	nose2',
88	nose3',
89	nose4',
90	right_nose_2',
91	right_nose_1',
92	nose_middle',
93	left_nose_1',
94	left_nose_2',
95	right_eye1',
96	right_eye2',
97	right_eye3',
98	right_eye4',
99	right_eye5',
100	right_eye6',
101	left_eye4',
102	left_eye3',
103	left_eye2',
104	left_eye1',
105	left_eye6',
106	left_eye5',
107	right_mouth_1',
108	right_mouth_2',
109	right_mouth_3',
110	mouth_top',
111	left_mouth_3',
112	left_mouth_2',
113	left_mouth_1',
114	left_mouth_5', # 59 in OpenPose output
115	left_mouth_4', # 58 in OpenPose output
116	mouth_bottom',
117	right_mouth_4',
118	right_mouth_5',
119	right_lip_1',
120	right_lip_2',
121	lip_top',
122	left_lip_2',
123	left_lip_1',
124	left_lip_3',
125	lip_bottom',
126	right_lip_3',
127	right_contour_1',
128	right_contour_2',
129	right_contour_3',
130	right_contour_4',
131	right_contour_5',
132	right_contour_6',
133	right_contour_7',
134	right_contour_8',
135	contour_middle',
136	left_contour_8',
137	left_contour_7',
138	left_contour_6',
139	left_contour_5',
140	left_contour_4',
141	left_contour_3',
142	left_contour_2',
143	left_contour_1'
"""


#SMPL Joints:
"""
0	pelvis',
1	left_hip',
2	right_hip',
3	spine1',
4	left_knee',
5	right_knee',
6	spine2',
7	left_ankle',
8	right_ankle',
9	spine3',
10	left_foot',
11	right_foot',
12	neck',
13	left_collar',
14	right_collar',
15	head',
16	left_shoulder',
17	right_shoulder',
18	left_elbow',
19	right_elbow',
20	left_wrist',
21	right_wrist',
22	l_hand',
23	r_hand',
24	nose',
25	right_eye',
26	left_eye',
27	right_ear',
28	left_ear',
29	left_big_toe',
30	left_small_toe',
31	left_heel',
32	right_big_toe',
33	right_small_toe',
34	right_heel',
35	left_thumb',
36	left_index',
37	left_middle',
38	left_ring',
39	left_pinky',
40	right_thumb',
41	right_index',
42	right_middle',
43	right_ring',
44	right_pinky',
"""
