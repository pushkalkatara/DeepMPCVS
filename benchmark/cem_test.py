import os
import sys
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"


import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)
from cem import CEM

from utils.frame_utils import read_gen, flow_to_image
from utils.photo_error import mse_
from calculate_flow import FlowNet2Utils

import time

from interactionmatrix import InteractionMatrix
import habitatenv as hs
import imageio
from os import listdir
from os.path import isfile, join

def MSE(A, B):
    size = A.shape[0]
    sq = np.square(A - B)
    mse = sq.mean(axis=-1).reshape(size, -1)
    return mse

def main():
    folder = sys.argv[1]
    x = np.float(sys.argv[2])
    y = np.float(sys.argv[3])
    z = np.float(sys.argv[4])
    w = np.float(sys.argv[5])
    p = np.float(sys.argv[6])
    q = np.float(sys.argv[7])
    r = np.float(sys.argv[8])


    # Create folder for results
    if not os.path.exists(folder+'/results'):
        os.makedirs(folder+'/results')

    flow_utils = FlowNet2Utils()
    intermat = InteractionMatrix()
    init_state = [x, y, z, w, p, q, r]
    env = hs.HabitatEnv(folder, init_state, 'FLOW')

    cem = CEM(MSE, 6, sampleMethod='Gaussian', v_min=[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5], v_max=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    f = open(folder + "/log.txt","w+")
    f_pe = open(folder + "/photo_error.txt", "w+")
    f_pose = open(folder + "/pose.txt", "w+")
    img_source_path = folder + "/results/" + "test.rgba.00000.00000.png"
    img_goal_path = folder + "/des.png"
    img_src = read_gen(img_source_path)
    img_goal = read_gen(img_goal_path)
    d1 = plt.imread(folder + "/results/" + "test.depth.00000.00000.png")

    photo_error_val=mse_(img_src,img_goal)
    print("Initial Photometric Error: ")
    print(photo_error_val)
    f.write("Photometric error = " + str(photo_error_val) + "\n")
    f_pe.write(str(photo_error_val) + "\n")
    start_time = time.time()
    step = 0
    while photo_error_val > 500 and step < 1500:
        f12 = flow_utils.flow_calculate(img_src, img_goal)
        if step == 0:
            vel, Lsx, Lsy = intermat.getData(f12, d1)
        else:
            flow_depth_proxy = flow_utils.flow_calculate(img_src, pre_img_src)
            flow_depth=np.linalg.norm(flow_depth_proxy,axis=2)
            flow_depth=flow_depth.astype('float64')
            vel, Lsx, Lsy = intermat.getData(f12, 1/flow_depth)
        
        gtf = np.array(f12)
        cem.Lsx = Lsx
        cem.Lsy = Lsy
        v = cem.eval(gtf)
        #print(v, MSE(gtf, np.stack([np.sum(Lsx*v, -1), np.sum(Lsy*v, -1)], -1)))

        f.write("Processing Optimization Step: " + str(step) + "\n")

        f.write("Predicted Velocities: \n")
        f.write(str(v))
        f.write("\n")

        img_src, pre_img_src, d1 = env.example(v.reshape(1,6), step+1, folder)
        
        photo_error_val = mse_(img_src,img_goal)
        f.write("Photometric error = " + str(photo_error_val) + "\n")
        print(photo_error_val)
        f.write("Step Number: " + str(step) + "\n")
        f_pe.write(str(photo_error_val) + "\n")
        f_pose.write("Step : "+ str(step)  + "\n")
        f_pose.write("Pose : " + str(env.get_agent_pose()) + '\n')
        step = step + 1
        
    time_taken = time.time() - start_time
    f.write("Time Taken: " + str(time_taken) + "secs \n")
    # Cleanup
    f.close()
    f_pe.close()
    env.end_sim()
    del flow_utils
    del intermat
    del env

    # save indvidial image and gif
    onlyfiles = [f for f in listdir(folder + "/results") if f.endswith(".png")]
    onlyfiles.sort()
    images = []
    for filename in onlyfiles:
        images.append(imageio.imread(folder + '/results/' + filename))
    imageio.mimsave(folder + '/results/output.gif', images, fps=4)

    

if __name__ == '__main__':
    main()
