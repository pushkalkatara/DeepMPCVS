from interactionmatrix import InteractionMatrix
import habitatenv as hs
import sys
import numpy as np
import os
import time
import matplotlib.pyplot as plt

from utils.frame_utils import read_gen, flow_to_image
from utils.photo_error import mse_
from calculate_flow import FlowNet2Utils


def main():
    folder = sys.argv[1]
    x = np.float(sys.argv[2])
    y = np.float(sys.argv[3])
    z = np.float(sys.argv[4])
    w = np.float(sys.argv[5])
    p = np.float(sys.argv[6])
    q = np.float(sys.argv[7])
    r = np.float(sys.argv[8])
    vel_init = int(sys.argv[9])
    rnn_type = int(sys.argv[10])
    depth_type = int(sys.argv[11])

    ITERS = 50
    SEQ_LEN = 5
    NUM_LAYERS = 5
    LR = 0.0001

    if vel_init == 1:
        vel_init_type = 'RANDOM'
    else:
        vel_init_type = 'IBVS'

    if rnn_type == 1:
        rnn_type = 'LSTM'
    else:
        rnn_type = 'GRU'

    if depth_type == 1:
        depth_type = 'TRUE'
    else:
        depth_type = 'FLOW'

    # Create folder for results
    if not os.path.exists(folder+'/results'):
        os.makedirs(folder+'/results')

    flow_utils = FlowNet2Utils()
    intermat = InteractionMatrix()
    init_state = [x, y, z, w, p, q, r]
    env = hs.HabitatEnv(folder, init_state, 'FLOW')

    flow_utils.save_flow_with_image(folder)

    f = open(folder + "/log.txt","w+")
    f_pe = open(folder + "/photo_error.txt", "w+")
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
    step=0

    while photo_error_val > 100 and step < 1500:
        f12 = flow_utils.flow_calculate(img_src, img_goal)
        if step == 0:
            vel, Lsx, Lsy = intermat.getData(f12, d1)
        else:
            flow_depth_proxy = flow_utils.flow_calculate(img_src, pre_img_src)
            flow_depth=np.linalg.norm(flow_depth_proxy,axis=2)
            flow_depth=flow_depth.astype('float64')
            vel, Lsx, Lsy = intermat.getData(f12, 1/flow_depth)
        
        print(vel)
        img_src, pre_img_src, d1 = env.example(vel, step+1, folder)
        photo_error_val = mse_(img_src,img_goal)
        f.write("Photometric error = " + str(photo_error_val) + "\n")
        print(photo_error_val)
        f.write("Step Number: " + str(step) + "\n")
        f_pe.write(str(photo_error_val) + "\n")
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
    del vs_lstm
    del loss_fn
    del optimiser    

if __name__ == '__main__':
    main()
