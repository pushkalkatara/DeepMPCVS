import os
import sys
from PIL import Image
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(0)

from utils.frame_utils import read_gen, flow_to_image
from utils.photo_error import mse_
from calculate_flow import FlowNet2Utils


import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)


from model import VisualServoingLSTM
from interactionmatrix import InteractionMatrix
import habitatenv as hs
import imageio
from os import listdir
from os.path import isfile, join

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

    ITERS = 100
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
    env = hs.HabitatEnv(folder, init_state, depth_type)

    loss_fn = torch.nn.MSELoss(size_average=False)

    f = open(folder + "/log.txt","w+")
    f_pe = open(folder + "/photo_error.txt", "w+")
    f_pose = open(folder + "/pose.txt", "w+")
    img_source_path = folder + "/results/" + "test.rgba.00000.00000.png"
    img_goal_path = folder + "/des.png"
    img_src = read_gen(img_source_path)
    img_goal = read_gen(img_goal_path)
    d1 = plt.imread(folder + "/results/" + "test.depth.00000.00000.png")
    vs_lstm = VisualServoingLSTM(rnn_type=rnn_type, layers=NUM_LAYERS, seq_len=SEQ_LEN).cuda()
    optimiser = torch.optim.Adam(vs_lstm.parameters(), lr=LR)
    #optimiser = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimiser, gamma=0.98)
    photo_error_val=mse_(img_src,img_goal)
    print("Initial Photometric Error: ")
    print(photo_error_val)
    f.write("Photometric error = " + str(photo_error_val) + "\n")
    f_pe.write(str(photo_error_val) + "\n")
    start_time = time.time()
    step=0
    while photo_error_val > 500 and step < 1500:
        f12 = flow_utils.flow_calculate(img_src, img_goal)
        vs_lstm.reset_hidden()
        if depth_type == 'TRUE':
            vel, Lsx, Lsy = intermat.getData(f12, d1)
        elif depth_type == 'FLOW':
            if step == 0:
                vel, Lsx, Lsy = intermat.getData(f12, d1)
            else:
                flow_depth_proxy = flow_utils.flow_calculate(img_src, pre_img_src)
                flow_depth=np.linalg.norm(flow_depth_proxy,axis=2)
                flow_depth=flow_depth.astype('float64')
                vel, Lsx, Lsy = intermat.getData(f12, 1/flow_depth)

        if step == 0:
            if vel_init_type == 'RANDOM':
                vel = np.random.normal(size=[1,1,6])
                vel = torch.tensor(vel, dtype = torch.float32).cuda()
            elif vel_init_type == 'IBVS':
                vel = np.random.normal(size=[6])
                #print(vel.shape)
                vel = torch.tensor(vel, dtype = torch.float32).cuda()
        else:
            vel = torch.tensor(vs_lstm.v_interm[1][0], dtype = torch.float32).cuda()
        Lsx = torch.tensor(Lsx, dtype = torch.float32).cuda()
        Lsy = torch.tensor(Lsy, dtype = torch.float32).cuda()
        f12 = torch.tensor(f12, dtype = torch.float32).cuda()

        f.write("Processing Optimization Step: " + str(step) + "\n")
        ts=time.time()
        print(step)
        for cnt in range(ITERS):
            vs_lstm.v_interm = []
            vs_lstm.f_interm = []
            vs_lstm.zero_grad()
            f_hat = vs_lstm.forward(vel, Lsx, Lsy)
            loss = loss_fn(f_hat, f12)
            f.write("Epoch " + str(cnt) + "\n")
            print("Epoch:", cnt)
            f.write("MSE: " + str(np.sqrt(loss.item())))
            print("MSE:", str(np.sqrt(loss.item())))
            loss.backward(retain_graph=True)
            optimiser.step()
        tt = time.time() - ts
        
        # Do not accumulate flow and velocity at train time
        vs_lstm.v_interm = []
        vs_lstm.f_interm = []

        with torch.no_grad():
            f_hat = vs_lstm.forward(vel, Lsx, Lsy)

        f.write("Predicted Velocities: \n")
        f.write(str(vs_lstm.v_interm))
        f.write("\n")

        # Update new Source Image and Depth
        if depth_type == 'FLOW':
            img_src, pre_img_src, d1 = env.example(vs_lstm.v_interm[1][0], step+1, folder)
        elif depth_type == 'TRUE':
            img_src, d1 = env.example(vs_lstm.v_interm[1][0], step+1, folder)
        
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
    del vs_lstm
    del loss_fn
    del optimiser

if __name__ == '__main__':
    main()
