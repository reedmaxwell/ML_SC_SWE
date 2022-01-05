## load PFCLM output and make plots / do anaylsis

from parflow.tools.fs import get_absolute_path
from parflowio.pyParflowio import PFData
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from rmm_nn import RMM_NN
from transform import float32_clamp_scaling
import utilities

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# intialize data and time arrays
data    = np.zeros([2,8,8760])
time    = np.zeros([8760])

# load forcing file as numpy array, note should fix so count is 1-8760 instead of 0-8759
# variables are:
# 0 DSWR: Downward Visible or Short-Wave radiation [W/m2].
# 1 DLWR: Downward Infa-Red or Long-Wave radiation [W/m2]
# 2 APCP: Precipitation rate [mm/s]
# 3 Temp: Air temperature [K]
# 4 UGRD: West-to-East or U-component of wind [m/s] 
# 5 VGRD: South-to-North or V-component of wind [m/s]
# 6 Press: Atmospheric Pressure [pa]
# 7 SPFH: Water-vapor specific humidity [kg/kg]
#
#ffname = 'forcing/forcing_1.txt'
ffname = 'forcing/LocalForcing_LowSWE.txt'
forcing_ls = np.loadtxt(ffname,max_rows=8760)
print(np.shape(forcing_ls))

# load in high swe forcing
ffname = 'forcing/LocalForcing_HighSWE.txt'
forcing_hs = np.loadtxt(ffname,max_rows=8760)
print(np.shape(forcing_hs))


# reading the CLM file PFCLM_SC.out.clm_output.<file number>.C.pfb
# variables are by layer:
# 0  total latent heat flux (Wm-2)
# 1  total upward LW radiation (Wm-2)
# 2  total sensible heat flux (Wm-2)
# 3  ground heat flux (Wm-2)
# 4  net veg. evaporation and transpiration and soil evaporation (mms-1)
# 5  ground evaporation (mms-1)
# 6  soil evaporation (mms-1)
# 7  vegetation evaporation (canopy) and transpiration (mms-1)
# 8  transpiration (mms-1)
# 9  infiltration flux (mms-1)
# 10 SWE (mm)
# 11 ground temperature (K)
# 12 irrigation flux
# 13 - 24 Soil temperature by layer (K)

slope    = 0.05
mannings = 2.e-6

# loop over a year of files (8760 hours) and load in the CLM output
# then map specific variables to the data array which holds things for analysis
# and plotting
for icount in range(0, 8759):
    base = "output/PFCLM_SC_LS.out.clm_output.{:05d}.C.pfb"
    filename = base.format(icount+1)
    data_obj = PFData(filename)
    data_obj.loadHeader()
    data_obj.loadData()
    data_arr = data_obj.getDataAsArray()
    data_obj.close()
    data[0,1,icount] = data_arr[0,0,0]  #total (really, it is net) latent heat flux (Wm-2)
    data[0,2,icount] = data_arr[4,0,0]  #net veg. evaporation and transpiration and soil evaporation (mms-1)
    data[0,3,icount] = data_arr[10,0,0] #SWE (mm)
    # base = "output/PFCLM_SC.LS.out.press.{:05d}.pfb"
    # filename = base.format(icount)
    # data_obj = PFData(filename)
    # data_obj.loadHeader()
    # data_obj.loadData()
    # data_arr = data_obj.getDataAsArray()
    # data_obj.close()
    # data[4,icount] = (np.sqrt(slope)/mannings) * np.maximum(data_arr[19,0,0],0.0)**(5.0/3.0)
    time[icount] = icount

# loop over a year of files (8760 hours) and load in the CLM output
# then map specific variables to the data array which holds things for analysis
# and plotting
for icount in range(0, 8759):
    base = "output/PFCLM_SC_HS.out.clm_output.{:05d}.C.pfb"
    filename = base.format(icount+1)
    data_obj = PFData(filename)
    data_obj.loadHeader()
    data_obj.loadData()
    data_arr = data_obj.getDataAsArray()
    data_obj.close()
    data[1, 1,icount] = data_arr[0,0,0]  #total (really, it is net) latent heat flux (Wm-2)
    data[1, 2,icount] = data_arr[4,0,0]  #net veg. evaporation and transpiration and soil evaporation (mms-1)
    data[1, 3,icount] = data_arr[10,0,0] #SWE (mm)
    # base = "output/PFCLM_SC.HS.out.press.{:05d}.pfb"
    # filename = base.format(icount)
    # data_obj = PFData(filename)
    # data_obj.loadHeader()
    # data_obj.loadData()
    # data_arr = data_obj.getDataAsArray()
    # data_obj.close()
    # data[4,icount] = (np.sqrt(slope)/mannings) * np.maximum(data_arr[19,0,0],0.0)**(5.0/3.0)
    time[icount] = icount

# Plot LH Flux, SWE and Runoff
fig, ax = plt.subplots()
ax2 = ax.twinx()
#ax.plot(time[1:17520],forcing[0:17519,2], color='g')
#ax2.plot(time[1:17520],data[3,1:17520], color='b')
#ax2.plot(time[1:17520], forcing[0:17519,3], color='r')
ax.plot(time,forcing_ls[:,2], color='g')
ax2.plot(time,data[0,3,:], color='b')
ax2.plot(time, forcing_ls[:,3], color='r')
ax.set_xlabel('Time, WY [hr]')
ax.set_ylabel('Precip [mm/s]')
ax2.set_ylabel('Temp [K], SWE[mm')
plt.show()


n_batch = 1
N = 8760
nchannel = 2
n_epoch = 200
learning_rate = 1E-3
# Choose the ML and associated options, models are stored in ML_models.py
# and need to be imported up top if used
model = RMM_NN(grid_size=[N],
               channels=nchannel)
model.to(DEVICE)
#model = FirstNeuroNetShapes(
#    grid_size=[n_dim2, n_dim1], channels=nchannel, MP2D_stride=2, verbose=True)
#model = CNN_model(grid_size=[n_dim2, n_dim1],
#                            channels=nchannel, MP2D_stride=1, MP2D_kernel_size=1,
#                            C2D_kernel_size=1, verbose=True)
model.verbose=False
model.use_dropout = True

print("-- Model Definition --")
print(model)

print("-- Model Parameters --")
utilities.count_parameters(model)

## options for different loss models and solvers
# loss function
#
#loss_fn = torch.nn.L1Loss()
#loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.SmoothL1Loss()

# optimizer and solver, Adam works the best, regular SGD seems to work the poorest
#
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 2 APCP: Precipitation rate [mm/s]
# 3 Temp: Air temperature [K]

# Setup transforms to get variables scaled from 0-1
precip_max = max(np.max(forcing_ls[:,2]),np.max(forcing_hs[:,2]))
swe_max = np.max(data[:,3,:])
Precip_TRANSF = float32_clamp_scaling(src_range=[0, precip_max], dst_range=[0, 1])
Temp_TRANSF = float32_clamp_scaling(src_range=[np.min(forcing_ls[:,3]), np.max(forcing_ls[:,3]) ], dst_range=[0, 1])
SWRad_TRANSF = float32_clamp_scaling(src_range=[np.min(forcing_ls[:,0]), np.max(forcing_ls[:,0]) ], dst_range=[0, 1])
LWRad_TRANSF = float32_clamp_scaling(src_range=[np.min(forcing_ls[:,1]), np.max(forcing_ls[:,1]) ], dst_range=[0, 1])
Humid_TRANSF = float32_clamp_scaling(src_range=[np.min(forcing_ls[:,7]), np.max(forcing_ls[:,7]) ], dst_range=[0, 1])
SWE_TRANSF = float32_clamp_scaling(src_range=[0, swe_max], dst_range=[0, 1])
INV_SWE_TRANSF = float32_clamp_scaling(src_range=[0, 1], dst_range=[0, swe_max])

print()
print("-- Model Settings --")
print(" Channels:",nchannel)
print(" Epochs:",n_epoch)
print(" Learning Rate:", learning_rate)
print()
print()


# %%
## loop over all the realization members
##

#print("---------------------------------")
#print("  Training Ensemble Member:", count)
# Setup the input arrays

#setup the input and ouput arrays
input_temp = np.zeros((n_batch, nchannel, N))
output_temp_temp = np.zeros((N))
output_temp = np.zeros((n_batch, N))
model_output = np.zeros(N)
# ___
# load all the transient variables for timesteps, ie the batch members and normalize

# time varying
input_temp[0, 0, :] = Precip_TRANSF(forcing_ls[:,2])
input_temp[0, 1, :] = Temp_TRANSF(forcing_ls[:,3])
#input_temp[0, 2, 0:N-1] = SWRad_TRANSF(forcing[0:8759,0])
#input_temp[0, 3, 0:N-1] = LWRad_TRANSF(forcing[0:8759,1])
#input_temp[0, 2, 0:N-1] = Humid_TRANSF(forcing[0:8759,7])
output_temp[0,:] = SWE_TRANSF(data[0,3,:])


# convert the inputs and outputs (labels) from the temp NumPy vectors to Torch format
# convert them to Floats
torch_input = torch.from_numpy(input_temp)
torch_label = torch.from_numpy(output_temp)
torch_input = torch_input.type(torch.FloatTensor).to(DEVICE)
torch_label = torch_label.type(torch.FloatTensor).to(DEVICE)
# ___
# Train the model
print()
print("Progress:")
for epoch in range(n_epoch):
    optimizer.zero_grad()
    # Forward pass
    batch_prediction = model(torch_input)
    loss = loss_fn(batch_prediction, torch_label)

    print("Epoch: %3d, loss: %5.3e" % (epoch, loss), end='\r')

    # Backward and optimize
    loss.backward()
    optimizer.step()
print()
print("____")
print("Training Complete")

# %%
# Save our progress
#trainer.save(model_path)
print("-----------------------------------")
print(" Saving the model")

torch.save(model, "SWE_test.pt")

print("-----------------------------------")
print(" Make Predictions")
print
# freeze the model
model.use_dropout = False
for parameter in model.parameters():
    parameter.requires_grad = False


# time varying
input_temp[0, 0, :] = Precip_TRANSF(forcing_hs[:,2])
input_temp[0, 1, :] = Temp_TRANSF(forcing_hs[:,3])
#input_temp[0, 2, 0:N-1] = SWRad_TRANSF(forcing[0:8759,0])
#input_temp[0, 3, 0:N-1] = LWRad_TRANSF(forcing[0:8759,1])
#input_temp[0, 2, 0:N-1] = Humid_TRANSF(forcing[0:8759,7])

predict_input = torch.from_numpy(input_temp)
# convert to Floats
predict_input = predict_input.type(torch.FloatTensor).to(DEVICE)
prediction = model(predict_input)
# copy into ML model output array
#
model_output[:] = INV_SWE_TRANSF(prediction.data.cpu().numpy())

# Plot LH Flux, SWE and Runoff
fig, ax = plt.subplots()
ax.plot(time,data[1,3,:], color='b')
ax.plot(time,model_output, color='r')
ax.set_xlabel('Time, WY [hr]')
ax.set_ylabel('Predicted, Simulated SWE')
plt.show()