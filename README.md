# Improving LiDAR Fidelity Using Temporal Infill
Official tensorflow implementation of "Improving LiDAR Fidelity Using Temporal Infill"

This research reviews the infill of vehicle-based occlusions common with ground LiDAR collection for the purpose of HD Map creation for Autonomous Driving. We propose a novel data preparation to convert 4D LiDAR to two-channel, 2.5D imagery. Our preparation pipeline provides the means to create randomized vehicle occlusion patterns accurately across a stack of contiguous Bird's Eye View imagery. We call our work Temporal Infill. Included are all of the networks tested during our research, and we found our final network configuration achieves highly accurate results in both Depth and Reflectance channels.

## Source Code
main_1st.py     -- The top-level python script to run for training for the Coarse model. This allows the user to select the network configuration.\
main_2nd.py     -- The top-level python script to run for training for the Refinement model. This requires a trained Coarse model to produce intermediate dense estimations.\
command_line.py -- The command line arguments for the program. This allows different permutations to be changed at runtime.\
pipeline.py     -- The data pipeline ingest for the training/validation data
train.py        -- The training routine with learning plateaus.
unet_2Df_2Df.py -- Full 2D Encoder, Full 2D Decoders -- Coarse Network 
unet_2Df_3Df.py -- Full 2D Encoder, Full 3D Decoders -- Coarse Network
unet_2Df_3Dp.py -- Full 2D Encoder, Partial 3D Decoders -- Coarse Network
unet_3Df_2Df.py -- Full 3D Encoder, Full 2D Decoders -- Coarse Network
unet_3Df_3Df.py -- Full 3D Encoder, Full 3D Decoders -- Coarse Network
unet_3Df_3Dp.py -- Full 3D Encoder, Partial 3D Decoders -- Coarse Network
unet_2nd.py     -- The Refinement Network.

### Example Command Lines:

**Example 1:** Training the Coarse model with Temporal Infill
```buildoutcfg
>>>python main_1st.py --in-channels 0 1 --out-channels 0 1 --encode-type 3Df --decode-type 2Df --kernels 128 128 128 128 --kernel-size 24 24 24 24 --strides 2 2 2 1 --l2-regularization 1e-5 --batch 1 --patience 50 --max-epochs 500 --learning-rate 5e-5 --momentum .0 --train-gan --model-dir ./tests/example1/ ./data/
```
*--in-channels and --out-channels ensures we are using and generating both channels. \
--encode-type and --decode-type select the type of encoder/decoder to use. \
--kernels, --kernel-size and --strides defines the model topology used in the experiment.\
--patience and --max-epochs defines the plateau-reductions schedule and total number of epochs.\
--learning-rate and --momentum define the base learning rate without momentum enabled.\
--model-dir is where the save models and tensorboard results will be saved.\
./data is a positional argument where the training/validation data exists.*

**Example 2:** Training the Refinement model with Temporal Infill
```buildoutcfg
>>>python main_2nd.py --in-channels 0 1 --out-channels 0 1 --noise 2. 4. --kernels 128 256 256 256 --kernel-size 4 4 4 5 --strides 2 2 2 1 --l2-regularization 1e-5 --batch 1 --patience 50 --max-epochs 500 --learning-rate 5e-5 --momentum .0 --train-gan --model-dir ./tests/example1/ ./data/
```
*--noise tuple to select the amount of Gaussian noise to add to the Coarse output per channel.*

## The Data
We provide the jpgs/pkl files for our data, and pipeline.py provides code to read and prepare the training data. The records are too large for github, so they must be downloaded separately:

[Download Dataset Now - 1.9 GB](https://improving-lidar-fidelity.s3.us-east-2.amazonaws.com/using/temporal-infill.zip)

Unzip into ./ImprovingLiDARTemporalInfill/data/*