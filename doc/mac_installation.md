Openpose Doc - Mac OS Installation
==================================

## Contents
Below are instructions for Mac OS Installation that are more detailed than in the instructions in [general installation doc](/doc/installation/0_index.md) for a more thorough walkthrough that's friendly for users who aren't as familiar with using the terminal.

Before beginning installation, make sure you have more 8 GB or more of RAM memory available on your machine.

## Instructions
1. Click the  icon in the top-left corner of your screen. Click 'About This Mac', and take note of whether you have an Intel Mac chip or an Apple Silicon chip.

2. Install Homebrew by opening the Terminal app (built into your Mac) and paste this link: 
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
After runnning this command, you'll be prompted to enter your computer password, but this should be the only time you're prompted for this.

3. Paste this into Terminal:
```bash
brew install cmake pkg-config ffmpeg glog opencv
```
This command installs all the necessary dependencies needed to properly install OpenPose.

4. Clone this repository by running this command:
```bash
git clone https://github.com/lkulkarni137/338OpenPoseDance.git
```

5. Get into the Openpose folder:
```bash
cd openpose
```

6. Download the Openpose model files:
```bash
python ./models/getModels.py
```
After running this command, model files should populate in the `openpose/models/` folder.

7. Next, we'll need to create a build folder to prepare the compiled files.
```bash
mkdir build
cd build
```

8. Run CMake to properly configure Openpose:
```bash
cmake -DGPU_MODE=CPU_ONLY -DCMAKE_POLICY_VERSION_MINIMUM=3.5 ..
```

9. Finish compiling Openpose:
```bash
make -j$(sysctl -n hw.logicalcpu)
```
This step usually takes a while (about 5-15 mins) to run. Once this step is done, Openpose is successfully configured and ready to use.

Now that Openpose is successfully installed, you should be able to run the program and upload/process videos using our user interface.

