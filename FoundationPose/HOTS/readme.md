Step 1: Understand the Goal
You want to run FoundationPose on the HOTS dataset instead of datasets like LineMOD or YCB. Here's a basic plan for this:

- Load the HOTS data (RGB images, instance masks).
- Modify FoundationPose's pipeline so it uses the RGB and instance masks for pose estimation.
- Test the new integration with HOTS

Step 2: What You Already Have

- FoundationPose code with existing scripts for LineMOD and YCB.
- HOTS dataset already downloaded and a script that loads it (hots.py).
- You’ve added a new run_hots.py script to integrate the HOTS data into FoundationPose.

Step 3: How to Connect HOTS to FoundationPose

- create run_hots.py
    - Load the HOTS dataset using the load_HOTS_scenes function from hots.py.
    - Pass the loaded images and masks to FoundationPose for pose estimation (using only the RGB and mask information).
- Modify FoundationPose for RGB-only Data
    - In FoundationPose's pose estimation (likely in estimater.py), the current method probably expects both RGB and depth data. However, HOTS only provides RGB and instance masks, so you need to modify the register() method in estimater.py to handle this:


Step 4: Keep It Simple for Now
- Ignore depth: HOTS doesn’t have depth, so your focus is on using the RGB and mask data for pose estimation.
- Modify only run_hots.py and register() in estimater.py to get started.
Once you can pass RGB and mask data from HOTS into the pose estimator and run it, you can improve the actual pose estimation method later.

Step 5: Next Steps
- Run the new script (run_hots.py) after these modifications.
- Test if the data is flowing correctly into the register() function from HOTS.
- Modify or improve the pose estimation logic for RGB-only data inside register().


https://github.com/gtziafas/HOTS
https://paperswithcode.com/dataset/linemod-1
https://www.ycbbenchmarks.com/#:~:text=YCB%20Object%20and%20Model%20Set,some%20widely%20used%20manipulation%20tests.
https://github.com/hz-ants/FFB6D?tab=readme-ov-file#datasets
https://github.com/ethnhe/PVN3D/tree/master





https://www.connectedpapers.com/main/dc4c9ae8c0cfc08ff6392aff69b0fd170da398a4/FoundationPose%3A-Unified-6D-Pose-Estimation-and-Tracking-of-Novel-Objects/graph
https://www.semanticscholar.org/paper/OnePose%3A-One-Shot-Object-Pose-Estimation-without-Sun-Wang/37f991349a7d389880d1ff0c62b248b64c296211
https://zju3dv.github.io/onepose_plus_plus/


TODO:
- understand and document again what demo and linemod datasets need for the FoundationPose model to predict
- modify HOTS dataset accordingly + see if anything is missing (and try as much as possible to put dummy data to have something)
- code a run_hots.py similar to demo/linemod scripts
- checkout the paper about one_shot from Hamidreza.