# Data Acquistion

Prerequisites:

* ROS 1 installed on the PC
* HTC Vive setup running
* RealSense D4** cameras are used (other cameras can be used but RS cameras are assumed for simplicity)
* ROS 1 workspace contains (besides this repository):
  * HTC Vive ROS interfacing package: https://github.com/robosavvy/vive_ros  
  * RealSense ROS 1 package: https://github.com/IntelRealSense/realsense-ros  


The following steps need to be performed in order to create a data for new tool(s): 

1) Calibration of the HTC Vive system with the cameras
2) Tracing the object(s)
3) Recording of the train (over green screen) & test data
4) Extraction of the recorded data (from ROS bag file)
5) Extraction of bounding boxes of the tool(s)
6) Data postprocessing (mask extraction, erroneous BB filtering)

These steps are described in detail in the following.

## 1) Data collection

Assuming you have created and built a ROS 1 workspace with this package and the packages mentioned above.

### 1a) Starting the HTC Vive and camera(s)

1) Start HTC Vive  
  ```
  roslaunch vive_ros server_vr.launch
  roslaunch vive_ros vive.launch
  ```

2) Start RS cameras
```
roslaunch cam_calib rs_multiple_devices.launch
```  
...or use your own camera launcher, specific to your setup.

### 1b) Camera calibration

4. Run TF -> Pose message Node
```
rosrun cam_calib tf_to_pose.py _target_frame:=<tracked_frame_name>
```
Tracked frame will be something like "tracker_LHR_786752BF" (factory preset tracker name)

5. Start calibration tool
```
rosrun cam_calib calibrator.py
```

### Calibration tool (calibrator.py) usage

Prerequisites:
* calibration checker board (8x6; cell size 72mm, can be adjusted in [calibrator.py](cam_calib/src/calibrator.py))
* 2x ArUco markers (helps to recognize checker board origin for multiple cameras)
  * Markers can be generate with OpenCV with the following dictionary parameters (any marker with this params will work):
  > (nMarkers=48, markerSize=4, seed=65536)
* a spike mounted to an HTC Vive tracker (please, see the related paper for more details)  

### Summary  

The calibration checkerboard should be placed on a table with the ArUco markers placed on the board origin (arbitrary but consistent during recordings). The markers are used to solve the ambiguity of the checkerboard origin.  
Afterwards, the camera extrinsics are auto-calibrated, then the board to Vive calibration is performed. It relies on pinpointing the checkerboard corners within the HTC Vive coordinate frame. This is done by collecting half-spheres by sticking a spike with a tracker on its end into each corner. The end of the spike with the tracker is rotated while the other end is kept at the board corner. We recommend using a plexiglass with a dent to protect the table and make the spike more stable.

### Camera <-> board calibration

First, cameras need to be added. You can click **Add** button to add a camera. Select image topic on the left side of the window and CameraInfo topic on the right side of the window. Alternatively, if a setup was saved previously, you can click **Load** or **Load last setup** button to load the previous setup. Afterwards, each camera should be calibrated by clicking **calibrate** button (extrinsic param. calibration board -> camera) .  
**Make sure the board is visible from both cameras and uncovered. Also, the markers should be around the first corner**  
Sometimes, the markers might not be recognized and the board will not be detected or the board frame will be detected incorrectly. Check visually from the camera image and re-apply calibration if necessary.

### Tracking calibration

##### Starting the tracking calibration tool
Once both cameras are calibrated, click the **Tracking calibration** button. A window will appear where you can select the tracked topic (left side - it shows "PoseStamped" topics) and the camera you want to use (right side).
Wait until the board is calibrated (a sound will play and a rectangle with frame arrows will show in the tracking window).

##### Point collection

Now, collect points using the tracker. Take the plexiglass and put it on a corner. Make sure it is seated exactly above a corner. Then, put the calibration stick with the tracker into the depression in the plexiglass. Start point collection process (see below) and weave with the tracker trying to describe a half-sphere. 

Use **arrow keys** to move the selection diamond marker onto the appropriate corner. Hit **space** to collect the points or stop collection process. The progress bar indicates how many points were collected per current corner (0-1000). A small square appears above each corner for which points were collected.  
Hitting space above an already collected corner will erase currently collected points and start collecting new ones. Alternatively, **backspace** can be used to erase collected points for the selected corner (without starting collection process).

##### Calibration and tracking
After collecting enough points, you can save them by clicking the **Save points** button. These can be loaded later using the **Load points** button.
Click **Calibrate** button to start the pose calibration process.

###  1c) Recording the data

Normal ROS [bagfile recording](http://wiki.ros.org/rosbag/Commandline) can be used to record the data. For convenience, we provide a [launchfile](cam_calib/launch/rec_data_sixdof.launch) that can be modified to suit your specific setup (the default is configured to use topics in our setup).

## 2) Object tracing

To be able to provide the reference bounding boxes for the new tool, we first have to find the bounding boxes of the manipulated objects relative to the tracker.

<img src="./trace-extractor/images/trace_workflow.png" width="1000"/>

We provide a docker installation for the method, sample data and ipython notebook where individual steps are demonstrated. For details see the [Trace extractor README.md](./trace-extractor/README.md)


## 3) Data extraction

### 3a) Extracting images from the bag files

To extract a dataset from the bag file(s), use the [extract_data_from_bag_BB.py](extraction/extract_data_from_bag_BB.py) script. It takes pairs of arguments <bagfile_path> & <tool_calibration_path> (from object tracing) and a range of arguments to specify relevant topic names. Default values for our setup are provided.

To use this script, ROS 1 must be installed and sourced (the script uses for example _rospy_ and _tf_ libraries from ROS 1)

Example usage:  
```bash
$ python extract_data_from_bag_BB.py some_bagfile.bag some_calibration_file.csv
```

### 3b) Processing the extracted images

The extracted training data (recorded over the green screen) needs to be processed to compute a pixel-wise segmentation of the foreground vs. background, i.e., the tool mask. The mask is use for background augmentation during training. To compute the mask the following steps need to be performed:

1) Prepare a `compute_bg_{gluegun, groutfloat, hammer, roller}/` folder:
   * copy empty green background images for C1 and C2  to `Image/` subfolder
   * prepare `C{1,2}_mask_bg.png` files: 255 = inside green cloth, 0 = elsewhere
   * if empty background image is not available, prepare also `C{1,2}_mask_bg_safe.png` files: 255 = pixel always green, 0 = outside cloth or not green in some frames in the `Image/` folder
2) Compute masks using thresholding by running the script [mask_thresholding.py](masking/mask_thresholding.py). The script accepts arguments specifying whether hand should be removed, etc. Use the flag `--help` to see more information about possible arguments.
3) Compute mask refinement using the F, B, Alpha Matting method via the script [mask2rgba.py](masking/mask2rgba.py).
See CLI help for the script to see information about available arguments. This script requires code from the [FBA Matting method repository](https://github.com/MarcoForte/FBA_Matting). Clone the code and either run the [mask2rgba.py](masking/mask2rgba.py) script from inside the main repository folder or specify path to the (cloned) repository as an argument.

The script [bbox_visualize.py](masking/bbox_visualize.py) can be used to visualize the bounding box projections in individual images.

Below is a visualization of the results from collection and postprocessing of the data

https://user-images.githubusercontent.com/17249817/185711525-d843e1ba-f15c-4c0c-bc9c-3b83eaa505a7.mp4
