
## CarND TERM1 P4

### Tao Yang


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./output_images/road_undist.png "Road Transformed"
[image3]: ./output_images/combined_binary_example.png "Binary Example"
[image4]: ./output_images/warped_image.png "Warp Example"
[image5]: ./output_images/color_fit_lines.png "Fit Visual"
[image6]: ./output_images/example_output.png "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  

The write up with other files can be found in this [repository](https://github.com/taoyang1/CarND-Advanced-Lane-Lines).

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in second code cell of [this](https://github.com/taoyang1/CarND-Advanced-Lane-Lines/blob/master/code/Advanced%20Lane%20Finding.ipynb) IPython notebook.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result. (see the third code cell)

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I apply the distortion correction step to the test1.png file. The result is shown below:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 175 through 212 in `Advanced_Lane_Finding.py`).  Here's an example of my output for this step.  

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform which appears in lines 243 through 280 in the file `Advanced_Lane_Finding.py`. It takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I manually choose the source and destination points in the following manner:

```python
# define four source points
src = np.float32([[255.0,686.2],      # bottom left
                  [575.0,469.4],      # top left
                  [711.8,469.4],      # top right
                  [1044.7,686.2]])     # bottom right

# define corresponding destination points
dst = np.float32([[330,720],
                  [330,0],
                  [950,0],
                  [950,720]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 255.0,686.2   | 330,720       | 
| 575.0,469.4   | 330,0         |
| 1127, 720     | 950,0         |
| 711.8,469.4   | 950,720       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![warped images][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Please see codes in `Advanced_Lane_Finding.py` from line 287 through line 408 and line 587 through line 641.

The following steps are followed to identify lane-line pixels and fit their positions with a second order polynomial.

1. First check if we have previously found left lane and right lane in the last step (i.e., the polynomial coefficient vector for left and right lanes, left_fit, right_fit is not empty).
    i. If it's empty, perform a window based search to find lanes.
    ii. If it's not empty, use the simple search based on previously deteced lanes.
2. Check if the newly deteced lanes are valid. Two sanity checks are considered here:
    i. If there's any lane pixels found at all
    ii. If the mean horizontal defference between two lanes stays within certain range of its running average
3. If fails the sanity check, and a window search hasn't been performed, perform the window search and check the sanity condition again.
4. If the lanes are still bad, return the last best lane fit based on a history of previous 12 frames.
5. If the lanes are good (i.e., pass the sanity check), add this line to the good fit history array, and output the averaged fit to smooth the result. 

The window based search and margin based search codes are borrowed from the course slide.

The final result is shown below.

![Fit visual][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 453 through 476 in my code in `Advanced_Lane_Finding.py` using function `getCurvature()` and `getOffset()`. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 486 through 503 in my code in `Advanced_Lane_Finding.py`.  Here is an example of my result on a test image:

![Output][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest problem I faced is how to tune the parameters to threshold the image so that the lane can be robustly detected. It was a trail and error process. It took me a while to find the right combination of gradient/color masks and the right thresholds. 

My pipeline initially did not have a sanity check and did not maintain a history of past detected lanes. It often failed when the car is driving at difficult parts of the road (i.e., changing colors, shadows, etc.). By adding the sanity check and use past detected lane result when there are bad lines helped to resolve this issue.

All being said, I'm wondering if there is deep learning based approach to the lane finding problem. One of the advantage I appreciate about DL in the behavorial cloning project is that very little manual tuning is required, as opposed in conventional computer vision approach, a lot of tuning is needed.




```python

```
