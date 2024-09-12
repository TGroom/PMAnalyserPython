
PCA Analyser Tool for Insect Data Kinematics in 3D Space (PMAnalyserPython)
=============

**Author:** *Thomas Groom*

# Table of Contents

- [PCA Analyser Tool for Insect Data Kinematics in 3D Space (PMAnalyserPython)](#pca-analyser-tool-for-insect-data-kinematics-in-3d-space-pmanalyserpython)
- [Table of Contents](#table-of-contents)
- [About](#about)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Documentation](#documentation)
  - [Load Project File Button](#load-project-file-button)
  - [Save Project File Button](#save-project-file-button)
  - [PCA Analysis Tab](#pca-analysis-tab)
    - [Subject Selected](#subject-selected)
    - [Open Files](#open-files)
    - [PCA Mode](#pca-mode)
    - [Delimiter](#delimiter)
    - [Sample Frequency](#sample-frequency)
    - [Delete Rows](#delete-rows)
    - [Delete Markers](#delete-markers)
    - [Gap Filling](#gap-filling)
    - [Data Filtering](#data-filtering)
    - [Centring](#centring)
    - [Align Orientation](#align-orientation)
    - [Orientation Freq. Cut-off](#orientation-freq-cut-off)
    - [Normalisation](#normalisation)
    - [Weights](#weights)
    - [Coordinate Transform](#coordinate-transform)
    - [Position/Velocity/Acceleration PCA Filters](#positionvelocityacceleration-pca-filters)
    - [Order](#order)
    - [Cut-off Freq.](#cut-off-freq)
    - [Cross-Validation](#cross-validation)
    - [Sine Approx. Freq. Ratios](#sine-approx-freq-ratios)
    - [Run PCA Button](#run-pca-button)
    - [Delete Subject-Specific Rows](#delete-subject-specific-rows)
    - [Delete Subject Specific Markers](#delete-subject-specific-markers)
    - [Flip X, Y \& Z](#flip-x-y--z)
    - [Eigenwalker PCA Population](#eigenwalker-pca-population)
    - [Eigenwalker PCA Group](#eigenwalker-pca-group)
    - [Powerspektrum Plot](#powerspektrum-plot)
    - [Cross-Validation Plot](#cross-validation-plot)
    - [Orientation Plot](#orientation-plot)
    - [3D Bug Plot](#3d-bug-plot)
  - [Visualisation Plots Tab](#visualisation-plots-tab)
    - [PP, PV \& PA Plots](#pp-pv--pa-plots)
    - [Explained Variance](#explained-variance)
  - [Videos Tab](#videos-tab)
    - [Subject Time Series](#subject-time-series)
    - [PC Loadings](#pc-loadings)
    - [PC Reconstruction](#pc-reconstruction)
  - [Eigenwalkers Tab](#eigenwalkers-tab)
    - [Eigenwalker Space Reconstruction](#eigenwalker-space-reconstruction)
    - [Projection Controls](#projection-controls)
  - [Settings Tab](#settings-tab)
    - [Theme Colour](#theme-colour)
    - [UI Scaling](#ui-scaling)
- [Outputs](#outputs)
  - [Result Files (\*\_pca.csv)](#result-files-_pcacsv)
  - [Pre-Processed Folder](#pre-processed-folder)
  - [Metadata File (\*\_metadata.txt)](#metadata-file-_metadatatxt)

# About

The PCA Analyser Tool is designed for analysing kinematic data of insects tracked in 3D space. This tool supports processing multiple subject files simultaneously and provides various analytical outputs to analyse movement patterns and variability.

The PCA Analyser Tool has been tested on Python 3.11.0 with the following dependencies:
- Numpy 1.24.3
- Scikit-learn 1.5.0
- Customtkinter 5.2.2
- Scipy 1.13.1
- Matplotlib 3.8.3

# Installation

Ensure you have Python 3.11.0 installed. Install the required packages using pip:
```bash
pip install numpy
pip install pandas
pip install scikit-learn
pip install customtkinter
pip install scipy
pip install matplotlib
```
# Running the Application

To run the application, execute main_controller.py.

# Documentation

The following documentation covers all application UI inputs and outputs, including graphs and files.

## Load Project File Button

This button is located in the top left corner of the application within the left sidebar panel.
It prompts the user to save changes to the current project before allowing the selection of a ".pca" project file to open. Note that when a project is open, the project file is not write-protected by the application, so changes made outside the application will be overwritten when saving the project within the application. The .pca file is in JSON format and can be modified in any text editor or copied to duplicate the project. Loading illegitimate settings should not break the application; however, improperly formatted JSON will raise an error. When a project is selected, its path is shown in the window header.

## Save Project File Button

This button is found below the load project button in the left sidebar panel.
It saves the application state into the currently open .pca project file. If no project file has been opened, it will prompt the user to select a location for saving a new project.


## PCA Analysis Tab

The button to navigate to this tab is in the left sidebar panel.
This is the main tab where the operation of the analysis can be configured. It allows the user to select data files, configure pre-processing steps, configure weights, and determine what to output. This tab is divided into the following three sections:

- **Input Data & Weights:** This section allows the user to specify subject-specific settings such as deletion of rows, markers, and flipping of x, y, and z axes. Furthermore, after running the PCA execution, it will show a 3D plot of the markers post-processing (More details in the [3D Bug Plot](#3D-Bug-Plot) section).

- **Computation Settings:** This section offers many options to configure the processing of data. Each option is discussed in detail in the following section.

- **Plots:** This section displays plots relevant for decision-making regarding pre-processing steps. These include a [Power Spectrum Plot](#Power-Spectrum-Plot), [Cross-Validation Plot](#Cross-Validation-Plot), and [Orientation Alignment Plot](#Orientation-Plot).

When running the analysis the order of pre-processing steps is as follows:
1. Removing Markers
2. Removing Rows
3. Gap Filling
4. Flipping Axes
5. Filtering
6. Centring
7. Aligning Orientation
8. Cartesian to Spherical
9. Feature-wise Centring
10. Applying Weights
11. Normalising

Unless normalisation is set to 'Mean Dist. 2 Markers (Centre Ref.)' in which case the last three pre-processing steps are as follows:

9. Normalising
10. Feature-wise Centring
11. Applying Weights

### Subject Selected

This button is located at the top of the application window.
It allows the user to switch the subject-specific inputs and outputs on the current tab, enabling selection of a subject for entering subject-specific settings or viewing the power spectrum.

### Open Files

This button is located at the top of the computation settings sidebar (in the [PCA Analysis Tab](#pca-analysis-tab)).
It prompts the user to select file(s) to load. Changes to the contents of these files are acceptable as they will be read in each time the PCA analysis is run. The application accepts both CSV and TXT files. Markers are automatically identified by the column heading suffixes (the last character of each column name, such as 'x', 'y', or 'z' in 'marker_x', 'marker_y', 'marker_z').

**Note:** The application does not write to the input data files. For example, specifying the deletion of rows does not remove these from the CSV input data file.

### PCA Mode

This button is found in the computation settings sidebar.
It allows the user to select between 'All Subjects Together' and 'Every Dataset Separately'.

- **All Subjects Together:** All subjects are pre-processed, and the resulting pre-processed datasets are concatenated before a single PCA is run across the concatenated data. PC loadings (components) are the same for all subjects. Explained variance within the app is shown as it pertains to each subject individually. The overall explained variance is output in the resulting .csv files and is common among all subjects. The [Subject Selection](#subject-selection) dropdown allows subject-specific info about the data that corresponds to a specific subject in the concatenated data to be shown. Note: The centring on the mean for PCA is always performed on a per-subject basis, so each subject has their own mean posture still (in line with how the PMAnalyser Matlab application does it).

- **Every Dataset Separately:** All subjects are pre-processed, and a PCA is run on each dataset separately. Each subject will have their own mean posture, PC loadings, and explained variance. The [Subject Selection](#subject-selection) dropdown selects which subject results to view. The overall explained variance in the output .csv file is the same as the subject explained variance.

**Note:** Subject-specific settings are in effect using either option.

### Delimiter

This text entry is found in the computation settings sidebar.
It allows the user to specify a delimiter for the input data files. If left blank, the default delimiter of ',' is used.

### Sample Frequency

This text entry is found in the computation settings sidebar.
This is a required entry that allows the user to specify the frequency of the given input data in Hz, interpreted as the number of samples taken every second. It expects an integer or floating point value.

### Delete Rows

This text entry is found in the computation settings sidebar.
It expects Python index/slicing syntax and is used to remove specified rows from each dataset.
Example: Setting this to "[:10], [200:]" will remove the first 10 rows as well as any rows beyond row 199 (zero indexed), leaving rows 10-199 in all datasets.

### Delete Markers

This text entry is found in the computation settings sidebar.
It expects Python index/slicing syntax and is used to remove markers from all datasets. The identified markers are shown in the weights table along with their index (on the left). This can be used to ensure the intended markers are being removed. Columns in the datasets without three matching suffixes are ignored and removed automatically. Example: "[39:]" removes all markers with an index greater than 38 (zero indexed).

### Gap Filling

This dropdown is found in the computation settings sidebar.
It imputes 'None' and Nan values in the dataset using the `pandas.DataFrame.interpolate` method. The imputation is done across a single column of data (gap filling is not aware of the x, y, z marker structure). Available options include:

- **Off:** No gap filling will take place. Note: Missing data will likely cause an error window to pop up.
- **Forward Fill:** Use the previous value to inform the next n missing values.
- **Backward Fill:** Use the next value to inform the previous n missing values.
- **Linear:** Use linear interpolation to inform n missing values between two known values.
- **Quadratic:** Use quadratic interpolation, followed by 'ffill' and 'bfill' to ensure no missing values remain.
- **Cubic:** Use cubic interpolation, followed by 'ffill' and 'bfill' to ensure no missing values remain.
- **1st Order Spline:** Use 1st Order Spline interpolation.
- **2nd Order Spline:** Use 2nd Order Spline interpolation.

See [pandas.DataFrame.interpolate](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html) for more details.

### Data Filtering

This dropdown is found in the computation settings sidebar.
It allows the user to filter the input data:

- **Off:** Does not filter the input data.
- **Butterworth:** Filters the input data using Butterworth filtering with an order specified by the UI entry '[Order](#order)' and cut-off frequency specified by the UI entry '[Cut-off Frequency](#cut-off-frequency)'.

### Centring

This dropdown is found in the computation settings sidebar.
It centres each row/sample of the data to a central reference position. This option is very useful when a subject moves within the 3D marker tracking space. The options include:

- **Off:** Does not centre the subject.
- **Mean Marker Position:** Centres each sample by subtracting the mean of all markers in the x, y, and z axes.
- **Mean Marker Pos. (Weights):** Centres each sample by subtracting the weighted mean of all markers in the x, y, and z axes using the weights specified in the weights column of the table.
- **Mean Marker Pos. (Centre Ref.):** Centres each sample by subtracting the weighted mean of all markers in the x, y, and z axes using the weights specified in the Centre Ref. column of the table.

### Align Orientation

This dropdown is found in the computation settings sidebar.
It allows the orientation of the subject to be aligned with a particular axis. This option is useful when subjects do not move in a straight line or when different subjects have different orientations in which they move. It uses the two highest values in the 'Centre Ref' column as reference points to calculate a normal. Hint: To invert this normal, make one of the two largest values slightly smaller than the other according to your needs. The value of this normal goes through Butterworth filtering with a cut-off specified in [Orientation Frequency Cut-off](#orientation-frequency-cut-off). This allows the user to specify a frequency of orientation oscillation to remove/keep.

### Orientation Freq. Cut-off

This text entry is found in the computation settings sidebar.
This value specifies the cut-off frequency for the orientation alignment. Orientation oscillation frequencies below this value are removed from the subject, while values above this frequency are retained. If set to 0.0, the orientation of the subject is not altered other than orientating the mean subject orientation to the specified axis (but all orientation variance is kept). It is recommended to set this value between 0.0 and 1.0 as this will remove orientation as a result of positioning rather than subject kinematics. Note: This entry is irrelevant when the [Align Orientation](#align-orientation) option is set to 'off'. Hint: Use the [Orientation Plot](#orientation-plot) to visualize the changes this makes.

### Normalisation

This dropdown is found in the computation settings sidebar.
It normalizes the data using the following methods:

- **Off:** Does not normalize the data. This is not recommended unless the input data has already been normalized. Without normalization, morphological differences and differences such as the distance the subject is from the camera can impact the PCA results.
- **MED (Mean Euclidean Distance):** Normalizes the data by dividing by the mean Euclidean distance. Note: This option is not the MED in 3D space (i.e., $\sqrt{x_i^2+y_i^2+z_i^2}$) but rather the MED for each column (i.e., $\sqrt{x_1^2 + x_2^2 + ... + x_n^2}$, $\sqrt{y_1^2 + y_2^2 + ... + y_n^2}$, $\sqrt{z_1^2 + z_2^2 + ... + z_n^2}$).
- **Mean Dist. 2 Markers (Centre Ref.):** Normalizes the data by dividing by the mean distance between two markers using the two largest values in the Centre Ref. column to determine which two markers to find the distance between.
- **Maximum Range (1st Coords):** Normalizes the data by dividing by the maximum range of the first coordinates (X if using Cartesian, Phi ($\phi$) if using spherical coordinates).
- **Maximum Range (2nd Coords):** Normalizes the data by dividing by the maximum range of the second coordinates (Y if using Cartesian, Theta ($\theta$) if using spherical coordinates).
- **Maximum Range (3rd Coords):** Normalizes the data by dividing by the maximum range of the third coordinates (Z if using Cartesian, R if using spherical coordinates).

### Weights

This dropdown is found in the computation settings sidebar.
It determines whether to weight the marker positions by the weights specified in the weight column. When enabled, this will multiply the data by the weight value in the weights column. Weights are not normalized.

### Coordinate Transform

This dropdown is found in the computation settings sidebar.
It allows the input data to be transformed into a different coordinate system before PCA is applied. Options:

- **Off:** Coordinates are not transformed and are kept as X, Y, Z.
- **Cartesian -> Spherical (3D):** Transforms the Cartesian (X, Y, Z) coordinates to spherical (Phi, Theta, R) coordinates. The transformation ensures that angle boundaries (e.g., going from $2\pi \rightarrow 0$) are treated specially so that they do not cause discontinuous values to appear. Instead, angle values can exceed $2\pi$ to maintain continuity, which is crucial when applying PCA to obtain meaningful results. The output CSV files will contain the (Phi, Theta, R) values instead of the X, Y, Z values for the loadings and time series (including time series metrics and velocity and acceleration time series).


### Position/Velocity/Acceleration PCA Filters

These dropdowns are found in the computation settings sidebar.
They specify if the time series position, velocity, and acceleration results should receive post-processing filtering. The Butterworth filter applied uses an order specified by the UI entry '[Order](#order)' and cut-off frequency specified by the UI entry '[Cut-off Frequency](#cut-off-frequency)'.


### Order

This entry is found in the computation settings sidebar.
It controls what order of Butterworth filtering to use for the pre-processing and post-processing of data and time series data, respectively.


### Cut-off Freq.

This entry is found in the computation settings sidebar.
It controls what Butterworth cut-off frequency to use for the pre-processing and post-processing of data and time series data, respectively.


### Cross-Validation

This checkbox is found in the computation settings sidebar.
If enabled, it calculates the Leave-One-Out Cross-Validation (LOOCV) of the PCA method. When the [PCA Mode](#pca-mode) is set to 'All Subjects Together,' it will compute the cross-validation on the concatenated data across all subjects. When the [PCA Mode](#pca-mode) is set to 'Each Dataset Separately,' it will compute the cross-validation for each subject's PCA. The Predicted Residual Error Sum of Squares (PRESS) results are plotted in the [Cross-Validation Plot](#cross-validation-plot) and output to the results CSV file. For more details on the methods used, see [Cross-Validation Plot](#cross-validation-plot).


### Sine Approx. Freq. Ratios

This entry is found in the computation settings sidebar.
Leaving this field blank will result in the sine approximations for the PC scores being estimated without considering each other's frequencies. However, if the frequency relationships between the PC scores are known, this setting allows you to fix the frequencies as factors of the PC1 frequency.
For example, if the first two PCs should have the same frequency, and PC3 and PC4 should be second-order harmonics of PC1, then set this field to "1, 1, 2, 2". In this example sine approximation frequencies for PCs from PC5 onwards are not fixed.
To skip setting a harmonic frequency for a specific PC, set its harmonic value to 0. For example, "1.0, 0.0, 1.5" allows PC2 to have any frequency, while PC3 will have 1.5 times the frequency of PC1. All harmonics are relative to PC1, so setting this to "2, 1" means that PC2 will have half the frequency of PC1 when approximating a sine wave.


### Run PCA Button

This button is located in the bottom right corner of the PCA Analysis tab.
When pressed, it performs pre-processing on all input data files as specified in the 'Computation Settings' and 'Input Data & Weights' sections. After pre-processing, it performs PCA on the data either all together or individually, depending on the [PCA Mode](#pca-mode) setting. It also performs cross-validation if enabled. Lastly, it saves the results. More information on saving results can be found in the '[Outputs](#outputs)' section.


### Delete Subject-Specific Rows

This text entry is found in the 'Input Data & Weights' section.
It expects Python index/slicing syntax. Note: This setting is unique to each subject in the [Subject Selection](#subject-selection) dropdown. It is used to remove specified rows from a specific subject only. Example: Setting this to "[:10], [200:]" will remove the first 10 rows as well as any rows beyond row 199 (zero indexed), leaving rows 10-199 for the specific subject.


### Delete Subject Specific Markers

This text entry is found in the 'Input Data & Weights' section.
It expects Python index/slicing syntax. Note: This setting is unique to each subject in the [Subject Selection](#subject-selection) dropdown. It is used to remove markers from a specific subject. The identified markers are shown in the weights table along with their index (on the left). This can be used to ensure the intended markers are being removed. Example: "[39:]" removes all markers larger than index 38 (zero indexed) for a particular subject.


### Flip X, Y & Z

These checkboxes are found in the 'Input Data & Weights' section.
These checkboxes allow the user to specify axes to be flipped/inverted. This option is useful when the input data axes are flipped in some subjects, or if it is desirable for positive Z to be up by convention, given data that has negative Z as the upward direction.


### Eigenwalker PCA Population

This feature allows you to assign a population to a subject. For further details on population usage, refer to the [Eigenwalkers Tab](#Eigenwalkers-Tab). Note that population assignment is case-insensitive. If no population is assigned, the subject defaults to the "None" population.


### Eigenwalker PCA Group

This feature enables you to assign a group to a subject. For additional information on group usage, see the [Eigenwalkers Tab](#Eigenwalkers-Tab). Group assignment is case-insensitive. 


### Powerspektrum Plot

This plot is the first of three tabs found below the 'Input Data & Weights' section.
This plot shows the power spectrum (Welch) plot of the resulting time series data. It can be useful for determining suitable order and cut-off values for pre-processing Butterworth filtering.


### Cross-Validation Plot

This plot is the second of three tabs found below the 'Input Data & Weights' section.
It plots the results of Leave-One-Out Cross-Validation as the number of PC used in the reconstruction are increased (X-Axis). For more details on the methods used see the 'PRESS Naive' and 'PRESS Approx' metrics in the [Outputs](#outputs) section.


### Orientation Plot

This plot is the third of three tabs found below the 'Input Data & Weights' section.
It shows the detected orientation of the subject using the normal vector calculated between the two Centre Ref. column markers in blue. It shows the estimated DC offset of this orientation (using a Butterworth filter with cut-off specified by the [Cut-off Frequency](#cut-off-frequency) entry) in orange. Lastly, it shows the resultant orientation after the offset is corrected in green. This plot also shows a frequency spectrum of the orientation to help determine the cut-off frequency.


### 3D Bug Plot

The 3D bug plot is only shown after a PCA is run on the left within the 'Input Data & Weights' section.
This plot shows a 3D skeleton plot of the currently specified subject post-orientation alignment but pre-feature mean centring (not the same as the [Centring](#centring) option), weighting, and normalization. By default all 3D plots will show the marker data as black point clouds. Lines between the markers can be specified using the 'Skeleton' column in the table.
Each marker can be connected to 1 other marker by entering the index of the marker to be connected in this column. The colour of the markers (and lines) can be specified in the 'Colour' column.
This column will accept any [matplotlib colours](#https://matplotlib.org/stable/users/explain/colors/colors.html) that are written as a string including the following:

- Greyscale values: '0', '0.5', '0.75', etc...
- Single character shorthand: 'b', 'g', 'r', etc...
- X11/CSS4 with no spaces: 'magenta', 'aquamarine', 'mediumseagreen', etc...
- Names from xkcd color survey: 'xkcd:sky blue', 'xkcd:eggshell', etc...
- Hex RGB or RGBA: '#D5A411B0', '#0F0F0F', '#ff0', etc...


## Visualisation Plots Tab

The button to navigate to this tab is in the left sidebar panel.


### PP, PV & PA Plots

These plots show the position, velocity, and acceleration time series/scores for each subject. The velocity is the central difference of the position, and the acceleration is the central difference of the velocity. Optionally, the sine approximations for the time series can be shown.


### Explained Variance

These plots are found on the right-hand side of the [Visualization Plots Tab](#visualization-plots-tab).

They have slightly different meanings depending on the PCA mode selected:

- **All Subjects Together:** This is the explained variance/std and cumulative variance/std of the PCA as it explains each subject individually. Therefore, the explained variances may not be in order from most to least for each subject-specific explained variance.
- **Each Data Set Separately:** This is the explained variance/std and cumulative variance/std of the PCA run on the specified subject. This will always be in descending order of variance explained.


## Videos Tab

The button to navigate to this tab is in the left sidebar panel.
This tab shows animated plots of time series data as well as visualizing PCA loadings/components.

- **PM Amplification Factors:** This entry can be used to amplify the visualized motion on the 3D plots. Example: A value of "[2, 3]" will amplify the motion of the PC1 3D plot by a factor of x2 and the motion of the PC2 plot by a factor of x3. It expects a list of floating point values. Note: The 'Refresh Animations' button must be pressed to enact changes in this entry.

- **View:** This option allows the view to be set to a predetermined orientation of 'Orthographic,' 'Frontal,' 'Sagittal,' or 'Transverse.' Hint: The view of the 3D plots can be shifted by dragging the mouse within the top left 3D plot.


### Subject Time Series

This animated plot shows the real-time contribution of each principal component visualized on the skeleton of the subject. It also shows the time series activity in real-time using a bar chart for each PC.


### PC Loadings

This animated plot is found in the 'PC Loadings' tab of the 'Videos' sidebar tab and shows the 3D skeleton plots of the principal component loadings using a symmetric triangle wave sweep.

### PC Reconstruction

Located in the 'PC Reconstruction' tab of the 'Videos' sidebar, this animated plot displays reconstructed 3D skeletons using the first N principal component loadings (Eigenpostures). The bottom-right plot provides a comparison by showing the original motion, reconstructed using all loadings.

## Eigenwalkers Tab

Navigate to this tab via the button in the left sidebar panel. It visualises the Eigenwalker space by reconstructing motion onto a 3D skeleton plot.

### Eigenwalker Space Reconstruction

This 3D plot illustrates the motion of a point in Eigenwalker space, reconstructed onto an animated skeleton plot. The point's position in the Eigenwalker space is adjusted using the sliders on the right.

### Projection Controls

- **Population Selected:** This dropdown above the Eigenwalker reconstruction plot allows you to switch between PCA results for different subsets of subjects. Populations are specified using the [Eigenwalker PCA Population](#Eigenwalker-PCA-Population) entry.

- **Walker Type:** Choose to view results from PCA on the full walker data, structural (Mean Posture) only, or dynamic (Eigenpostures, phase offsets, and fundamental frequency) only.

- **Axes:** Select whether to control reconstruction using Eigenwalkers (Principal Components) or by linear interpolation along the average position of subject groups in Eigenwalker space. Subject groups are specified using the [Eigenwalker PCA Group](#Eigenwalker-PCA-Group) entry.
  
- **Sliders:** Use these sliders to adjust the point being reconstructed in Eigenwalker space.


## Settings Tab

The button to navigate to this tab is in the left sidebar panel. This tab allows generic changes to the UI to be made, and these options are saved with each project file.


### Theme Colour

This option changes the theme of the UI between 'System', 'Light' or 'Dark'.


### UI Scaling

This option changes the scale of the UI.


# Outputs

When the '[Run PCA](#run-pca-button)' button is pressed, results from the analysis will be saved in the folder "{Project Name}_pca_all_together" or "{Project Name}_pca_each_separately" depending on the [PCA mode](#pca-mode) used. It is important not to rename these folders as the files contained within them are used by the application for plotting the results on graphs and animated plots. Within the results folder, there will be result CSV files containing all the results for a particular subject with filenames of the form "{Subject Name}_pca.csv." Along with these, a folder called "pre-processed" will be created within which the data after it has been pre-processed but before going through PCA will be saved. This is useful if using this application for pre-processing only or to verify the pre-processing results are as expected. Lastly, a text file named "{Project Name}_metadata.txt" will be saved.

The following sections document the files output from running PCA within the application.

## Result Files (*_pca.csv)

The following example Python code can be used to read from these files:

```python
import os
import pandas as pd

# Directory containing the result CSV files
base_dir = '{Project Path}/{Project Name}_pca_all_together'  # OR '{Project Path}/{Project Name}_pca_each_separately'

# Collect all CSV file paths from the base directory
result_csv_files = [os.path.join(base_dir, f) for f in os.listdir(base_dir) if f.endswith('.csv')]

# Initialize a list to store the subject results as DataFrames
subject_results = []

# Read each CSV file into a pandas DataFrame and store it in the subject_results list
for csv_path in result_csv_files:
    df = pd.read_csv(csv_path)
    if not df.empty:
        df.set_index(['level_0', 'level_1'], inplace=True)
        subject_results.append(df)

# Extract data from the first subject's results for demonstration
subj_0_loadings = subject_results[0].loc['Loadings']
subj_0_explained_var = subject_results[0].loc['Overall Explained VAR Ratio'].iloc[0]
subj_0_pp_std = subject_results[0].loc['PP Metrics'].loc['std [AU]']
subj_0_pp_score = subject_results[0].loc['PP Time Series (Position)']

# Print the extracted data for the first subject
print("Loadings:\n", subj_0_loadings)
print("\nOverall Explained Variance Ratio:\n", subj_0_explained_var)
print("\nPP Metrics Standard Deviation [AU]:\n", subj_0_pp_std)
print("\nPP Time Series (Position) Scores:\n", subj_0_pp_score)
```

The following list describes the rows present in these files.

- **Data Mean:** The mean posture of the data. These are the values removed from the markers before applying PCA and are required to reconstruct the data from the PC loadings: 
  $\text{Reconstruction} = \frac{(PP \times \text{Loadings})}{\text{Weights}} + \text{Mean}$.
  Note: Reversing any coordinate transformations would take place after this reconstruction step. This is the only row where the column headings $(PC1-N)$ are not correct as these correspond to the mean values of the input data columns, so it should have the input data column headings.
        
- **Total Var (Eigenvalues):** These are the eigenvalues/total variance for a specific PC axis of the PCA.

- **Overall Explained VAR Ratio:** The total explained variance for all subjects (for PCA mode 'Every Data Set Separately' this is the same as Explained VAR Ratio).

- **Explained VAR Ratio:** The amount of variance each principal component explains for a given subject. Overall, all PCs will sum to 1.

- **Explained STD Ratio:** The standard deviation of the 'Explained VAR Ratio'.

- **Explained Cumulative VAR:** The cumulative 'Explained VAR Ratio'. This will always approach 1 the more PCs are used.

- **Explained Cumulative STD:** The standard deviation of the 'Explained Cumulative VAR'.

- **PRESS Naive:** The predicted residual error sum of squares (PRESS) using the naive method. This method uses a naive reconstruction error formula:
  $\text{PRESS}_{\text{Naive}} = \sum_{i=1}^{n}∥x^{(i)}−\hat{x}^{(i)}∥^2$.
  More info can be found [here](https://stats.stackexchange.com/questions/93845/how-to-perform-cross-validation-for-pca-to-determine-the-number-of-principal-com). Each 'PCN' column corresponds to the PRESS of using N number of PCs in the reconstruction.

- **PRESS Approx:** The predicted residual error sum of squares (PRESS) using an approximation method. This method uses the approximation:
  $\text{PRESS}_{\text{Approx}}=\sum_{i=1}^{n}\sum_{j=1}^{d}\mid x^{(i)}_j - [U^{(-i)}[U^{(-i)}_{-j}]^+x^{(i)}_{-j}]_j \mid^2$.
  Where d is a random sample of data samples. This method was chosen as it compromises between speed and accuracy. More info can be found [here](https://stats.stackexchange.com/questions/93845/how-to-perform-cross-validation-for-pca-to-determine-the-number-of-principal-com). Each 'PCN' column corresponds to the PRESS of using N number of PCs in the reconstruction.

- **Loadings $[0-N]$:** The principal component loadings/components for each input data column/feature for each PC.

- **PP Metrics mean [AU]:** The mean of the PP Time Series data for each PC. This is typically 0.0 (or close to 0.0) for all PCs as data is mean-centred before PCA is applied.

- **PP Metrics std [AU]:** The standard deviation of the PP Time Series data for each PC.

- **PP Metrics meanPos [AU]:** The mean of the positive values of the PP Time Series data for each PC.

- **PP Metrics meanNeg [AU]:** The mean of the negative values of the PP Time Series data for each PC.

- **PP Metrics stdPos [AU]:** The standard deviation of the positive values of the PP Time Series data for each PC.

- **PP Metrics stdNeg [AU]:** The standard deviation of the negative values of the PP Time Series data for each PC.

- **PP Metrics NoZC [#]:** The number of zero crossings that the PP Time Series data makes for each PC. This metric is an integer.

- **PP Metrics NoPeaks [#]:** The number of peaks that the PP Time Series data has for each PC. This metric is an integer.

- **PP Metrics meanTbZC [s]:** The mean time between zero crossings in the PP Time Series data for each PC.

- **PP Metrics meanTbPeaks [s]:** The mean time between peaks in the PP Time Series data for each PC.

- **PP Metrics stdTbZC [s]:** The standard deviation of time between zero crossings in the PP Time Series data for each PC.

- **PP Metrics stdTbPeaks [s]:** The standard deviation of time between peaks in the PP Time Series data for each PC.

- **PP Metrics ratioZC/Peaks [AU]:** The ratio of zero crossings to peaks:
  $\text{Ratio} = \frac{\text{ZC}}{\text{Peaks}}$.
  for each PC.

- **PP Metrics averagePower [AU]:** The mean square of the PP Time Series for each PC:
  $\text{Power} = \frac{1}{N}\sum_{i=0}^{N - 1} (\text{PP}_i)^2$.
 

- **PP Metrics RMS [AU]:** The root mean square of the PP Time Series for each PC:
  $\text{RMS} = \sqrt{\frac{1}{N}\sum_{i=0}^{N - 1} (\text{PP}_i)^2}$.
  Equivalent to the square root of 'PP Metrics averagePower [AU]'.

- **PP Metrics averageActivity [AU]:** The mean absolute PP Time Series for each PC:
  $\text{AverageActivity} = \frac{1}{N}\sum_{i=0}^{N - 1} \mid \text{PP}_i \mid$.
 

- **PP Time Series (Position) $[0-M]$:** The PP Time Series data for each PC and each row of input data (rows 0 to M).

- **PV Metrics:** Same metrics as PP Metrics but for the velocity time series data.

- **PV Time Series (Velocity) $[0-M]$:** The PV Time Series data for each PC and each row of input data (rows 0 to M). This is the differentiation of 'PP Time Series (Position)' using the central difference method.

- **PA Metrics:** Same metrics as PP Metrics but for the acceleration time series data.

- **PA Time Series (Acceleration) $[0-M]$:** The PA Time Series data for each PC and each row of input data (rows 0 to M). This is the differentiation of 'PV Time Series (Position)' using the central difference method.

## Pre-Processed Folder

This folder contains files with filenames in the form "{Subject Name}_preprocessed.csv". These contain the data after it has been pre-processed by the application (but before concatenation) and before being fed into the PCA.

## Metadata File (\*_metadata.txt)

This file stores a copy of the project file (in its native JSON format) used to generate the accompanying data. It can be used to check which settings and file paths were used to generate the data. Furthermore, if desired, the file extension could be changed from ".txt" to ".pca" turning it into a valid project file that can be opened to inspect the settings used.
