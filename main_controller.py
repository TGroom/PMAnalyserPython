"""
Main module for running PCA analysis application.

This module defines an application (`App_Controller`) that extends `PCA_View` and provides methods 
to execute PCA analysis on multiple subject files.

Tested on Python 3.11.0 using:
    - numpy 1.24.3
    - scikit-learn 1.5.0
    - customtkinter 5.2.2
    - scipy 1.13.1
    - matplotlib 3.8.3

These packages need to be installed via pip:
    pip install numpy
    pip install scikit-learn
    pip install customtkinter
    pip install scipy
    pip install matplotlib
    
"""
import os
from threading import Thread
import json
import tkinter
import itertools
import time

import pandas as pd
import numpy as np

import pca_model
from pca_view import PCA_View, TerminatingError, string_to_list


class Subject(pca_model.PCA_Model):
    def __init__(self, df, sample_freq):
        super().__init__(sample_freq)
        self.raw_data = df.copy()
        self.df = df.copy()
        self.markers_to_del_set = []
        self.rows_to_del_set = []
        self.func_order = 0
        self.centre_refs = []
        self.skeleton = []
        self.line_colors_raw = []
        self.line_colors = []
        self.mean_of_data = []
        self.bug_plot_data = pd.DataFrame()
        self.results_file_path = None
        self.subject_UI_settings = {
                    'rows_to_del': tkinter.StringVar(),
                    'markers_to_del': tkinter.StringVar(),
                    'flip_x': tkinter.BooleanVar(),
                    'flip_y': tkinter.BooleanVar(),
                    'flip_z': tkinter.BooleanVar(),
                    'eigenwalker_group': tkinter.StringVar()
                }
        

    def get_skeleton(self):
        skeleton_temp = np.array([[int(l[0]), int(l[1] or l[0])] for l in self.skeleton])
        skeleton_temp = self._dissolve_net(skeleton_temp, self.markers_to_del_set)
        self.line_colors = self.line_colors_raw[skeleton_temp[:, 0]]
        return self._unique_transform(skeleton_temp)
    
    
    # Removes a skeleton line segment 
    def _dissolve_net(self, segments, points_to_remove):
        '''
        Dissolves line segments from `segments` based on `points_to_remove`.

        Args:
        - segments (numpy.ndarray): Array of line segments represented as pairs of points.
        - points_to_remove (list): List of points to be removed from `segments`.

        Returns:
        - numpy.ndarray: Updated segments after removal, sorted by the first point of each segment.
        '''
        for point in range(np.amax(segments) + 1):  # Iterate through all points up to max point in segments
            if point in points_to_remove:
                # Find segments containing 'point' and flatten to 1D
                to_remove = segments[np.any(np.isin(segments, point), axis=1)].flatten()
                for i in to_remove:
                    if i != point and np.count_nonzero(segments == i) == 1:  # If point is orphaned (only occurs once)
                        segments = np.append(segments, [[i, i]], axis=0)  # Add orphaned points as segments
                segments = segments[~np.any(np.isin(segments, point), axis=1)]   # Remove segments containing 'point'

        return segments[np.argsort(segments[:, 0])] # Sort the segments


    def _unique_transform(self, arr):
        '''
        Transforms a numpy array such that all unique values are mapped to a contiguous range starting from 0.
        This allows removal of items from the skeleton array while maintining the correct indexing for the marker columns.
        
        Args:
        - arr (numpy.ndarray): Input array to be transformed.

        Returns:
        - numpy.ndarray: Transformed array with values replaced by their mapped indices.
        '''
        flat_arr = arr.flatten()
        unique_values = np.unique(flat_arr)  # Get unique values
        value_mapping = {val: idx for idx, val in enumerate(unique_values)}  # Create mapping from value to index
        transformed_flat_arr = np.vectorize(value_mapping.get)(flat_arr)  # Apply mapping to flattened array
        transformed_arr = transformed_flat_arr.reshape(arr.shape)  # Reshape back to original shape
        return transformed_arr
    


class Controller():
    def __init__(self):
        '''Initialise the application with PCA backend processing.'''
        #super().__init__()
        self.subj_pca_models = {}  # Stores all subject specific data
        self.svd_model = pca_model.SVD()
        self.view = PCA_View(self)

        self.current_project = ''
        self.data_file_path_list = []
        self.output_folder_path = ''
        self.current_subj = None
        self.weight_ui = []

        self.current_subject_id = tkinter.StringVar()
        self.current_eigenwalker_group_id = tkinter.StringVar()
        self.pca_mode_option_menu = tkinter.StringVar()
        self.delimiter_entry = tkinter.StringVar()
        self.freq_entry = tkinter.StringVar()
        self.del_rows_entry = tkinter.StringVar()
        self.del_markers_entry = tkinter.StringVar()
        self.gap_filling_option_menu = tkinter.StringVar()
        self.data_filtering_option_menu = tkinter.StringVar()
        self.centring_option_menu = tkinter.StringVar()
        self.align_orientation_option_menu = tkinter.StringVar()
        self.orientation_cutoff_freq_entry = tkinter.StringVar()
        self.normalisation_option_menu = tkinter.StringVar()
        self.weights_option_menu = tkinter.StringVar()
        self.coordinate_transformation_option_menu = tkinter.StringVar()
        self.pp_filter_option_menu = tkinter.StringVar()
        self.pv_filter_option_menu = tkinter.StringVar()
        self.pa_filter_option_menu = tkinter.StringVar()
        self.pm_filter_order_entry = tkinter.StringVar()
        self.pm_filter_cut_off_entry = tkinter.StringVar()
        self.loocv_checkbox = tkinter.BooleanVar()
        self.freq_harmonics_entry = tkinter.StringVar()
        self.plot_save_extension_option_menu = tkinter.StringVar()
        self.ev_num_of_pcs_entry = tkinter.StringVar()
        self.sine_approx_checkbox = tkinter.BooleanVar()
        self.amp_factors_entry = tkinter.StringVar()
        self.appearance_mode_option_menu = tkinter.StringVar()
        self.scaling_option_menu = tkinter.StringVar()
        self.tooltips_checkbox = tkinter.BooleanVar()

        self.eigenwalker_pca_runs = {}
        
        self.view.setup_ui()

        self.save_configuration = {
            'pca_mode_option_menu': self.pca_mode_option_menu,
            'delimiter_entry': self.delimiter_entry,
            'freq_entry': self.freq_entry,
            'del_rows_entry': self.del_rows_entry,
            'del_markers_entry': self.del_markers_entry,
            'gap_filling_option_menu': self.gap_filling_option_menu,
            'data_filtering_option_menu': self.data_filtering_option_menu,
            'centring_option_menu': self.centring_option_menu,
            'align_orientation_option_menu': self.align_orientation_option_menu,
            'orientation_cutoff_freq_entry': self.orientation_cutoff_freq_entry,
            'normalisation_option_menu': self.normalisation_option_menu,
            'weights_option_menu': self.weights_option_menu,
            'coordinate_transformation_option_menu': self.coordinate_transformation_option_menu,
            'pp_filter_option_menu': self.pp_filter_option_menu,
            'pv_filter_option_menu': self.pv_filter_option_menu,
            'pa_filter_option_menu': self.pa_filter_option_menu,
            'pm_filter_order_entry': self.pm_filter_order_entry,
            'pm_filter_cut_off_entry': self.pm_filter_cut_off_entry,
            'loocv_checkbox': self.loocv_checkbox,
            'freq_harmonics_entry': self.freq_harmonics_entry,
            'plot_save_extension_option_menu': self.plot_save_extension_option_menu,
            'ev_num_of_pcs_entry': self.ev_num_of_pcs_entry,
            'sine_approx_checkbox': self.sine_approx_checkbox,
            'amp_factors_entry': self.amp_factors_entry,
            'appearance_mode_option_menu': self.appearance_mode_option_menu,
            'scaling_option_menu': self.scaling_option_menu,
            'tooltips_checkbox': self.tooltips_checkbox
        }


    def run_analysis(self):
        '''Executes PCA processing when 'Run PCA' button is pushed, running in a separate thread.'''
        start_time = time.time()
        self.view.prepare_ui()

        # Start a thread for PCA processing
        PCA_Thread = Thread(target = self._run_analysis_thread)
        PCA_Thread.start()

        # Update the GUI while the thread is running
        while PCA_Thread.is_alive():
            self.view.update()

        # Wait for the thread to close safely before continuing
        PCA_Thread.join()

        # Finalise the UI components after PCA processing
        self.view.finalise_ui()

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        print(f"Program Took: {elapsed_time:.2f} seconds")


    def _run_analysis_thread(self):
        '''Run PCA on the subject data based on selected options.'''
        self.check_ui_settings()

        self._sync_pca_with_ui()
        
        self._preprocess_subjects()

        if self.pca_mode_option_menu.get() == 'All Subjects Together':
            self._apply_pca_all_subjects_together()
        else:
            self._apply_pca_separately()

        # Save a copy of the project file to record the state of the app used to generate the output
        self.save_project_file(
            f'{self.output_folder_path}/{os.path.basename(self.current_project).split(".")[0]}_metadata.txt'
        )


    def _preprocess_subjects(self):
        '''Preprocess the data for each subject'''

        for i, (subject_id, subj_pca_model) in enumerate(self.subj_pca_models.items()):
            subj_pca_model.df = subj_pca_model.raw_data.copy()
            #print(subj_pca_model.markers_to_del_set)
            transformations = {
                'Removing Markers': (subj_pca_model.remove_markers, subj_pca_model.markers_to_del_set),
                'Removing Rows': (subj_pca_model.remove_rows, subj_pca_model.rows_to_del_set),
                'Gap Filling': (subj_pca_model.fill_gaps, self.gap_filling_option_menu.get()),
                'Initial Data Format Checking': (subj_pca_model.check_data_format, subject_id),
                'Flipping Axes': (subj_pca_model.flip_axes, [subj_pca_model.subject_UI_settings[key].get() for key in ['flip_x', 'flip_y', 'flip_z']]),
                'Filtering': (subj_pca_model.filter, self.data_filtering_option_menu.get(), int(string_to_list(self.pm_filter_order_entry.get(), [0])[0]), string_to_list(self.pm_filter_cut_off_entry.get(), [-1])[0]),
                'Centring': (subj_pca_model.centre, self.centring_option_menu.get(), self.weight_ui, subj_pca_model.centre_refs),
                'Aligning Orientation': (subj_pca_model.align_orientation, self.align_orientation_option_menu.get(), subj_pca_model.centre_refs, string_to_list(self.orientation_cutoff_freq_entry.get(), [0.0])[0]),
                'Saving result for Bug Plot': (self.save_data_bug_plot, subj_pca_model),
                'Cartesian to Spherical': (subj_pca_model.coordinate_transformation, self.coordinate_transformation_option_menu.get()),
                'Normalising, Weighting & Centring': (subj_pca_model.norm_weight_centre, subj_pca_model.func_order, self.weights_option_menu.get(), self.weight_ui, self.normalisation_option_menu.get(), subj_pca_model.centre_refs),
                'Checking Data Formatting': (subj_pca_model.check_data_format, subject_id),
                f'Preprocessing Complete: {subject_id}\n': (lambda *args: None, ())
            }
            
            # Apply the transformations to the data
            for step_name, (transform_function, *args) in transformations.items():
                print(f'{step_name:<45}', end='\r', flush=True)
                try:
                    transform_function(subj_pca_model.df, *args)
                except Exception as e:
                    raise TerminatingError(str(e))

            preprocessed_path = f'{self.output_folder_path}/preprocessed/{subject_id.split(".")[0]}_preprocessed.csv'
            self.save_df_to_csv(preprocessed_path, subj_pca_model.df)
            self.view.progress_bar.set((i + 1) / (2 * len(self.subj_pca_models)))


    def _apply_pca_all_subjects_together(self):
        '''Apply PCA on combined data of all subjects.'''

        data_for_pca = []
        for id, subj_pca_model in self.subj_pca_models.items():
            data_for_pca.append(subj_pca_model.df)

        try:
            merged_data = self.merge_dataframes(data_for_pca)
        except Exception as e:
            raise TerminatingError(str(e))

        self.svd_model.fit(merged_data)  # Fit PCA on combined data of all subjects
        print(f'PCs:\n{self.svd_model.components[0:5].T[0:5]}')
        print(f'Eigenvalues:\n{self.svd_model.eigenvalues[0:5]}')

        press_naive, press_approx = pd.DataFrame(), pd.DataFrame()
        if self.loocv_checkbox.get():
            press_naive, press_approx = pca_model.loocv(merged_data.to_numpy(), 25, 500)

        for i, (subj_preprocessed_data, (id, subj_pca_model)) in enumerate(zip(data_for_pca, self.subj_pca_models.items())):

            self._process_and_save_pca_results(id, subj_pca_model, subj_preprocessed_data, press_naive, press_approx)
            self.view.progress_bar.set(0.5 + ((i + 1) / (2 * len(self.subj_pca_models))))


    def _apply_pca_separately(self):
        '''Apply PCA separately for each subject's data.'''

        for i, (id, subj_pca_model) in enumerate(self.subj_pca_models.items()):
            self.svd_model.fit(subj_pca_model.df)  # Fit PCA separately for each subject's data
            
            press_naive, press_approx = pd.DataFrame(), pd.DataFrame()
            if self.loocv_checkbox.get():
                press_naive, press_approx = pca_model.loocv(subj_pca_model.df.to_numpy(), 25, 50)

            self._process_and_save_pca_results(id, subj_pca_model, subj_pca_model.df, press_naive, press_approx)
            self.view.progress_bar.set(0.5 + ((i + 1) / (2 * len(self.subj_pca_models))))


    def merge_dataframes(self, data_to_combine):
        # Check if all dataframes have the same columns
        expected_columns = set(data_to_combine[0].columns)
        for i, df in enumerate(data_to_combine):
            current_columns = set(df.columns)
            missing_columns = expected_columns - current_columns
            extra_columns = current_columns - expected_columns
            if missing_columns:
                raise Exception(f'Subject {i} is missing columns {missing_columns} compared to subject 0')
            if extra_columns:
                raise Exception(f'Subject {i} has extra columns {extra_columns} compared to subject 0')

        # If all dataframes have the same columns, merge them
        return pd.concat(data_to_combine, ignore_index=True) #merged_subject_data
    

    def _process_and_save_pca_results(self, id, subj_pca_model, subj_preprocessed_data, press_naive, press_approx):
        '''Process and save PCA results for a given subject.'''
        print(f'{("Processing PCA Results"):<45}', end='\r', flush=True)
        try:
            results_df = subj_pca_model.postprocess_pca_results(  # TODO: Improve this
                subj_preprocessed_data.columns,
                self.svd_model.project(subj_preprocessed_data),
                self.svd_model.components,
                self.svd_model.eigenvalues,
                self.pp_filter_option_menu.get(),
                self.pv_filter_option_menu.get(),
                self.pa_filter_option_menu.get(),
                int(string_to_list(self.pm_filter_order_entry.get(), [0])[0]),
                string_to_list(self.pm_filter_cut_off_entry.get(), [-1])[0],
                np.array(string_to_list(self.freq_harmonics_entry.get())),
                PRESS_Naive = press_naive,
                PRESS_Approx = press_approx
                # Add custom outputs as kwargs here ....
            )
            self.save_df_to_csv(subj_pca_model.results_file_path, results_df)
        except Exception as e:
            raise TerminatingError(str(e))
        
        print(f'Results Saved: {os.path.basename(subj_pca_model.results_file_path)}')


    def calc_eigenwalkers(self):

        groups = {}
        for subj_id, subj_pca_model in self.subj_pca_models.items():
            group_id = subj_pca_model.subject_UI_settings['eigenwalker_group'].get()
            if group_id != '':
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(subj_pca_model)

                print(f'{group_id}: {subj_id}')

        keys_to_remove = [group_id for group_id, group in groups.items() if len(group) <= 1]
        for key in keys_to_remove:
            groups.pop(key, None)

        group_ids = list(groups.keys())
        self.view.eigenwalker_group_option_menu.configure(values=group_ids)
        if self.view.eigenwalker_group_option_menu.get() not in group_ids:
            self.view.eigenwalker_group_option_menu.set(group_ids[0])

        self.eigenwalker_pca_runs = {}

        for group_id, group_subj_models in groups.items():
            all_dfs_in_group = [model.results.copy() for model in group_subj_models if not model.results.empty]

            for wtype in ['structural', 'dynamic', 'full']: # Order here is important
                eigenwalker_pca_run = pca_model.TrojePCA(2)
                W = eigenwalker_pca_run.preprocess(all_dfs_in_group, walker_type=wtype)
                W_0, V, K, eigenvalues = eigenwalker_pca_run.fit_troje(W)
                eigenwalker_results = eigenwalker_pca_run.process_results(W)

                self.save_df_to_csv(f'{self.output_folder_path}/eigenwalkers/{group_id}_{wtype}.csv', eigenwalker_results)

                '''for subj in group_subj_models:
                    # Save reconstructed walking data
                    num_eigen_features = subj.results.loc['Loadings'].shape[0]
                    bug_data = eigenwalker_pca_run.reconstruct(..., num_eigen_features,
                                                        float(self.freq_entry.get()),
                                                        float(subj.results.loc['Normalisation Factor'].to_numpy()[0, 0]),
                                                        subj.results.loc['Weight Vector'].to_numpy()[0],
                                                        self.coordinate_transformation_option_menu.get()
                                                        )
                    bug_data_df = pd.DataFrame(data=bug_data)
                    self.save_df_to_csv(f'{self.output_folder_path}/eigenwalkers/{group_id}_{wtype}_{os.path.basename(subj.results_file_path)}', bug_data_df)
                  '''  

            self.eigenwalker_pca_runs[group_id] = (group_subj_models[0], eigenwalker_pca_run)

            # Now calculate the structural and dynamic reconstruction results (this was used for the survey)
            W_bar = eigenwalker_pca_run.W_0.T[0]

            num_eigen_features = 102
            num_subj = eigenwalker_pca_run.K.shape[1]

            W_S = np.zeros((num_subj, 308)) # Structural Only

            for i in range(num_subj):
                W_S[i] = np.concatenate([W[i][:num_eigen_features], W_bar[num_eigen_features:]])
            
            W_D = np.zeros((num_subj, 308)) # Dynamic Only
            for i in range(num_subj):
                W_D[i] = np.concatenate([W_bar[:num_eigen_features], W[i][num_eigen_features:]])

            # Save reconstructed walking data
            for i, subj in enumerate(group_subj_models):
                num_eigen_features = subj.results.loc['Loadings'].shape[0]
                static_subj = eigenwalker_pca_run.reconstruct(W_S[i], num_eigen_features,
                                            float(self.freq_entry.get()),
                                            float(subj.results.loc['Normalisation Factor'].to_numpy()[0, 0]),
                                            subj.results.loc['Weight Vector'].to_numpy()[0],
                                            self.coordinate_transformation_option_menu.get()
                                            )
                static_subj_df = pd.DataFrame(data=static_subj)
                self.save_df_to_csv(f'{self.output_folder_path}/eigenwalkers/structural/structural_{group_id}_{os.path.basename(subj.results_file_path)}', static_subj_df)

                static_subj = eigenwalker_pca_run.reconstruct(W_D[i], num_eigen_features,
                                            float(self.freq_entry.get()),
                                            float(subj.results.loc['Normalisation Factor'].to_numpy()[0, 0]),
                                            subj.results.loc['Weight Vector'].to_numpy()[0],
                                            self.coordinate_transformation_option_menu.get()
                                            )
                static_subj_df = pd.DataFrame(data=static_subj)
                self.save_df_to_csv(f'{self.output_folder_path}/eigenwalkers/dynamic/dynamic_{group_id}_{os.path.basename(subj.results_file_path)}', static_subj_df)


        '''# Calculate static and dynamic reconstructions
        front_full_path = dir + 'B/B1_ALL_pca_each_separately/eigenwalkers/Front_full.csv'
        front_full_results = pd.read_csv(front_full_path, delimiter=',', header=0, index_col=0)
        front_W_bar = front_full_results.loc['W_0']

        hind_full_path = dir + 'B/B1_ALL_pca_each_separately/eigenwalkers/Hind_full.csv'
        hind_full_results = pd.read_csv(hind_full_path, delimiter=',', header=0, index_col=0)
        hind_W_bar = hind_full_results.loc['W_0']

        num_eigen_features = 102

        # Front
        W_F_S = np.zeros((10, 308)) # Static Only
        for i in range(10):
            W_F_S[i] = pd.concat([front_full_results.loc[f'W_{i + 1}'][0:num_eigen_features], front_W_bar[num_eigen_features:]])

        # Front
        W_F_D = np.zeros((10, 308)) # Dynamic Only
        for i in range(10):
            W_F_D[i] = pd.concat([front_W_bar[0:num_eigen_features], front_full_results.loc[f'W_{i + 1}'][num_eigen_features:]])

        # Hind
        W_H_S = np.zeros((24, 308)) # Static Only
        for i in range(24):
            W_H_S[i] = pd.concat([hind_full_results.loc[f'W_{i + 1}'][0:num_eigen_features], hind_W_bar[num_eigen_features:]])

        # Hind
        W_H_D = np.zeros((24, 308)) # Dynamic Only
        for i in range(24):
            W_H_D[i] = pd.concat([hind_W_bar[0:num_eigen_features], hind_full_results.loc[f'W_{i + 1}'][num_eigen_features:]])'''



    def check_ui_settings(self):
        if not self.current_project:
            raise TerminatingError('Project must be saved before PCA can be run.')

        if not self.subj_pca_models:
            raise TerminatingError('No data files provided.')
        
        if not self.freq_entry.get():
            raise TerminatingError('No sample frequency provided.')
            
        # Check for missing parameters
        if (self.data_filtering_option_menu.get() == 'Butterworth' or
            self.pp_filter_option_menu.get() == 'Low Pass Butterworth' or
            self.pv_filter_option_menu.get() == 'Low Pass Butterworth' or
            self.pa_filter_option_menu.get() == 'Low Pass Butterworth'):
                
            if not self.pm_filter_order_entry.get():
                raise TerminatingError('No order for butterworth filter provided')
            
            if not self.pm_filter_cut_off_entry.get():
                raise TerminatingError('No cutoffs for butterworth filter provided')


    def open_data_files(self, file_paths=[]):
        '''
        Open and process subject files, updating the file path label and table.
        '''

        if not self.freq_entry.get():
            raise TerminatingError('No sample frequency provided.')

        if not file_paths:
            file_paths = self.view._ask_open_filenames()

        if file_paths:
            self.data_file_path_list = file_paths

        if len(self.data_file_path_list) == 0:
            return

        self.view.update_file_path_label(self.data_file_path_list)

        # Update subject selection dropdown and related attributes.
        subject_basenames = [os.path.basename(path) for path in self.data_file_path_list]

        if not subject_basenames:
            return False
        
        # Check if any data files cannot be found
        for i, id in enumerate(subject_basenames):
            if not os.path.exists(self.data_file_path_list[i]):
                raise TerminatingError(f'Data file for subject "{id}" could not be found at path: {self.data_file_path_list[i]}.')
        

        self.view.subject_selection_option_menu_1.configure(values=subject_basenames)
        self.view.subject_selection_option_menu_2.configure(values=subject_basenames)
        self.view.subject_selection_option_menu_3.configure(values=subject_basenames)
        self.current_subject_id.set(subject_basenames[0])
        
        # Retrieve table data to save it from being erased
        temp_table_data = [self.view._get_table_values('weights'), self.view._get_table_values('centre_ref'), self.view._get_table_values('skeleton'), self.view._get_table_values('colour')]

        # Clear existing table entries
        for item in self.view.table.get_children():
            self.view.table.delete(item)

        # Get delimiter from entry or set to None
        delim = self.delimiter_entry.get() or None

        raw_data_subj_0 = pd.read_csv(self.data_file_path_list[0], delimiter=delim, header=0)

        # Group columns by their base name (e.g., 'marker_x', 'marker_y', 'marker_z')
        column_struct = [list(group) for _, group in itertools.groupby(
            list(raw_data_subj_0), key=lambda string: string[:-1]
        )]

        # Extract markers (groups of columns with the same base name)
        markers = [group for group in column_struct if len(group) == 3]

        # Initialise all the PCASubject objects
        self.subj_pca_models = {}  # Clear existing subject PCAs
        for i, id in enumerate(subject_basenames):
            raw_data = pd.read_csv(self.data_file_path_list[i], delimiter=delim, header=0)
            self.subj_pca_models[id] = Subject(raw_data.loc[:, np.concatenate(markers)], string_to_list(self.freq_entry.get())[0])

        # Update the subject selection in the UI if there are subjects
        if subject_basenames:
            self.view._update_subject_selection(subject_basenames[0])

        # Populate the table with markers and their children as formatted strings
        for i, marker in enumerate(markers):
            children_suffixes = ','.join([var[-1] for var in marker])
            parent_text = f'{i:<5} {marker[0][:-1]}({children_suffixes})'
            self.view.table.insert('', tkinter.END, text=parent_text, values=('1.0', '0.0', '', '0'), iid=i, open=False, tags=(str(i),))

        # Update table values with weights if available
        for i, marker in enumerate(markers):
            if i < len(temp_table_data[0]):
                self.view.table.set(i, '#1', temp_table_data[0][i])
                self.view.table.set(i, '#2', temp_table_data[1][i])
                self.view.table.set(i, '#3', temp_table_data[2][i])
                self.view.table.set(i, '#4', temp_table_data[3][i])

        self._sync_pca_with_ui()

        return True


    def save_project_file(self, save_path=None):
        '''
        Save the current project to a file.
        If no current project is set, prompt the user to choose a save location.
        '''

        if not save_path:
            if not self.current_project:
                self.current_project = self.view._ask_save_as_filename()
                if not self.current_project:
                    return
                
                if not self.current_project.endswith('.pca'):
                    self.current_project += '.pca'
            
            self.view.title(f'PMAnalyserPython [{self.current_project}]')

            save_path = self.current_project

        # Gather project data into a dictionary
        project_dict = {}
        for key, value in self.save_configuration.items():
            if isinstance(value, tkinter.StringVar):
                value = str(value.get())
            elif isinstance(value, (tkinter.BooleanVar, tkinter.IntVar)):
                value = value.get()

            project_dict[key] = value 

        project_dict['current_project'] = self.current_project
        project_dict['data_file_path_list'] = self.data_file_path_list
        project_dict['subject_UI_settings_dict'] = {
                key: {sub_key: sub_value.get() for sub_key, sub_value in value.subject_UI_settings.items()}
                for key, value in self.subj_pca_models.items()
            }
        project_dict['weights'] = self.view._get_table_values('weights')
        project_dict['centre_ref_column'] = self.view._get_table_values('centre_ref')
        project_dict['skeleton'] = self.view._get_table_values('skeleton')
        project_dict['colour'] = self.view._get_table_values('colour')


        with open(save_path, 'w') as project_file:
            project_file.write(json.dumps(project_dict, indent=2))

        self._sync_pca_with_ui()

        print(f'PROJECT SAVED: {save_path}')


    def load_project_file(self, project_to_open=''):
        '''
        Load a project from a file.
        If no file is specified, prompt the user to choose one.
        '''
        if self.current_project:
            self.view._on_closing(return_action=True)

        if not project_to_open:
            project_to_open = self.view._ask_open_filename()
            
            if not project_to_open:
                return
            
            if not project_to_open.endswith('.pca'):
                raise TerminatingError("Not a valid '.pca' project file")
    
        self.current_project = project_to_open
        self.view.title(f'PMAnalyserPython [{self.current_project}]')

        try:
            with open(self.current_project, 'r') as project_file:
                loaded_project_dict = json.load(project_file)
        except json.JSONDecodeError as e:
            raise TerminatingError(f'Invalid JSON format: {e}')

        self.data_file_path_list =  loaded_project_dict.get('data_file_path_list', [])
        
        # Load all of the variables
        for key, setter in self.save_configuration.items():
            value = loaded_project_dict.get(key)
            if value is not None:
                setter.set(value)
            else:
                print(f'Invalid JSON format. Missing: {key}')
        
        files_provided = self.open_data_files(self.data_file_path_list)

        if files_provided:
            # Sets the weights and reference columns in the table
            weights = loaded_project_dict.get('weights', [])
            num_of_table_rows = len(weights)
            for i in range(num_of_table_rows):
                self.view.table.set(i, '#1', weights[i])
                self.view.table.set(i, '#2', loaded_project_dict.get('centre_ref_column', np.zeros(num_of_table_rows))[i])
                self.view.table.set(i, '#3', loaded_project_dict.get('skeleton', [''] * num_of_table_rows)[i])
                self.view.table.set(i, '#4', loaded_project_dict.get('colour', ['0'] * num_of_table_rows)[i])

        for key, value in loaded_project_dict.get('subject_UI_settings_dict', {}).items():
            if key in self.subj_pca_models:
                self.subj_pca_models[key].subject_UI_settings = {
                    'rows_to_del': tkinter.StringVar(value=value.get('rows_to_del', '')),
                    'markers_to_del': tkinter.StringVar(value=value.get('markers_to_del','')),
                    'flip_x': tkinter.BooleanVar(value=value.get('flip_x', False)),
                    'flip_y': tkinter.BooleanVar(value=value.get('flip_y', False)),
                    'flip_z': tkinter.BooleanVar(value=value.get('flip_z', False)),
                    'eigenwalker_group': tkinter.StringVar(value=value.get('eigenwalker_group', ''))
                }
        
        self.view.update_appearance()
        self.view.update_scaling()
        self.view.progress_bar.grid_remove()
        self.view.animation_saving_progress_bar.grid_remove()

        for id, subj_pca_model in self.subj_pca_models.items():
            if self.subj_pca_models[id].results_file_path and subj_pca_model.results.empty:
                subj_pca_model.results = self._read_csv_safe(subj_pca_model.results_file_path)

        self._sync_pca_with_ui()

        # Update the subject selection in the UI if there are subjects
        if self.subj_pca_models:
            self.view._update_subject_selection(list(self.subj_pca_models.keys())[0])

        print(f'PROJECT LOADED: {self.current_project}')


    def save_data_bug_plot(self, df, subj_pca_model):
        subj_pca_model.bug_plot_data = df.copy()


    def _string_to_index(self, index_string, list_length):
        '''Convert a string of indices to a list of integers.'''
        index_string = index_string.replace('[', '').replace(']', '').replace(' ', '')
        if not index_string:
            return []
        
        indices = []
        for item in index_string.split(','):
            parts = item.split(':')

            # Handle cases where slice start, stop, or step might be empty
            start = int(parts[0]) if parts[0] != '' else None
            stop = int(parts[1]) if len(parts) > 1 and parts[1] != '' else None
            step = int(parts[2]) if len(parts) > 2 and parts[2] != '' else None
            
            # Adjust for None values and negative indices
            if start is None:
                start = 0
            elif start < 0:
                start = list_length + start + 1
                
            if stop is None:
                stop = list_length
            elif stop < 0:
                stop = list_length + stop + 1

            # If no slice, just append the start value
            if len(parts) == 1:
                indices.append(start)
            else:
                # Convert slice to a list of indices and extend the main list
                slice_range = slice(start, stop, step)
                indices.extend(range(slice_range.start, slice_range.stop, slice_range.step or 1))

        return indices


    def _read_csv_safe(self, csv_path):
        '''Read a CSV file safely. Used only for reading result CSV files generated by running PCA using the application.

        Args:
            csv_path (str): The path to the CSV file.

        Returns:
            pandas.DataFrame: The DataFrame read from the CSV file.
        '''
        df = pd.DataFrame()

        try:
            df = pd.read_csv(csv_path, low_memory=False)
            df.set_index(['level_0', 'level_1'], inplace=True) # Convert the first three columns to a MultiIndex
        except Exception as e:
            pass  #raise TerminatingError(str(e))
        
        return df
    

    def change_current_subj(self, subject_id):
        self.current_subj = self.subj_pca_models[subject_id]


    def _sync_pca_with_ui(self):
        '''
        Updates metadata for each subject by processing markers to be deleted,
        modifying the skeleton, and setting line colors and CSV file paths.
        '''

        # Define the path for the CSV output files
        pca_mode_name_component = 'pca_each_separately'
        if self.pca_mode_option_menu.get() == 'All Subjects Together':
            pca_mode_name_component = 'pca_all_together'

        self.output_folder_path = f'{os.path.dirname(self.current_project)}/{os.path.basename(self.current_project).split(".")[0]}_{pca_mode_name_component}'

        for id, subj_pca_model in self.subj_pca_models.items():
            # Combine and deduplicate markers to be deleted
            num_of_markers = subj_pca_model.raw_data.shape[1] // 3

            markers_to_del_temp = list(set(
                self._string_to_index(subj_pca_model.subject_UI_settings['markers_to_del'].get(), num_of_markers) + 
                self._string_to_index(self.del_markers_entry.get(),  num_of_markers)
            ))

            # Store updated marker deletion set and skeleton in subject metadata
            subj_pca_model.markers_to_del_set = markers_to_del_temp
            self.weight_ui = np.array([w for i, w in enumerate(self.view._get_table_values('weights')) if i not in list(subj_pca_model.markers_to_del_set)])
            subj_pca_model.func_order = 4 if self.normalisation_option_menu.get() == 'Mean Dist. 2 Markers (Centre Ref.)' else 0
            subj_pca_model.centre_refs = np.delete(self.view._get_table_values('centre_ref'), list(subj_pca_model.markers_to_del_set))
            subj_pca_model.skeleton = list(enumerate(self.view._get_table_values('skeleton')))
            subj_pca_model.line_colors_raw = np.array(self.view._get_table_values('colour'))
            subj_pca_model.sample_freq = string_to_list(self.freq_entry.get())[0]

            num_rows = subj_pca_model.raw_data.shape[0]
            rows_to_del_temp = list(set(
                self._string_to_index(subj_pca_model.subject_UI_settings['rows_to_del'].get(), num_rows) +
                self._string_to_index(self.del_rows_entry.get(), num_rows)
            ))
            subj_pca_model.rows_to_del_set = rows_to_del_temp
            subj_pca_model.results_file_path = f'{self.output_folder_path}/{id.split(".")[0]}_pca.csv' 
            subj_pca_model.mean_of_data = []

        file_paths_str_list = [subject.results_file_path for subject in self.subj_pca_models.values()]

        if len(file_paths_str_list) > 0:
            # Update the PCA file path textbox in the UI
            self.view.update_pca_file_path_textbox(file_paths_str_list)

    
    def save_plots(self):
        '''Save all plots.'''
        print('Saving Plots')
        self.view.fig_ev.savefig(f'{self.output_folder_path}/EV_Plots.{self.plot_save_extension_option_menu.get()}', dpi=300)
        for i, score_plot_tab_name in enumerate(self.view.score_plot_tab_names):
            self.view.fig_score_plots[i].savefig(f'{self.output_folder_path}/Score_Plot_{score_plot_tab_name}.{self.plot_save_extension_option_menu.get()}', dpi=300)
    

    def save_animated_plots(self):
        '''Save animated plots.'''
        print('Saving Animations')
        path = f'{self.output_folder_path}/animations'
        if not os.path.exists(path):
            os.makedirs(path)

        subj_name = self.current_subject_id.get().split(".")[0]
        stages = [
            (0.0, self.view._show_bug_plot),
            (0.05, lambda: self.view.bug_plot_anim.save(f'{path}/3D_Input_Data_Animation_{subj_name}.mp4', fps=30, dpi=150)),
            (0.25, lambda: self.view._video_plot_tab_selection(2)),
            (0.30, lambda: self.view.loadings_anim.save(f'{path}/PC_Loadings_Animation_{subj_name}.mp4', fps=30, dpi=150)),
            (0.45, lambda: self.view._video_plot_tab_selection(3)),
            (0.50, lambda: self.view.reconstruction_anim.save(f'{path}/PC_Reconstruction_Animation_{subj_name}.mp4', fps=30, dpi=150)),
            (0.70, lambda: self.view._video_plot_tab_selection(1)),
            (0.75, lambda: self.view.time_series_anim.save(f'{path}/Time_Series_Animation_{subj_name}.mp4', fps=30, dpi=150)),
            (0.95, self.view._video_plot_tab_selection)
        ]

        self.view.animation_saving_progress_bar.grid()
        for progress, action in stages:
            self.view.animation_saving_progress_bar.set(progress)
            self.view.update_idletasks()
            try:
                action()
            except AttributeError as e:
                print(f"Skipping the saving of animation plot as it's not available. {e}")
                pass

        self.view.animation_saving_progress_bar.grid_remove()


    def save_df_to_csv(self, file_path, df):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        try:
            df.to_csv(file_path, encoding='utf-8', index=True)
        except Exception as e:
            raise Exception(f'Could not save CSV file {os.path.dirname(file_path)}: {e}')




if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=5)

    controller = Controller()
    controller.view.mainloop()
