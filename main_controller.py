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
    pip install -r requirements.txt
    
"""
import os
from threading import Thread
import json
import tkinter as tk
import itertools
import time
import gc
from dataclasses import dataclass, field, fields
import re
import sys

import pandas as pd
import numpy as np

import pca_model
from pca_view import PCA_View, TerminatingError, Warning, isfloat, isposint

@dataclass
class Configuration:
    pca_mode: tk.StringVar = field(default_factory=tk.StringVar)
    delimiter: tk.StringVar = field(default_factory=tk.StringVar)
    freq: tk.StringVar = field(default_factory=tk.StringVar)
    del_rows: tk.StringVar = field(default_factory=tk.StringVar)
    del_markers: tk.StringVar = field(default_factory=tk.StringVar)
    gap_filling: tk.StringVar = field(default_factory=tk.StringVar)
    data_filtering: tk.StringVar = field(default_factory=tk.StringVar)
    centring: tk.StringVar = field(default_factory=tk.StringVar)
    align_orientation: tk.StringVar = field(default_factory=tk.StringVar)
    orientation_cutoff_freq: tk.StringVar = field(default_factory=tk.StringVar)
    normalisation: tk.StringVar = field(default_factory=tk.StringVar)
    weights_mode: tk.StringVar = field(default_factory=tk.StringVar)
    coordinate_transformation: tk.StringVar = field(default_factory=tk.StringVar)
    pp_filter: tk.StringVar = field(default_factory=tk.StringVar)
    pv_filter: tk.StringVar = field(default_factory=tk.StringVar)
    pa_filter: tk.StringVar = field(default_factory=tk.StringVar)
    pm_filter_order: tk.StringVar = field(default_factory=tk.StringVar)
    pm_filter_cut_off: tk.StringVar = field(default_factory=tk.StringVar)
    loocv: tk.BooleanVar = field(default_factory=tk.BooleanVar)
    freq_harmonics: tk.StringVar = field(default_factory=tk.StringVar)
    plot_save_extension: tk.StringVar = field(default_factory=tk.StringVar)
    ev_num_of_pcs: tk.StringVar = field(default_factory=tk.StringVar)
    sine_approx: tk.BooleanVar = field(default_factory=tk.BooleanVar)
    amp_factors: tk.StringVar = field(default_factory=tk.StringVar)
    appearance_mode: tk.StringVar = field(default_factory=tk.StringVar)
    scaling: tk.StringVar = field(default_factory=tk.StringVar)
    tooltips: tk.BooleanVar = field(default_factory=tk.BooleanVar)

    @property
    def get_delimiter(self):
        value = self.delimiter.get()
        if value:
            if any(char.isdigit() for char in value):
                raise TerminatingError(f"Invalid format in 'Delimiter' entry: '{value}'. Entry should not contain numbers.")
            return value
        return None
    
    @property
    def get_freq(self):
        value = self.freq.get()
        if not isfloat(value):
            raise TerminatingError(f"Sample Frequency '{value}' is not a valid number.")
        value = float(value)
        if value <= 0:
            raise TerminatingError(f"Sample Frequency '{value}' must be positive.")
        return value

    @property
    def get_del_rows(self):
        value = self.del_rows.get()
        if not _validate_index_string(value):
            raise TerminatingError(f"Invalid format in 'Delete Rows' entry: '{value}'.")
        return value
    
    @property
    def get_del_markers(self):
        value = self.del_markers.get()
        if not _validate_index_string(value):
            raise TerminatingError(f"Invalid format in 'Delete Markers' entry: '{value}'.")
        return value
    
    @property
    def get_orientation_cutoff_freq(self):
        value = self.orientation_cutoff_freq.get()
        if value:
            if not isfloat(value):
                raise TerminatingError(f"Orientation Cutoff Freq. '{value}' is not a valid number.")
            value = float(value)
            if value <= 0:
                raise TerminatingError(f"Orientation Cutoff Freq. '{value}' must be positive.")
            return value
        return 0
    
    @property
    def get_pm_filter_order(self):
        value = self.pm_filter_order.get()
        if value:
            if not isposint(value):
                raise TerminatingError(f"Invalid format in 'PM Filter Order' entry: '{value}'.")
            return int(value)
        return 0
    
    @property
    def get_pm_filter_cut_off(self):
        value = self.pm_filter_cut_off.get()
        if value:
            if value and not isposint(value):
                raise TerminatingError(f"Invalid format in 'PM Filter Cutoff' entry: '{value}'.")
            return int(value)
        return 0
    
    @property
    def get_ev_num_of_pcs(self):
        value = self.ev_num_of_pcs.get()
        if value:
            if not isposint(value):
                raise TerminatingError(f"Invalid format in 'Number of PCs to plot' entry: '{value}'.")
            return int(value)
        return 12  # Defaults to 12
    
    @property
    def get_freq_harmonics(self):
        value = self.freq_harmonics.get()
        try:
            value = string_to_list(value)
        except Exception:
            raise TerminatingError(f"Invalid format in 'Sine Approx. Freq. Ratios' entry: '{value}'.")
        return value
    
    @property
    def get_amp_factors(self):
        value = self.amp_factors.get()
        try:
            value = string_to_list(value)
        except Exception:
            raise TerminatingError(f"Invalid format in 'PM Amplification Factors' entry '{value}'.")
        return value

    def set_all(self, dict):
        for field in fields(self):
            key = field.name  # Field name is the key
            tk_variable = getattr(self, key)  # Get the tk.Variable (e.g., StringVar, BooleanVar)

            value = dict.get(key)  # Retrieve the value from the loaded dictionary
            if value is not None:
                tk_variable.set(value)  # Set the value on the Tkinter variable
            else:
                print(f'Invalid format. Missing: {key}')


    def get_raw(self):
        dict = {}
        for field_name in vars(self):
            field_value = getattr(self, field_name)
            dict[field_name] = field_value.get()

        return dict


class Subject(pca_model.PCA_Model):
    def __init__(self, df, sample_freq):
        super().__init__(sample_freq)
        self.raw_data = df.copy()
        self.df = df.copy()
        self.markers_to_del_set = []
        self.rows_to_del_set = []
        self.func_order = 0
        self.weights = []
        self.centre_refs = []
        self.skeleton = []
        self.line_colours = []
        self.mean_of_data = []
        self.bug_plot_data = pd.DataFrame()
        self.results_file_path = None
        self.subject_UI_settings = {
                    'rows_to_del': tk.StringVar(),
                    'markers_to_del': tk.StringVar(),
                    'flip_x': tk.BooleanVar(),
                    'flip_y': tk.BooleanVar(),
                    'flip_z': tk.BooleanVar(),
                    'eigenwalker_population': tk.StringVar(),
                    'eigenwalker_group': tk.StringVar()
                }
    

class Controller():
    def __init__(self):
        '''Initialise the application with PCA backend processing.'''
        self.svd_model = pca_model.SVD()
        self.view = PCA_View(self)
        self.project_dir = ''
        self.project_name = ''
        self.init_empty_project()

    def init_empty_project(self):
        self.subj_pca_models = {}
        self.data_file_path_list = []
        self.output_folder_path = ''
        self.current_subj = None
        self.g_weight = np.array([])
        self.g_centre_ref = np.array([])
        self.g_skeleton = np.array([], dtype=object)
        self.g_colour = np.array([], dtype=object)
        self.eigenwalker_pca_runs = {}
        self.current_subject_id = tk.StringVar()
        self.current_eigenwalker_population_id = tk.StringVar()
        self.subject_UI_settings_dict = {}
        self.configuration = Configuration()
        self.view.setup_ui()

    def run_analysis(self):
        '''Executes PCA processing when 'Run PCA' button is pushed, running in a separate thread.'''
        gc.collect()
        
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
        if not self.project_dir:
            raise TerminatingError('Project must be saved before PCA can be run.')
        
        self.open_data_files(self.data_file_path_list, ask_open=False)

        if not self.subj_pca_models:
            raise TerminatingError('No data files provided.')

        self._sync_pca_with_ui()

        self._preprocess_subjects()

        print(f'Saving results to: {self.output_folder_path}')
        if self.configuration.pca_mode.get() == 'All Subjects Together':
            self._apply_pca_all_subjects_together()
        else:
            self._apply_pca_separately()

        self.calc_eigenwalkers()

        # Save a copy of the project file to record the state of the app used to generate the output
        self.save_project_file(
            f'{self.output_folder_path}/{self.project_name.split(".")[0]}_metadata.txt'
        )


    def _preprocess_subjects(self):
        '''Preprocess the data for each subject'''
        total_steps = len(self.subj_pca_models.items())
        for i, (subject_id, subj_pca_model) in enumerate(self.subj_pca_models.items()):
            subj_pca_model.df = subj_pca_model.raw_data.copy()
            #print(subj_pca_model.markers_to_del_set)
            transformations = {
                'Removing Markers': (subj_pca_model.remove_markers, subj_pca_model.markers_to_del_set),
                'Removing Rows': (subj_pca_model.remove_rows, subj_pca_model.rows_to_del_set),
                'Gap Filling': (subj_pca_model.fill_gaps, self.configuration.gap_filling.get()),
                'Initial Data Format Checking': (subj_pca_model.check_data_format, subject_id),
                'Flipping Axes': (subj_pca_model.flip_axes, [subj_pca_model.subject_UI_settings[key].get() for key in ['flip_x', 'flip_y', 'flip_z']]),
                'Filtering': (subj_pca_model.filter, self.configuration.data_filtering.get(), self.configuration.get_pm_filter_order, self.configuration.get_pm_filter_cut_off),
                'Centring': (subj_pca_model.centre, self.configuration.centring.get(), subj_pca_model.weights, subj_pca_model.centre_refs),
                'Aligning Orientation': (subj_pca_model.align_orientation, self.configuration.align_orientation.get(), subj_pca_model.centre_refs, self.configuration.get_orientation_cutoff_freq),
                'Saving result for Bug Plot': (self.save_data_bug_plot, subj_pca_model),
                'Cartesian to Spherical': (subj_pca_model.coordinate_transformation, self.configuration.coordinate_transformation.get()),
                'Normalising, Weighting & Centring': (subj_pca_model.norm_weight_centre, subj_pca_model.func_order, self.configuration.weights_mode.get(), subj_pca_model.weights, self.configuration.normalisation.get(), subj_pca_model.centre_refs),
                'Checking Data Formatting': (subj_pca_model.check_data_format, subject_id),
                #f'Preprocessing Complete: {subject_id}\n': (lambda *args: None, ())
            }
            
            print_loading_bar(i, total_steps, f'{subject_id}')
            # Apply the transformations to the data
            for step_name, (transform_function, *args) in transformations.items():
                #print(f'{step_name:<45}', end='\r', flush=True)
                try:
                    transform_function(subj_pca_model.df, *args)
                except Exception as e:
                    raise TerminatingError(str(e))

            preprocessed_path = f'{self.output_folder_path}/preprocessed/{subject_id.split(".")[0]}_preprocessed.csv'
            self.save_df_to_csv(preprocessed_path, subj_pca_model.df)
            self.view.progress_bar.set((i + 1) / (2 * len(self.subj_pca_models)))

        print_loading_bar(100, 100, f'Preprocessed')


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
        if self.configuration.loocv.get():
            press_naive, press_approx = pca_model.loocv(merged_data.to_numpy(dtype=float), 25, 500)

        total_steps = len(self.subj_pca_models.items())
        for i, (subj_preprocessed_data, (id, subj_pca_model)) in enumerate(zip(data_for_pca, self.subj_pca_models.items())):
            self._process_and_save_pca_results(subj_pca_model, subj_preprocessed_data, press_naive, press_approx)
            self.view.progress_bar.set(0.5 + ((i + 1) / (2 * len(self.subj_pca_models))))
            print_loading_bar(i, total_steps, f'Saved: {os.path.basename(subj_pca_model.results_file_path)}')
        print_loading_bar(100, 100, f'Saved')

    def _apply_pca_separately(self):
        '''Apply PCA separately for each subject's data.'''
        total_steps = len(self.subj_pca_models.items())
        for i, (id, subj_pca_model) in enumerate(self.subj_pca_models.items()):
            self.svd_model.fit(subj_pca_model.df)  # Fit PCA separately for each subject's data
            
            press_naive, press_approx = pd.DataFrame(), pd.DataFrame()
            if self.configuration.loocv.get():
                press_naive, press_approx = pca_model.loocv(subj_pca_model.df.to_numpy(dtype=float), 25, 50)

            self._process_and_save_pca_results(subj_pca_model, subj_pca_model.df, press_naive, press_approx)
            self.view.progress_bar.set(0.5 + ((i + 1) / (2 * len(self.subj_pca_models))))
            print_loading_bar(i, total_steps, f'Saved: {os.path.basename(subj_pca_model.results_file_path)}')
        print_loading_bar(100, 100, f'Saved')


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
    

    def _process_and_save_pca_results(self, subj_pca_model, subj_preprocessed_data, press_naive, press_approx):
        '''Process and save PCA results for a given subject.'''
        #print(f'{("Processing PCA Results"):<45}', end='\r', flush=True)
        try:
            results_df = subj_pca_model.postprocess_pca_results(
                subj_preprocessed_data.columns,
                self.svd_model.project(subj_preprocessed_data),
                self.svd_model.components,
                self.svd_model.eigenvalues,
                self.configuration.pp_filter.get(),
                self.configuration.pv_filter.get(),
                self.configuration.pa_filter.get(),
                self.configuration.get_pm_filter_order,
                self.configuration.get_pm_filter_cut_off,
                np.array(self.configuration.get_freq_harmonics),
                PRESS_Naive = press_naive,
                PRESS_Approx = press_approx,
                Group = pd.DataFrame([0.0], index=[subj_pca_model.subject_UI_settings['eigenwalker_group'].get()], columns=["PC1"])
                # Add custom outputs as kwargs here ....
            )
            self.save_df_to_csv(subj_pca_model.results_file_path, results_df)
        except Exception as e:
            raise TerminatingError(str(e))
        
        #print(f'Results Saved: {os.path.basename(subj_pca_model.results_file_path)}')


    def calc_eigenwalkers(self):

        if not self.subj_pca_models:
            Warning(f"Cannot calculate Eigenwalker Space as there are no subjects.")
            return

        populations = {}
        for subj_id, subj_pca_model in self.subj_pca_models.items():
            population_id = subj_pca_model.subject_UI_settings['eigenwalker_population'].get()
            if population_id == '':
                population_id = "None"
            if population_id not in populations:
                populations[population_id] = []
            populations[population_id].append(subj_pca_model)
            #print(f'{population_id}: {subj_id}')

        keys_to_remove = [population_id for population_id, population in populations.items() if len(population) <= 1]
        for key in keys_to_remove:
            populations.pop(key, None)

        population_ids = list(populations.keys())
        self.view.eigenwalker_population_option_menu.configure(values=population_ids)

        if self.view.eigenwalker_population_option_menu.get() not in population_ids:
            self.view.eigenwalker_population_option_menu.set(population_ids[0])

        self.eigenwalker_pca_runs = {}

        unique_groups = set()
        for population_id, population_subj_models in populations.items():
            all_dfs_in_population = [model.results.copy() for model in population_subj_models if not model.results.empty]
            self.eigenwalker_pca_runs[population_id] = {}
            for wtype in ['structural', 'dynamic', 'full']: # Order here is important
                eigenwalker_pca_run = pca_model.EigenwalkerPCA(2)
                W, groups, grouped_average_walker = eigenwalker_pca_run.preprocess(all_dfs_in_population, wtype=wtype)
                eigenwalker_pca_run.fit_walkers(W)
                eigenwalker_results = eigenwalker_pca_run.process_results(W)
                eigenwalker_results = pd.concat([eigenwalker_results, pd.DataFrame({'PC1': groups}, index=[f'K_{i + 1}' for i in range(len(groups))])])
                self.save_df_to_csv(f'{self.output_folder_path}/eigenwalkers/{population_id}_{wtype}.csv', eigenwalker_results)
                
                projected_group_centres = {key: eigenwalker_pca_run.project_walkers(value.reshape(1, -1)).flatten() for key, value in grouped_average_walker.items()}
                self.eigenwalker_pca_runs[population_id][wtype] = (population_subj_models[0], eigenwalker_pca_run, projected_group_centres)

                unique_groups.update(groups)
        
        group_combinations = list(itertools.combinations(list(unique_groups), 2))
        formatted_combinations = [f"{a} - {b}" for a, b in group_combinations]
        self.view.eigenwalker_reconstruct_axes_option_menu.configure(values=['Eigenwalkers (PCs)'] + formatted_combinations)


    def open_data_files(self, file_paths=[], ask_open=True):
        '''Open and process subject files, updating the file path label and table.'''

        if not file_paths and ask_open:
            file_paths = self.view._ask_open_filenames()

        if file_paths:
            if self.project_dir:
                self.data_file_path_list = self.global_to_local(file_paths)
            else:
                self.data_file_path_list = file_paths

        #print(self.data_file_path_list)

        # If there are any missing files, issue a single warning
        missing_files = [file_path for file_path in self.data_file_path_list if not os.path.exists(file_path)]
        if missing_files:
            Warning(''.join(f'\nFile not found: {file_path}' for file_path in missing_files))
            
        # Remove data files that cannot be found
        self.data_file_path_list = [file_path for file_path in self.data_file_path_list if os.path.exists(file_path)]

        if not self.data_file_path_list:
            return

        # Store UI settings from existing subjects
        for i, (subject_id, subj_pca_model) in enumerate(self.subj_pca_models.items()):
            self.subject_UI_settings_dict[subject_id] = {sub_key: sub_value.get() for sub_key, sub_value in subj_pca_model.subject_UI_settings.items()}
        
        # Update subject selection dropdown and related attributes.
        self.view.update_file_path_label(self.data_file_path_list)
        subject_basenames = [os.path.basename(path) for path in self.data_file_path_list]
        
        self.view.subject_selection_option_menu_1.configure(values=subject_basenames)
        self.view.subject_selection_option_menu_2.configure(values=subject_basenames)
        self.view.subject_selection_option_menu_3.configure(values=subject_basenames)
        self.current_subject_id.set(subject_basenames[0])
        
        # Clear existing table entries
        for item in self.view.table.get_children():
            self.view.table.delete(item)

        self.subj_pca_models = {}  # Clear existing subject PCAs
        for i, id in enumerate(subject_basenames):
            raw_data = pd.read_csv(self.data_file_path_list[i], delimiter=self.configuration.get_delimiter or None, header=0)
            if i == 0:
                # Group columns by their base name (e.g., 'marker_x', 'marker_y', 'marker_z')
                column_struct = [list(group) for _, group in itertools.groupby(
                    list(raw_data), key=lambda string: string[:-1]
                )]
                # Extract markers (groups of columns with the same base name)
                markers = [group for group in column_struct if len(group) == 3]
                
                if len(markers) < 1:
                    raise TerminatingError("No 3D marker data found in the input (missing columns ending with x, y, z). The number of markers must be greater than 0.")
            
            self.subj_pca_models[id] = Subject(raw_data.loc[:, np.concatenate(markers)], self.configuration.get_freq)

            if id in self.subject_UI_settings_dict:
                saved_settings = self.subject_UI_settings_dict[id]
                self.subj_pca_models[id].subject_UI_settings = {
                    'rows_to_del': tk.StringVar(value=saved_settings.get('rows_to_del', '')),
                    'markers_to_del': tk.StringVar(value=saved_settings.get('markers_to_del','')),
                    'flip_x': tk.BooleanVar(value=saved_settings.get('flip_x', False)),
                    'flip_y': tk.BooleanVar(value=saved_settings.get('flip_y', False)),
                    'flip_z': tk.BooleanVar(value=saved_settings.get('flip_z', False)),
                    'eigenwalker_population': tk.StringVar(value=saved_settings.get('eigenwalker_population', '')),
                    'eigenwalker_group': tk.StringVar(value=saved_settings.get('eigenwalker_group', ''))
                }

        # Update the subject selection in the UI if there are subjects
        if subject_basenames:
            self.view._update_subject_selection(subject_basenames[0])

        # Populate the table with markers and their children as formatted strings
        for i, marker in enumerate(markers):
            children_suffixes = ','.join([var[-1] for var in marker])
            parent_text = f'{i:<5} {marker[0][:-1]}({children_suffixes})'
            self.view.table.insert('', tk.END, text=parent_text, values=('1.0', '0.0', '', '0'), iid=i, open=False, tags=(str(i),))

        self.g_weight = np.pad(self.g_weight, (0, max(0, len(markers) - len(self.g_weight))), mode='constant', constant_values=1.0)
        self.g_centre_ref = np.pad(self.g_centre_ref, (0, max(0, len(markers) - len(self.g_centre_ref))), mode='constant', constant_values=0.0)
        self.g_skeleton = np.pad(self.g_skeleton, (0, max(0, len(markers) - len(self.g_skeleton))), mode='constant', constant_values="")
        self.g_colour = np.pad(self.g_colour, (0, max(0, len(markers) - len(self.g_colour))), mode='constant', constant_values="0")

        for i in range(len(markers)):
            self.view.table.set(i, '#1', self.g_weight[i])
            self.view.table.set(i, '#2', self.g_centre_ref[i])
            self.view.table.set(i, '#3', self.g_skeleton[i])
            self.view.table.set(i, '#4', self.g_colour[i])

        self._sync_pca_with_ui()
    

    def set_table_vars(self):
        self.g_weight = np.array(self.view._get_table_values('weights')) #np.array([w for i, w in enumerate(self.view._get_table_values('weights')) if i not in list(subj_pca_model.markers_to_del_set)])
        self.g_centre_ref = np.array(self.view._get_table_values('centre_ref'))
        self.g_skeleton = np.array(self.view._get_table_values('skeleton'), dtype=object)
        self.g_colour = np.array(self.view._get_table_values('colour'), dtype=object)


    def global_to_local(self, global_file_paths):
        return [os.path.relpath(path, self.project_dir) for path in global_file_paths]


    def new_project_file(self):
        if self.project_dir:
            if self.view._on_closing(return_action=True) == 'cancel':
                return
            
        project_file_path = self.view._ask_save_as_filename()
        if not project_file_path:
            return
        
        if not project_file_path.endswith('.pca'):
            project_file_path += '.pca'

        self.project_dir = os.path.dirname(project_file_path)
        self.project_name = os.path.basename(project_file_path)

        os.chdir(self.project_dir)  # Switch to project's working directory for local paths
        print(f'Working Directory: {self.project_dir}')

        self.init_empty_project()

        save_path = os.path.join(self.project_dir, self.project_name)
        self.save_project_file(save_path)


    def save_project_file(self, save_path=None):
        '''
        Save the current project to a file.
        If no current project is set, prompt the user to choose a save location.
        '''

        if not save_path:
            if not self.project_dir or not self.project_name:
                project_file_path = self.view._ask_save_as_filename()
                if not project_file_path:
                    return
                
                if not project_file_path.endswith('.pca'):
                    project_file_path += '.pca'

                self.project_dir = os.path.dirname(project_file_path)
                self.project_name = os.path.basename(project_file_path)

            save_path = os.path.join(self.project_dir, self.project_name)
        self.view.title(f'PMAnalyserPython [{self.project_name}]')

        os.chdir(self.project_dir)  # Switch to project's working directory for local paths
        print(f'Working Directory: {self.project_dir}')
            
        # Gather project data into a dictionary
        project_dict = self.configuration.get_raw()

        # Additional settings to save
        project_dict['project_name'] = self.project_name
        project_dict['data_file_path_list'] = self.global_to_local(self.data_file_path_list)
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

        print(f'Project Saved: {save_path}')


    def load_project_file(self, project_to_open=''):
        '''
        Load a project from a file.
        If no file is specified, prompt the user to choose one.
        '''
        if self.project_dir:
             if self.view._on_closing(return_action=True) == 'cancel':
                return

        if not project_to_open:
            project_to_open = self.view._ask_open_filename()
            
            if not project_to_open:
                return
            
            if not project_to_open.endswith('.pca'):
                raise TerminatingError("Not a valid '.pca' project file")
    
        self.project_dir = os.path.dirname(project_to_open)
        self.project_name = os.path.basename(project_to_open)
        self.view.title(f'PMAnalyserPython [{self.project_name}]')
        os.chdir(self.project_dir)  # Switch to project's working directory for local paths
        print(f'Working Directory: {self.project_dir}')
        
        try:
            with open(self.project_name, 'r') as project_file:
                loaded_project_dict = json.load(project_file)
        except json.JSONDecodeError as e:
            raise TerminatingError(f'Invalid JSON format: {e}')

        self.data_file_path_list =  loaded_project_dict.get('data_file_path_list', [])

        self.configuration.set_all(loaded_project_dict)
        
        self.g_weight = np.array(loaded_project_dict.get('weights', []))
        self.g_centre_ref = np.array(loaded_project_dict.get('centre_ref_column', np.zeros(len(self.g_weight))))
        self.g_skeleton = np.array(loaded_project_dict.get('skeleton', [''] * len(self.g_weight)))
        self.g_colour = np.array(loaded_project_dict.get('colour', ['0'] * len(self.g_weight)))

        self.subject_UI_settings_dict = loaded_project_dict.get('subject_UI_settings_dict', {})

        self.open_data_files(self.data_file_path_list, ask_open=False)
        
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

        print(f'Project Loaded: {self.project_name}')


    def save_data_bug_plot(self, df, subj_pca_model):
        subj_pca_model.bug_plot_data = df.copy()


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
        if self.configuration.pca_mode.get() == 'All Subjects Together':
            pca_mode_name_component = 'pca_all_together'

        self.output_folder_path = f'{self.project_name.split(".")[0]}_{pca_mode_name_component}'

        for id, subj_pca_model in self.subj_pca_models.items():
            # Combine and deduplicate markers to be deleted
            num_of_markers = subj_pca_model.raw_data.shape[1] // 3

            markers_to_del_temp = list(set(
                _string_to_index(subj_pca_model.subject_UI_settings['markers_to_del'].get(), num_of_markers) + 
                _string_to_index(self.configuration.get_del_markers,  num_of_markers)
            ))

            # Store updated marker deletion set and skeleton in subject metadata
            subj_pca_model.markers_to_del_set = markers_to_del_temp
            subj_pca_model.func_order = 4 if self.configuration.normalisation.get() == 'Mean Dist. 2 Markers (Centre Ref.)' else 0
            subj_pca_model.weights = np.delete(self.g_weight, list(subj_pca_model.markers_to_del_set)) #self.view._get_table_values('weights')
            subj_pca_model.centre_refs = np.delete(self.g_centre_ref, list(subj_pca_model.markers_to_del_set))
            temp_skel = np.array([[int(l[0]), int(l[1] or l[0])] for l in list(enumerate(self.g_skeleton))])
            temp_skel = _dissolve_net(temp_skel, subj_pca_model.markers_to_del_set)
            subj_pca_model.line_colours = self.g_colour[temp_skel[:, 0]]
            subj_pca_model.skeleton = _unique_transform(temp_skel)
        
            subj_pca_model.sample_freq = self.configuration.get_freq

            num_rows = subj_pca_model.raw_data.shape[0]
            rows_to_del_temp = list(set(
                _string_to_index(subj_pca_model.subject_UI_settings['rows_to_del'].get(), num_rows) +
                _string_to_index(self.configuration.get_del_rows, num_rows)
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
        self.view.fig_ev.savefig(f'{self.output_folder_path}/EV_Plots.{self.configuration.plot_save_extension.get()}', dpi=300)
        for i, score_plot_tab_name in enumerate(self.view.score_plot_tab_names):
            self.view.fig_score_plots[i].savefig(f'{self.output_folder_path}/Score_Plot_{score_plot_tab_name}.{self.configuration.plot_save_extension.get()}', dpi=300)
    

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


def string_to_list(index_string, no_str_result = []):
    index_string = index_string.replace('[', '').replace(']', '').replace(' ', '')
    if not index_string:
        return no_str_result
    
    result_list = []
    for item in index_string.split(','):
        if item:
            if isfloat(item):
                result_list.append(float(item))
            elif item:
                raise TerminatingError(f"Invalid literal for float: '{item}'")
    
    if len(result_list) == 0:
        return no_str_result
    return result_list


def _validate_index_string(index_string):
    index_string = index_string.replace('[', '').replace(']', '').replace(' ', '')
    pattern = r'^(:?\d*(?::\d*){0,2})(?:,(?:\d*(?::\d*){0,2}))*$' # Regex to match valid index or slice patterns
    
    # Check if the index_string matches the pattern
    if not re.match(pattern, index_string):
        return False
    
    # Check each part of the index_string
    for item in index_string.split(','):
        parts = item.split(':')
        
        if len(parts) > 3:
            return False

    return True


def _string_to_index(index_string, list_length):
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



# Removes a skeleton line segment 
def _dissolve_net(segments, points_to_remove):
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


def _unique_transform(arr):
    '''
    Transforms a numpy array such that all unique values are mapped to a contiguous range starting from 0.
    This allows removal of items from the skeleton array while maintining the correct indexing for the marker columns.
    '''
    flat_arr = arr.flatten()
    unique_values = np.unique(flat_arr)  # Get unique values
    value_mapping = {val: idx for idx, val in enumerate(unique_values)}  # Create mapping from value to index
    transformed_flat_arr = np.vectorize(value_mapping.get)(flat_arr)  # Apply mapping to flattened array
    transformed_arr = transformed_flat_arr.reshape(arr.shape)  # Reshape back to original shape
    return transformed_arr


def print_loading_bar(iteration, total, task_name, length=30):
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r|{bar}| {percent:.1f}% Complete - {task_name:<25}', end='\r')
    if iteration == total:
        print()


if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=5)

    controller = Controller()
    controller.view.mainloop()
