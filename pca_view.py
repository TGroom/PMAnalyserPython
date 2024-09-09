"""
This module defines the FrontendApp class for handling the graphical user interface (GUI) operations
related to the PCA Analyser Tool.

Usage:
    Instantiate the FrontendApp class to create the GUI interface for interacting with the PCA Analyser Tool.

Attributes:
    No public attributes.
"""
import os
import tkinter
import tkinter.messagebox

import numpy as np
import pandas as pd
from scipy import signal
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
import customtkinter
from matplotlib.widgets import Slider
from matplotlib.colors import is_color_like
import sys


class PCA_View(customtkinter.CTk):
    def __init__(self, controller):
        super().__init__()

        self.corner_radius = 0
        if 'win32' in sys.platform:
            self.corner_radius = 5

        # Set appearance mode and color theme
        customtkinter.set_appearance_mode('Dark')  # Modes: 'System' (standard), 'Dark', 'Light'
        customtkinter.set_default_color_theme('blue')  # Themes: 'blue' (standard), 'green', 'dark-blue'

        # Initialise variables
        self.controller = controller
        
        self.quit_dialog = None
        self.canvas_score_plots = [[], [], []]
        self.fig_score_plots = [[], [], []]
        self.score_plot_axes = [[], [], []]
        self.toolbar_score_plots = [[], [], []]
        self.pca_busy = False
        self.prev_subject_id = None
        self.bug_plot_anim = None
        self.loadings_anim = None
        self.time_series_anim = None
        self.all_tooltips = []

        theme_color = ('#bfbfbf', '#1F6AA5')
        theme_color_hover = ('#b3b3b3', '#144870')
        text_color = ('gray10', '#DCE4EE')
        text_color_disabled = ('gray74', 'gray60')

        self.frame_settings = {
            'fg_color': ('gray99', 'gray17'),
            'corner_radius': 0
        }
        self.sidebar_button_settings = {
            'text_color': text_color,
            'fg_color': ('gray90', '#343638'),
            'hover_color': ('gray95', '#7A848D'),
            'text_color_disabled': text_color_disabled,
            'corner_radius': self.corner_radius
        }
        self.frame_settings_level_2_tab = {
            'fg_color': ('gray96', 'gray21'),
            'text_color': text_color,
            'segmented_button_fg_color': ('gray88', 'gray29'),
            'segmented_button_selected_color': theme_color,
            'segmented_button_selected_hover_color': theme_color_hover,
            'segmented_button_unselected_color': ('gray88', 'gray29'),
            'segmented_button_unselected_hover_color': ('gray95', 'gray41'),
            'text_color_disabled': text_color_disabled,
            'corner_radius': self.corner_radius
        }
        self.frame_settings_level_2_scroll = {
            'fg_color': ('gray96', 'gray21'),
            'label_fg_color': ('gray92', 'gray25'),
            'corner_radius': self.corner_radius
        }
        self.button_settings = {
            'fg_color': theme_color,
            'hover_color': theme_color_hover,
            'text_color': text_color,
            'text_color_disabled': text_color_disabled,
            'corner_radius': self.corner_radius
        }
        self.progress_bar_settings = {
            'border_width': 5,
            'orientation': 'horizontal',
            'mode': 'determinate',
            'indeterminate_speed': 1,
            'border_color': ('#939BA2', '#4A4D50'),
            'progress_color': theme_color,
            'fg_color': ('gray96', 'gray21'),
            'corner_radius': self.corner_radius
        }
        self.option_menu_settings = {
            'text_color': text_color,
            'fg_color': ('#e4e6e7', '#404244'),
            'button_color': ('#d7d9da', '#565B5E'),
            'button_hover_color': self.sidebar_button_settings['hover_color'],
            'text_color_disabled': text_color_disabled,
            'corner_radius': self.corner_radius
        }
        self.entry_settings = {
            'text_color': text_color,
            'text_color_disabled': text_color_disabled,
            'corner_radius': self.corner_radius
        }
        self.checkbox_settings = {
            'fg_color': theme_color,
            'border_width': 2,
            'border_color': ('#979DA2', '#565B5E'),
            'hover_color': theme_color_hover,
            'corner_radius': self.corner_radius
        }
        self.checkbox_fg_color_disabled = ('#d9d9d9', '#44657e')
        
        self.grid_settings = {'padx': 10, 'pady': 5}
        self.frame_grid_settings = {'padx': 8, 'pady': 8}
        

    def setup_ui(self):
        # Configure window
        self.title('PMAnalyserPython')
        self.geometry('1350x750')
        self.focus_force()
        self.protocol('WM_DELETE_WINDOW', self._on_closing)

        # Configure grid layout
        self.grid_columnconfigure(1, weight=1, minsize=100)
        self.grid_rowconfigure(0, weight=1, minsize=100)

        # Create sidebar frame with widgets
        sidebar_frame = customtkinter.CTkFrame(self, width=10, **self.frame_settings)
        sidebar_frame.grid(row=0, column=0, sticky='nsw')
        sidebar_frame.grid_rowconfigure(8, weight=1)

        # Add logo label to sidebar
        logo_label = customtkinter.CTkLabel(sidebar_frame, text='PMAnalyserPython', font=customtkinter.CTkFont(size=20, weight='bold'))
        logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # PCA Analysis frame
        self.pca_analysis_frame = customtkinter.CTkFrame(self, **self.frame_settings)
        self.pca_analysis_frame.grid_columnconfigure((2, 3, 4), weight=1, minsize=1)
        self.pca_analysis_frame.grid_columnconfigure(0, weight=0, minsize=150)
        self.pca_analysis_frame.grid_columnconfigure(1, weight=0, minsize=300)
        self.pca_analysis_frame.grid_columnconfigure(5, weight=1, minsize=400)
        self.pca_analysis_frame.grid_rowconfigure(0, weight=0, minsize=30)
        self.pca_analysis_frame.grid_rowconfigure((1, 2, 3, 4, 5, 6, 7), weight=2, minsize=1)
        self.pca_analysis_frame.grid_rowconfigure((7, 8), weight=1, minsize=20)

        # Visualisation Plots frame
        self.visualisation_plots_frame = customtkinter.CTkFrame(self, **self.frame_settings)
        self.visualisation_plots_frame.grid_columnconfigure((2, 3, 4, 5), weight=1, minsize=1)
        self.visualisation_plots_frame.grid_columnconfigure(0, weight=0, minsize=150)
        self.visualisation_plots_frame.grid_columnconfigure(1, weight=0, minsize=300)
        self.visualisation_plots_frame.grid_columnconfigure(6, weight=1, minsize=400)
        self.visualisation_plots_frame.grid_rowconfigure(0, weight=0, minsize=30)
        self.visualisation_plots_frame.grid_rowconfigure(1, weight=1, minsize=100)

        # Animation frame
        self.videos_frame = customtkinter.CTkFrame(self, **self.frame_settings)
        self.videos_frame.grid_columnconfigure((2, 3, 4, 5), weight=1, minsize=1)
        self.videos_frame.grid_columnconfigure(0, weight=0, minsize=150)
        self.videos_frame.grid_columnconfigure(1, weight=0, minsize=300)
        self.videos_frame.grid_columnconfigure(6, weight=1, minsize=400)
        self.videos_frame.grid_rowconfigure(0, weight=0, minsize=30)
        self.videos_frame.grid_rowconfigure(1, weight=1, minsize=100)
        
        # Animation frame
        self.eigenwalkers_frame = customtkinter.CTkFrame(self, **self.frame_settings)
        self.eigenwalkers_frame.grid_columnconfigure((2, 3, 4, 5), weight=1, minsize=1)
        self.eigenwalkers_frame.grid_columnconfigure(0, weight=0, minsize=150)
        self.eigenwalkers_frame.grid_columnconfigure(1, weight=0, minsize=300)
        self.eigenwalkers_frame.grid_columnconfigure(6, weight=1, minsize=400)
        self.eigenwalkers_frame.grid_rowconfigure(0, weight=0, minsize=30)
        self.eigenwalkers_frame.grid_rowconfigure(1, weight=1, minsize=100)

        # Settings frame
        self.settings_frame = customtkinter.CTkFrame(self, **self.frame_settings)
        self.settings_frame.grid_columnconfigure((1, 8), weight=1, minsize=100)
        self.settings_frame.grid_rowconfigure((1, 20), weight=1, minsize=100)

        # List of frame widgets to manage dynamically
        self.sidebar_frames_list = [
            self.pca_analysis_frame,
            self.visualisation_plots_frame,
            self.videos_frame,
            self.eigenwalkers_frame,
            self.settings_frame
        ]

        # Subject selection settings for different frames
        subject_selection_label = customtkinter.CTkLabel(self.pca_analysis_frame, text='Subject Selected')
        subject_selection_label.grid(row=0, column=0, sticky='e', padx=(10, 10), pady=(10, 0))
        self.subject_selection_option_menu_1 = customtkinter.CTkOptionMenu(self.pca_analysis_frame, values=['None'], command=lambda x: (self._update_subject_selection(x), self._show_bug_plot()), variable=self.controller.current_subject_id, **self.option_menu_settings)
        self.subject_selection_option_menu_1.grid(row=0, column=1, sticky='we', padx=(10, 10), pady=(10, 0))

        subject_selection_label = customtkinter.CTkLabel(self.visualisation_plots_frame, text='Subject Selected')
        subject_selection_label.grid(row=0, column=0, sticky='e', padx=(10, 10), pady=(10, 0))
        self.subject_selection_option_menu_2 = customtkinter.CTkOptionMenu(self.visualisation_plots_frame, values=['None'], command=lambda x: (self._update_subject_selection(x), self._show_plots()), variable=self.controller.current_subject_id, **self.option_menu_settings)
        self.subject_selection_option_menu_2.grid(row=0, column=1, sticky='we', padx=(10, 10), pady=(10, 0))

        subject_selection_label = customtkinter.CTkLabel(self.videos_frame, text='Subject Selected')
        subject_selection_label.grid(row=0, column=0, sticky='e', padx=(10, 10), pady=(10, 0))
        self.subject_selection_option_menu_3 = customtkinter.CTkOptionMenu(self.videos_frame, values=['None'], command=lambda x: (self._update_subject_selection(x), self._video_plot_tab_selection()), variable=self.controller.current_subject_id, **self.option_menu_settings)
        self.subject_selection_option_menu_3.grid(row=0, column=1, sticky='we', padx=(10, 10), pady=(10, 0))

        eigenwalker_group_selection_label = customtkinter.CTkLabel(self.eigenwalkers_frame, text='Group Selected')
        eigenwalker_group_selection_label.grid(row=0, column=0, sticky='e', padx=(10, 10), pady=(10, 0))
        self.eigenwalker_group_option_menu = customtkinter.CTkOptionMenu(self.eigenwalkers_frame, values=['None'], variable=self.controller.current_eigenwalker_group_id, **self.option_menu_settings)
        self.eigenwalker_group_option_menu.configure(command=lambda x: (self._show_eigenwalker_space()))
        self.eigenwalker_group_option_menu.grid(row=0, column=1, sticky='we', padx=(10, 10), pady=(10, 0))


        # Sidebar Buttons
        self.sidebar_buttons = [
            customtkinter.CTkButton(sidebar_frame, text='Load Project File', command=self.controller.load_project_file, **self.sidebar_button_settings),
            customtkinter.CTkButton(sidebar_frame, text='Save Project File', command=self.controller.save_project_file, **self.sidebar_button_settings),
            customtkinter.CTkButton(sidebar_frame, text='PCA Analysis', command=lambda: (self._show_pane(self.pca_analysis_frame), self._show_bug_plot()), **self.sidebar_button_settings),
            customtkinter.CTkButton(sidebar_frame, text='Visualisation Plots', command=lambda: (self._show_pane(self.visualisation_plots_frame), self._show_plots()), **self.sidebar_button_settings),
            customtkinter.CTkButton(sidebar_frame, text='Videos', command=lambda: (self._show_pane(self.videos_frame), self._video_plot_tab_selection()), **self.sidebar_button_settings),
            customtkinter.CTkButton(sidebar_frame, text='Eigenwalkers', command=lambda: (self._show_pane(self.eigenwalkers_frame), self._show_eigenwalker_space()), **self.sidebar_button_settings),
            customtkinter.CTkButton(sidebar_frame, text='Settings', command=lambda: (self._show_pane(self.settings_frame), self._destroy_all_canvas()), **self.sidebar_button_settings)
        ]
        
        for i, btn in enumerate(self.sidebar_buttons, start=1):
            btn.grid(row=i, column=0, padx=20, pady=10)

        # Set default view
        self._show_pane(self.pca_analysis_frame)

        # Settings Frame
        appearance_mode_label = customtkinter.CTkLabel(self.settings_frame, text='Theme Colour')
        appearance_mode_label.grid(row=1, column=3, sticky='se', **self.grid_settings)
        self.appearance_mode_option_menu = customtkinter.CTkOptionMenu(
            self.settings_frame, dynamic_resizing=False,
            values=['System', 'Light', 'Dark'],
            command=customtkinter.set_appearance_mode,
            variable=self.controller.appearance_mode_option_menu,
             **self.option_menu_settings
        )
        self.appearance_mode_option_menu.grid(row=1, column=4, sticky='swe', **self.grid_settings)
        self.appearance_mode_option_menu.set('Dark')

        scaling_label = customtkinter.CTkLabel(self.settings_frame, text='UI Scaling')
        scaling_label.grid(row=2, column=3, sticky='ne', **self.grid_settings)
        self.scaling_option_menu = customtkinter.CTkOptionMenu(
            self.settings_frame, dynamic_resizing=False,
            values=['50%', '75%', '90%', '100%', '110%', '125%', '150%', '175%', '200%', '225%', '250%'],
            command=self.update_scaling,
            variable=self.controller.scaling_option_menu,
             **self.option_menu_settings
        )
        self.scaling_option_menu.grid(row=2, column=4, sticky='nwe', **self.grid_settings)
        self.scaling_option_menu.set('100%')

        tooltips_label = customtkinter.CTkLabel(self.settings_frame, text='Show Tooltips')
        tooltips_label.grid(row=3, column=3, sticky='ne', **self.grid_settings)
        self.tooltips_checkbox = customtkinter.CTkCheckBox(self.settings_frame, text='', command=self._toggle_tooltips, variable=self.controller.tooltips_checkbox, **self.checkbox_settings)
        self.tooltips_checkbox.grid(row=3, column=4, sticky='swe', **self.grid_settings)

        # Input Data Frame
        self.input_data_frame = customtkinter.CTkScrollableFrame(self.pca_analysis_frame, label_text='Computation Settings', **self.frame_settings_level_2_scroll)
        self.input_data_frame.grid(row=0, rowspan=8, column=5, sticky='nsew', **self.frame_grid_settings)
        self.input_data_frame.columnconfigure(0, weight=2)

        self.split_frame_input_data = customtkinter.CTkFrame(self.input_data_frame, fg_color='transparent')
        self.split_frame_input_data.grid(row=0, column=0, sticky='nsew')
        self.split_frame_input_data.columnconfigure((0, 1), weight=2)

        self.file_path_button = customtkinter.CTkButton(self.split_frame_input_data, text='Open Files', command=self.controller.open_data_files, **self.button_settings)
        self.file_path_button.grid(row=0, column=0, columnspan=2, sticky='nsew', **self.grid_settings)
        self.file_path_label = customtkinter.CTkTextbox(self.split_frame_input_data, height=100, state='disabled', wrap='word', corner_radius=self.corner_radius)
        self.file_path_label.grid(row=1, column=0, columnspan=2, sticky='nsew', **self.grid_settings)

        # Data Information & Treatment
        self.pca_mode_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['All Subjects Together', 'Every Data Set Seperately'], variable=self.controller.pca_mode_option_menu, **self.option_menu_settings)
        self.delimiter_entry = customtkinter.CTkEntry(self.split_frame_input_data, justify='center', textvariable=self.controller.delimiter_entry, corner_radius=self.corner_radius)
        self.freq_entry = customtkinter.CTkEntry(self.split_frame_input_data, justify='center', textvariable=self.controller.freq_entry, corner_radius=self.corner_radius)
        self.del_rows_entry = customtkinter.CTkEntry(self.split_frame_input_data, justify='center', textvariable=self.controller.del_rows_entry, corner_radius=self.corner_radius)
        self.del_markers_entry = customtkinter.CTkEntry(self.split_frame_input_data, justify='center', textvariable=self.controller.del_markers_entry, corner_radius=self.corner_radius)
        self.gap_filling_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['Off', 'Forward Fill', 'Backward Fill', 'Linear', 'Quadratic', 'Cubic', '1st Order Spline', '2nd Order Spline'], variable=self.controller.gap_filling_option_menu, **self.option_menu_settings)
        self.data_filtering_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['Off', 'Butterworth'], **self.option_menu_settings)
        self.centring_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['Off', 'Mean Marker Position', 'Mean Marker Pos. (Weights)', 'Mean Marker Pos. (Centre Ref.)'], variable=self.controller.centring_option_menu, **self.option_menu_settings)
        self.align_orientation_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['Off', 'Body-Fixed (Centre Ref.)', 'Soft Align Z (X-Axis)', 'Soft Align Z (Y-Axis)'], variable=self.controller.align_orientation_option_menu, **self.option_menu_settings)
        self.orientation_cutoff_freq_entry = customtkinter.CTkEntry(self.split_frame_input_data, justify='center', textvariable=self.controller.orientation_cutoff_freq_entry, corner_radius=self.corner_radius)
        self.normalisation_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['Off', 'MED (Mean Euclidean Distance)', 'Mean Dist. 2 Markers (Centre Ref.)', 'Maximum Range (1st Coords)', 'Maximum Range (2nd Coords)', 'Maximum Range (3rd Coords)'], variable=self.controller.normalisation_option_menu, **self.option_menu_settings)
        self.weights_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['Off', 'Manual Weight Vector'], variable=self.controller.weights_option_menu, **self.option_menu_settings)
        self.coordinate_transformation_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['Off', 'Cartesian -> Spherical (3D)'], variable=self.controller.coordinate_transformation_option_menu, **self.option_menu_settings)

        # Output/Save/Load
        self.pp_filter_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['No Filter', 'Low Pass Butterworth'], variable=self.controller.pp_filter_option_menu, **self.option_menu_settings)
        self.pv_filter_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['No Filter', 'Low Pass Butterworth'], variable=self.controller.pv_filter_option_menu, **self.option_menu_settings)
        self.pa_filter_option_menu = customtkinter.CTkOptionMenu(self.split_frame_input_data, values=['No Filter', 'Low Pass Butterworth'], variable=self.controller.pa_filter_option_menu, **self.option_menu_settings)
        self.pm_filter_order_entry = customtkinter.CTkEntry(self.split_frame_input_data, justify='center', textvariable=self.controller.pm_filter_order_entry, corner_radius=self.corner_radius)
        self.pm_filter_cut_off_entry = customtkinter.CTkEntry(self.split_frame_input_data, justify='center', textvariable=self.controller.pm_filter_cut_off_entry, corner_radius=self.corner_radius)
        self.loocv_checkbox = customtkinter.CTkCheckBox(self.split_frame_input_data, text='', **self.checkbox_settings, variable=self.controller.loocv_checkbox)
        self.loocv_checkbox.select()
        self.freq_harmonics_entry = customtkinter.CTkEntry(self.split_frame_input_data, justify='center', textvariable=self.controller.freq_harmonics_entry, corner_radius=self.corner_radius)

        # Dictionary to store label texts and corresponding widgets
        components = {
            2: ('PCA Mode', self.pca_mode_option_menu),
            3: ('Delimiter', self.delimiter_entry),
            4: ('Sample Freq. (Hz)', self.freq_entry),
            5: ('Delete Rows', self.del_rows_entry),
            6: ('Delete Markers', self.del_markers_entry),
            7: ('Gap Filling', self.gap_filling_option_menu),
            8: ('Data Filtering', self.data_filtering_option_menu),
            9: ('Centring', self.centring_option_menu),
            10: ('Align Orientation', self.align_orientation_option_menu),
            11: ('Orientation Freq. Cutoff', self.orientation_cutoff_freq_entry),
            12: ('Normalisation', self.normalisation_option_menu),
            13: ('Weights', self.weights_option_menu),
            14: ('Coordinate Transform', self.coordinate_transformation_option_menu),
            15: ('Position PCA Filter', self.pp_filter_option_menu),
            16: ('Velocity PCA Filter', self.pv_filter_option_menu),
            17: ('Acceleration PCA Filter', self.pa_filter_option_menu),
            18: ('Order', self.pm_filter_order_entry),
            19: ('Cutoff Freq.', self.pm_filter_cut_off_entry),
            20: ('Cross-Validation', self.loocv_checkbox),
            21: ('Sine Approx. Freq. Ratios', self.freq_harmonics_entry)
        }

        # Create labels and widgets dynamically
        for row, (label_text, widget) in components.items():
            label = customtkinter.CTkLabel(self.split_frame_input_data, text=label_text)
            label.grid(row=row, column=0, sticky='e', **self.grid_settings)
            widget.grid(row=row, column=1, sticky='we', **self.grid_settings)

        ## EMBEDDED GRAPHS
        self.pca_analysis_graphs_frame = customtkinter.CTkTabview(master=self.pca_analysis_frame, **self.frame_settings_level_2_tab)
        self.pca_analysis_graphs_frame.grid(row=7, rowspan=2, column=0, columnspan=5, sticky='nsew', **self.frame_grid_settings)
        self.pca_analysis_graphs_frame.add('Powerspektrum Plot')
        self.pca_analysis_graphs_frame.add('Cross-Validation Plot')
        self.pca_analysis_graphs_frame.add('Orientation Plot')
        self.pca_analysis_graphs_frame.set('Powerspektrum Plot')

        self.fig_welch = plt.Figure(figsize=(15, 2.5), dpi=75)
        self.plot_welch = self.fig_welch.add_subplot(111)
        self.fig_welch.subplots_adjust(bottom=0.2)
        self.canvas_welch = FigureCanvasTkAgg(self.fig_welch, master=self.pca_analysis_graphs_frame.tab('Powerspektrum Plot'))
        self.canvas_welch.get_tk_widget().pack(expand=1, fill=tkinter.BOTH)
        self.canvas_welch.draw()

        self.fig_loocv = plt.Figure(figsize=(15, 2.5), dpi=75)
        self.plot_loocv = self.fig_loocv.add_subplot(111)
        self.fig_loocv.subplots_adjust(bottom=0.2)
        self.plot_loocv_2 = self.plot_loocv.twinx()
        self.canvas_loocv = FigureCanvasTkAgg(self.fig_loocv, master=self.pca_analysis_graphs_frame.tab('Cross-Validation Plot'))
        self.canvas_loocv.get_tk_widget().pack(expand=1, fill=tkinter.BOTH)
        self.canvas_loocv.draw()

        self.fig_orientation = plt.Figure(figsize=(15, 2.5), dpi=75)
        self.plot_orientation_series = self.fig_orientation.add_subplot(121)
        self.plot_orientation_freq = self.fig_orientation.add_subplot(122)
        self.fig_orientation.subplots_adjust(bottom=0.2)
        self.canvas_orientation = FigureCanvasTkAgg(self.fig_orientation, master=self.pca_analysis_graphs_frame.tab('Orientation Plot'))
        self.canvas_orientation.get_tk_widget().pack(expand=1, fill=tkinter.BOTH)
        self.canvas_orientation.draw()

        self.run_pca_button = customtkinter.CTkButton(self.pca_analysis_frame, text='Run PCA', command=self.controller.run_analysis, **self.button_settings)
        self.run_pca_button.grid(row=8, column=5, sticky='nsew', **self.frame_grid_settings)

        self.progress_bar = customtkinter.CTkProgressBar(self.pca_analysis_frame, height=self.run_pca_button.cget('height'), **self.progress_bar_settings)
        self.progress_bar.grid(row=8, column=5, sticky='nsew', **self.frame_grid_settings)
        self.progress_bar.grid_remove()

        # 3D Graph and Controls
        self.plot_3d_frame = customtkinter.CTkScrollableFrame(self.pca_analysis_frame, label_text='Input Data & Weights', **self.frame_settings_level_2_scroll)
        self.plot_3d_frame.grid(row=1, rowspan=6, column=0, columnspan=5, sticky='nsew', **self.frame_grid_settings)
        self.plot_3d_frame.columnconfigure((0, 1), weight=1, minsize=300)
        self.plot_3d_subframe = customtkinter.CTkFrame(self.plot_3d_frame, fg_color='transparent')
        self.plot_3d_subframe.grid(row=0, column=0, columnspan=2, sticky='nsew', **self.frame_grid_settings)

        self.style = tkinter.ttk.Style()
        self.style.layout('Edge.Treeview', [('Edge.Treeview.treearea',{})])
        self.style.configure('Edge.Treeview', highlightthickness=0, bd=0)

        self.table = Tableview(self.plot_3d_subframe, columns=('#0', '#1', '#2', '#3'), style='Edge.Treeview')
        self.table.heading('#0', text='Index   Markers', anchor=tkinter.W)
        self.table.column('#0', width=200, minwidth=200, anchor=tkinter.W)
        self.table.heading('#1', text='Weights', anchor=tkinter.CENTER)
        self.table.column('#1', width=75, minwidth=75, anchor=tkinter.CENTER)
        self.table.heading('#2', text='Centre Ref.', anchor=tkinter.CENTER)
        self.table.column('#2', width=75, minwidth=75, anchor=tkinter.CENTER)
        self.table.heading('#3', text='Skeleton', anchor=tkinter.CENTER)
        self.table.column('#3', width=75, minwidth=75, anchor=tkinter.CENTER)
        self.table.heading('#4', text='Colour', anchor=tkinter.CENTER)
        self.table.column('#4', width=100, minwidth=100, anchor=tkinter.CENTER)
        self.table.pack(side=tkinter.RIGHT, fill=tkinter.BOTH)#(row=0, column=1, sticky='nsew', **self.frame_grid_settings)

        self.figAnim3D = plt.Figure(figsize=(8, 8), dpi=70)
        self.BugPlot = self.figAnim3D.add_subplot(111, projection='3d')
        self._configure_axes(self.BugPlot, 0.15)
        self.canvas = FigureCanvasTkAgg(self.figAnim3D, self.plot_3d_subframe)
        self.canvas.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)#(row=0, column=0, sticky='ns', **self.grid_settings)
        self.canvas.draw()

        subject_del_rows_label = customtkinter.CTkLabel(self.plot_3d_frame, text='Delete Subject Specific Rows')
        subject_del_rows_label.grid(row=3, column=0, sticky='e', **self.grid_settings)
        self.subject_del_rows_entry = customtkinter.CTkEntry(self.plot_3d_frame, justify='center', corner_radius=self.corner_radius)
        self.subject_del_rows_entry.grid(row=3, column=1, sticky='we', **self.grid_settings)

        subject_del_markers_label = customtkinter.CTkLabel(self.plot_3d_frame, text='Delete Subject Specific Markers')
        subject_del_markers_label.grid(row=4, column=0, sticky='e', **self.grid_settings)
        self.subject_del_markers_entry = customtkinter.CTkEntry(self.plot_3d_frame, justify='center', corner_radius=self.corner_radius)
        self.subject_del_markers_entry.grid(row=4, column=1, sticky='we', **self.grid_settings)

        self.flip_axes_frame = customtkinter.CTkFrame(self.plot_3d_frame, fg_color='transparent')
        self.flip_axes_frame.grid(row=6, column=1, sticky='nw')
        self.flip_axes_frame.columnconfigure((0, 2, 4), weight=1, minsize=50)
        self.flip_axes_frame.columnconfigure((1, 3, 5), weight=1, minsize=0)

        subject_flip_x_label = customtkinter.CTkLabel(self.flip_axes_frame, text='Flip X')
        subject_flip_x_label.grid(row=0, column=0, sticky='e', **self.grid_settings)
        self.subject_flip_x_checkbox = customtkinter.CTkCheckBox(self.flip_axes_frame, text='', **self.checkbox_settings)
        self.subject_flip_x_checkbox.grid(row=0, column=1, sticky='nw', **self.grid_settings)

        subject_flip_y_label = customtkinter.CTkLabel(self.flip_axes_frame, text='Flip Y')
        subject_flip_y_label.grid(row=0, column=2, sticky='e', **self.grid_settings)
        self.subject_flip_y_checkbox = customtkinter.CTkCheckBox(self.flip_axes_frame, text='', **self.checkbox_settings)
        self.subject_flip_y_checkbox.grid(row=0, column=3, sticky='nw', **self.grid_settings)

        subject_flip_z_label = customtkinter.CTkLabel(self.flip_axes_frame, text='Flip Z')
        subject_flip_z_label.grid(row=0, column=4, sticky='e', **self.grid_settings)
        self.subject_flip_z_checkbox = customtkinter.CTkCheckBox(self.flip_axes_frame, text='', **self.checkbox_settings)
        self.subject_flip_z_checkbox.grid(row=0, column=5, sticky='nw', **self.grid_settings)

        subject_eigenwalker_group_label = customtkinter.CTkLabel(self.plot_3d_frame, text='Eigenwalker PCA Group')
        subject_eigenwalker_group_label.grid(row=7, column=0, sticky='e', **self.grid_settings)
        self.subject_eigenwalker_group_entry = customtkinter.CTkEntry(self.plot_3d_frame, justify='center', corner_radius=self.corner_radius)
        self.subject_eigenwalker_group_entry.grid(row=7, column=1, sticky='we', **self.grid_settings)

        # Initialise plots and videos
        self._initialise_plots()
        self._initialise_videos()
        self._initialise_tooltips()
        self._toggle_tooltips()


    def _initialise_plots(self):
        '''Initialise plot-related UI elements.'''
        # Score plots tab
        self.score_plot_tab_names = ['Position', 'Velocity', 'Acceleration']
        self.score_plots_tab = customtkinter.CTkTabview(master=self.visualisation_plots_frame, **self.frame_settings_level_2_tab)#, command=self._show_plots)
        self.score_plots_tab.grid(row=1, rowspan=1, column=0, columnspan=6, sticky='nsew', **self.frame_grid_settings)
        for tab in self.score_plot_tab_names:
            self.score_plots_tab.add(tab)
        self.score_plots_tab.set('Position')

        self.pp_score_plots_frame = customtkinter.CTkScrollableFrame(self.score_plots_tab.tab('Position'), fg_color='transparent')
        self.pp_score_plots_frame.pack(expand=1, fill=tkinter.BOTH)
        self.pv_score_plots_frame = customtkinter.CTkScrollableFrame(self.score_plots_tab.tab('Velocity'), fg_color='transparent')
        self.pv_score_plots_frame.pack(expand=1, fill=tkinter.BOTH)
        self.pa_score_plots_frame = customtkinter.CTkScrollableFrame(self.score_plots_tab.tab('Acceleration'), fg_color='transparent')
        self.pa_score_plots_frame.pack(expand=1, fill=tkinter.BOTH)

        # Initialise Score Plots
        for i, frame in enumerate([self.pp_score_plots_frame, self.pv_score_plots_frame, self.pa_score_plots_frame]):
            self.fig_score_plots[i] = plt.Figure(figsize=(15, 15), dpi=60)
            self.canvas_score_plots[i] = FigureCanvasTkAgg(self.fig_score_plots[i], master=frame)
            self.toolbar_score_plots[i] = NavigationToolbar2Tk(self.canvas_score_plots[i], frame)
            self.toolbar_score_plots[i].update()
            self.toolbar_score_plots[i].pack(side=tkinter.TOP, fill=tkinter.X)
            self.canvas_score_plots[i].get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.X, expand=1)

        # General plots frame
        self.plots_frame = customtkinter.CTkScrollableFrame(self.visualisation_plots_frame, label_text='Plots', **self.frame_settings_level_2_scroll)
        self.plots_frame.grid(row=0, rowspan=2, column=6, sticky='nsew', **self.frame_grid_settings)
        self.plots_frame.columnconfigure(0, weight=2)

        self.split_frame_plots = customtkinter.CTkFrame(self.plots_frame, fg_color='transparent')
        self.split_frame_plots.grid(row=0, column=0, sticky='nsew')
        self.split_frame_plots.columnconfigure((0, 1), weight=2)

        self.pca_file_path_textbox = customtkinter.CTkTextbox(self.split_frame_plots, height=100, state='disabled', wrap='word')
        self.pca_file_path_textbox.grid(row=1, column=0, columnspan=2, sticky='nsew', **self.grid_settings)

        self.plot_save_extension_label = customtkinter.CTkLabel(self.split_frame_plots, text='Save Plots As')
        self.plot_save_extension_label.grid(row=3, column=0, sticky='e', **self.grid_settings)
        self.plot_save_extension_option_menu = customtkinter.CTkSegmentedButton(self.split_frame_plots, values=['png', 'pdf', 'svg'], variable=self.controller.plot_save_extension_option_menu) #customtkinter.StringVar(value='png')
        self.plot_save_extension_option_menu.grid(row=3, column=1, sticky='nsew', **self.grid_settings)

        ### Plots settings
        self.ev_num_of_pcs_label = customtkinter.CTkLabel(self.split_frame_plots, text='Number of PCs to plot')
        self.ev_num_of_pcs_label.grid(row=4, column=0, sticky='e', **self.grid_settings)
        self.ev_num_of_pcs_entry = customtkinter.CTkEntry(self.split_frame_plots, justify='center', textvariable=self.controller.ev_num_of_pcs_entry, corner_radius=self.corner_radius)
        self.ev_num_of_pcs_entry.grid(row=4, column=1, sticky='w', **self.grid_settings)

        sine_approx_label = customtkinter.CTkLabel(self.split_frame_plots, text='Show Sine Approx.')
        sine_approx_label.grid(row=5, column=0, sticky='e', **self.grid_settings)
        self.sine_approx_checkbox = customtkinter.CTkCheckBox(self.split_frame_plots, text='', command=self._show_plots, variable=self.controller.sine_approx_checkbox, **self.checkbox_settings)
        self.sine_approx_checkbox.grid(row=5, column=1, sticky='nw', **self.grid_settings)
        self.sine_approx_checkbox.select()

        self.plots_refresh_button = customtkinter.CTkButton(self.split_frame_plots, text='Refresh Plots', command=self._show_plots, **self.button_settings)
        self.plots_refresh_button.grid(row=6, column=0, columnspan=1, sticky='nesw', **self.grid_settings)
        self.plots_save_button = customtkinter.CTkButton(self.split_frame_plots, text='Save Plots', command=self.controller.save_plots, **self.button_settings)
        self.plots_save_button.grid(row=6, column=1, columnspan=1, sticky='nesw', **self.grid_settings)

        # Initialise plot canvases for EV plots
        self.fig_ev = plt.Figure(figsize = (5, 12), dpi = 75)
        self.plot_ev_1 = self.fig_ev.add_subplot(411)
        self.plot_ev_2 = self.fig_ev.add_subplot(412)
        self.plot_ev_3 = self.fig_ev.add_subplot(413)
        self.plot_ev_4 = self.fig_ev.add_subplot(414)

        figure_frame = customtkinter.CTkFrame(self.plots_frame)
        figure_frame.grid(row=3, column=0, sticky='nsew')
        figure_frame.columnconfigure(0, weight=2)

        self.canvas_ev = FigureCanvasTkAgg(self.fig_ev, master=figure_frame)
        toolbar_ev = NavigationToolbar2Tk(self.canvas_ev, figure_frame)
        toolbar_ev.update()
        toolbar_ev.pack(side=tkinter.TOP, fill=tkinter.X)
        self.canvas_ev.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)
        

    def _initialise_videos(self):
        '''Initialise video-related UI elements.'''
        self.video_visualisation = customtkinter.CTkTabview(master=self.videos_frame, command=self._video_plot_tab_selection, **self.frame_settings_level_2_tab)
        self.video_visualisation.grid(row=1, rowspan=1, column=0, columnspan=6, sticky='nsew', **self.frame_grid_settings)
        self.video_visualisation.add('Subject Time Series')
        self.video_visualisation.add('PC Loadings')
        self.video_visualisation.add('PC Reconstruction')
        self.video_visualisation.set('Subject Time Series')

        self.time_series_scroll_frame = customtkinter.CTkScrollableFrame(self.video_visualisation.tab('Subject Time Series'), fg_color='transparent')
        self.time_series_scroll_frame.pack(expand=1, fill=tkinter.BOTH)

        self.videos_sub_frame = customtkinter.CTkScrollableFrame(self.videos_frame, label_text='Videos', **self.frame_settings_level_2_scroll)
        self.videos_sub_frame.grid(row=0, rowspan=2, column=6, sticky='nesw', **self.frame_grid_settings)
        self.videos_sub_frame.columnconfigure(0, weight=2)
        
        self.split_frame_videos = customtkinter.CTkFrame(self.videos_sub_frame, fg_color='transparent')
        self.split_frame_videos.grid(row=0, column=0, sticky='nesw')
        self.split_frame_videos.columnconfigure((0, 1), weight=2)

        self.pca_file_path_textbox_2 = customtkinter.CTkTextbox(self.split_frame_videos, height=100, state='disabled', wrap='word')
        self.pca_file_path_textbox_2.grid(row=1, column=0, columnspan=2, sticky='nsew', **self.grid_settings)

        self.amp_factors_label = customtkinter.CTkLabel(self.split_frame_videos, text='PM Amplification Factors')
        self.amp_factors_label.grid(row=2, column=0, sticky='e', **self.grid_settings)
        self.amp_factors_entry = customtkinter.CTkEntry(self.split_frame_videos, justify='center', textvariable=self.controller.amp_factors_entry, corner_radius=self.corner_radius)
        self.amp_factors_entry.grid(row=2, column=1, sticky='w', **self.grid_settings)

        self.view_type_option_label = customtkinter.CTkLabel(self.split_frame_videos, text='View')
        self.view_type_option_label.grid(row=3, column=0, sticky='e', **self.grid_settings)
        self.view_type_option_menu = customtkinter.CTkOptionMenu(self.split_frame_videos, values=['Orthographic', 'Frontal', 'Sagittal', 'Transverse'], command=self._video_plot_tab_selection, **self.option_menu_settings)
        self.view_type_option_menu.grid(row=3, column=1, sticky='nsew', **self.grid_settings)

        self.refresh_animations_button = customtkinter.CTkButton(self.split_frame_videos, text='Refresh Animations', command=self._video_plot_tab_selection, **self.button_settings)
        self.refresh_animations_button.grid(row=4, column=0, sticky='nesw', **self.grid_settings)
        self.save_animations_button = customtkinter.CTkButton(self.split_frame_videos, text='Save Animated Plots', command=self.controller.save_animated_plots, **self.button_settings)
        self.save_animations_button.grid(row=4, column=1, columnspan=1, sticky='nesw', **self.grid_settings)

        self.animation_saving_progress_bar = customtkinter.CTkProgressBar(self.split_frame_videos, height=32, **self.progress_bar_settings)
        self.animation_saving_progress_bar.grid(row=5, column=0, columnspan=2, sticky='nsew', **self.grid_settings)
        self.animation_saving_progress_bar.grid_remove()

        # Create and configure the time series video plot
        self.canvas_video_time_series = []

        # Create a new figure for the video plots
        self.fig_video_plots = plt.Figure(figsize=(16, 16), dpi=80)

        # Define gridspec layout (5 rows, 4 columns)
        gs = self.fig_video_plots.add_gridspec(5, 4, height_ratios=[1.5, 1.5, 1, 1, 1])

        self.stick_figures = [
            self.fig_video_plots.add_subplot(gs[i, j], projection='3d') if i < 2 else self.fig_video_plots.add_subplot(gs[i, j])
            for i in range(2) for j in range(4)
        ]

        # Create additional subplots for the bar plots (3 full-row subplots)
        self.pp_video_plot = self.fig_video_plots.add_subplot(gs[2, :])
        self.pv_video_plot = self.fig_video_plots.add_subplot(gs[3, :])
        self.pa_video_plot = self.fig_video_plots.add_subplot(gs[4, :])

        # Create and configure the PC loadings plot
        self.canvas_pc_loadings = []
        self.fig_pc_loadings = plt.Figure(figsize=(16, 8), dpi=80)
        gs = self.fig_pc_loadings.add_gridspec(2, 4)
        self.stick_figures_pc_loadings = [self.fig_pc_loadings.add_subplot(gs[i, j], projection='3d') for i in range(2) for j in range(4)]

        # Create and configure the PC loadings plot
        self.canvas_pc_reconstruction = []
        self.fig_pc_reconstruction = plt.Figure(figsize=(16, 8), dpi=80)
        gs = self.fig_pc_reconstruction.add_gridspec(2, 4)
        self.stick_figures_pc_reconstruction = [self.fig_pc_reconstruction.add_subplot(gs[i, j], projection='3d') for i in range(2) for j in range(4)]


        self.eigenwalkers_subframe = customtkinter.CTkScrollableFrame(self.eigenwalkers_frame)
        self.eigenwalkers_subframe.grid(row=1, column=0, columnspan=10, sticky='nsew', padx=(10, 10), pady=(10, 0))
        
        self.eigenwalker_figAnim3D = plt.Figure(figsize=(8, 8), dpi=80)
        self.eigenwalker_axes = self.eigenwalker_figAnim3D.add_subplot(111, projection='3d')
        self._configure_axes(self.eigenwalker_axes, 0.15)
        self.eigenwalker_axes.set_xlabel('X')
        self.eigenwalker_axes.set_ylabel('Y')
        self.eigenwalker_axes.set_zlabel('Z')
        self.slider_ax_1 = self.eigenwalker_figAnim3D.add_axes([0.1, 0.15, 0.65, 0.03])
        self.slider_ax_2 = self.eigenwalker_figAnim3D.add_axes([0.1, 0.1, 0.65, 0.03])
        self.eigenwalker_canvas = FigureCanvasTkAgg(self.eigenwalker_figAnim3D, self.eigenwalkers_subframe)
        self.eigenwalker_canvas.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)
        self.eigenwalker_canvas.draw()


    def _initialise_tooltips(self):
        self.all_tooltips.append(CreateToolTip(self.file_path_button, text="Select file(s) to load\nAccepts CSV and TXT files\nMarkers identified by column suffixes\n(e.g., 'marker_x', 'marker_y')\nChanges to file contents are acceptable\nNote: Input files are read-only"))
        self.all_tooltips.append(CreateToolTip(self.pca_mode_option_menu, text="Select between 'All Subjects Together' and 'Every Dataset Separately'\nAll Subjects Together: One PCA across all subjects, shared PC loadings\nEvery Dataset Separately: Individual PCA for each subject, unique PC loadings"))
        self.all_tooltips.append(CreateToolTip(self.delimiter_entry, text="Specify delimiter for input data files\nDefault is ',' if left blank"))
        self.all_tooltips.append(CreateToolTip(self.freq_entry, text="Enter sample frequency\nin Hz (e.g., 1000.0)"))
        self.all_tooltips.append(CreateToolTip(self.del_rows_entry, text="Remove specified rows from datasets\nUse Python index/slicing syntax\nExample: \"[:10], [200:]\" removes rows 0-9 and 200 onwards"))
        self.all_tooltips.append(CreateToolTip(self.del_markers_entry, text="Remove markers using \nPython index/slicing syntax \nfrom all datasets \nExample: \"[39:]\""))
        self.all_tooltips.append(CreateToolTip(self.gap_filling_option_menu, text="Imputes None and NaN values using pandas.DataFrame.interpolate"))
        self.all_tooltips.append(CreateToolTip(self.data_filtering_option_menu, text="Filter data:\nOff - No filtering\nButterworth - Use specified order and cut-off frequency"))
        self.all_tooltips.append(CreateToolTip(self.centring_option_menu, text="Centring each row/sample to a reference position\n Useful for subject movement in 3D tracking\n Options: Off - no centring\n Mean Marker Position - subtract mean of markers\n Mean Marker Pos. (Weights) - subtract weighted mean of markers using weights column\n Mean Marker Pos. (Centre Ref.) - subtract weighted mean of markers using Centre Ref. column"))
        self.all_tooltips.append(CreateToolTip(self.align_orientation_option_menu, text="Align orientation to a specific axis\nUse Centre Ref column's highest values\nHint: adjust values to invert normal\nButterworth filtering applied"))
        self.all_tooltips.append(CreateToolTip(self.orientation_cutoff_freq_entry, text="Set orientation frequency cut-off\nRecommended 0.0 to 1.0\nIrrelevant if 'Align Orientation' is off"))
        self.all_tooltips.append(CreateToolTip(self.normalisation_option_menu, text="Normalisation methods\nOff: No normalisation. Not recommended unless data is pre-normalised\nMED: Divides by mean Euclidean distance for each column\nMean Dist. 2 Markers: Divides by mean distance between two markers\nMax Range 1st Coords: Divides by maximum range of first coordinates\nMax Range 2nd Coords: Divides by maximum range of second coordinates\nMax Range 3rd Coords: Divides by maximum range of third coordinates"))
        self.all_tooltips.append(CreateToolTip(self.weights_option_menu, text="Enable the weighting of marker positions\nby the values in the weight column\n(weights are not normalised)"))
        self.all_tooltips.append(CreateToolTip(self.coordinate_transformation_option_menu, text="Coordinate Transform\nTransform data into a different coordinate system before PCA\nOptions:\nOff: Keep X, Y, Z\nCartesian -> Spherical (3D): Convert to Phi, Theta, R with continuous angle values"))
        self.all_tooltips.append(CreateToolTip(self.pp_filter_option_menu, text="Specify post-processing filtering for\nposition, velocity, and acceleration\nusing Butterworth filter\nOrder and cut-off frequency are configurable"))
        self.all_tooltips.append(CreateToolTip(self.pv_filter_option_menu, text="Specify post-processing filtering for\nposition, velocity, and acceleration\nusing Butterworth filter\nOrder and cut-off frequency are configurable"))
        self.all_tooltips.append(CreateToolTip(self.pa_filter_option_menu, text="Specify post-processing filtering for\nposition, velocity, and acceleration\nusing Butterworth filter\nOrder and cut-off frequency are configurable"))
        self.all_tooltips.append(CreateToolTip(self.pm_filter_order_entry, text="Set Butterworth filter order\nfor data pre-processing\nand post-processing"))
        self.all_tooltips.append(CreateToolTip(self.pm_filter_cut_off_entry, text="Controls Butterworth cut-off frequency\nfor pre- and post-processing of\ndata and time series"))
        self.all_tooltips.append(CreateToolTip(self.loocv_checkbox, text="Calculates LOOCV of PCA\n'All Subjects Together': on concatenated data\n'Each Dataset Separately': for each subject's PCA\nResults in plot and CSV"))
        self.all_tooltips.append(CreateToolTip(self.freq_harmonics_entry, text="Leave blank to estimate sine approximations independently\nEnter values to set harmonic relationships\nExample: \"1,1,2,2\" sets PC2 same as PC1\n and PC3, PC4 are double PC1 frequency\nUse \"0\" to skip a PC's harmonic setting"))
        self.all_tooltips.append(CreateToolTip(self.run_pca_button, text="Pre-process input files \nPerform PCA analysis \nCross-validate if enabled \nSave results"))
        self.all_tooltips.append(CreateToolTip(self.subject_del_rows_entry, text="Delete rows for a specific subject\nUse Python index/slicing syntax\nExample: '[:10], [200:]'\nAffects only selected subject"))
        self.all_tooltips.append(CreateToolTip(self.subject_del_markers_entry, text="Delete markers for a specific subject\nUse Python index/slicing syntax\nExample: [39:] to remove markers larger than index 38"))
        self.all_tooltips.append(CreateToolTip(self.subject_flip_x_checkbox, text="Specify axes to flip\nUseful for correcting flipped data\nor adjusting Z-axis direction"))
        self.all_tooltips.append(CreateToolTip(self.subject_flip_y_checkbox, text="Specify axes to flip\nUseful for correcting flipped data\nor adjusting Z-axis direction"))
        self.all_tooltips.append(CreateToolTip(self.subject_flip_z_checkbox, text="Specify axes to flip\nUseful for correcting flipped data\nor adjusting Z-axis direction"))
        self.all_tooltips.append(CreateToolTip(self.subject_eigenwalker_group_entry, text="..."))
        self.all_tooltips.append(CreateToolTip(self.appearance_mode_option_menu, text="This option changes the theme of the UI between 'System', 'Light' or 'Dark'"))
        self.all_tooltips.append(CreateToolTip(self.scaling_option_menu, text="This option changes the scale of the UI"))
        self.all_tooltips.append(CreateToolTip(self.amp_factors_entry, text="Amplify 3D plot motion\nExample: '[2, 3]' amplifies\nPC1 by 2x and PC2 by 3x\nPress 'Refresh Animations' to apply"))
        self.all_tooltips.append(CreateToolTip(self.view_type_option_menu, text="Set the view to Orthographic, Frontal, Sagittal, or Transverse\nHint: Drag the mouse to shift the 3D view"))


    def _toggle_tooltips(self):
        if self.controller.tooltips_checkbox.get():
            for tooltip in self.all_tooltips:
                tooltip.enable()
        else:
            for tooltip in self.all_tooltips:
                tooltip.disable()


    def _ask_save_as_filename(self):
        file_path = tkinter.filedialog.asksaveasfilename(parent=self, title='Choose Save Folder', filetypes=[('PCA', '.pca')])
        return file_path


    def _ask_open_filename(self):
        file_path = tkinter.filedialog.askopenfilename(parent=self, title='Choose a File', filetypes=[('PCA', '.pca')])
        return file_path


    def _ask_open_filenames(self):
        # Prompt the user to select files if not provided
        new_file_path_list = tkinter.filedialog.askopenfilenames(
            parent=self, title='Choose Files', filetypes=[('CSV', '.csv'), ('TEXT', '.txt')]
        )

        return new_file_path_list


    def _on_closing(self, return_action=False):
        '''
        Handle the closing event of the main window. Create a quit dialog
        if it doesn't exist or has been destroyed. Otherwise, bring the 
        existing dialog into focus.
        '''
        # Check if quit_dialog is None or destroyed
        if not (self.quit_dialog and self.quit_dialog.winfo_exists()):
            # Create and display the quit dialog
            self.quit_dialog = QuitDialog(self, return_action=return_action)
            self.quit_dialog.transient(self)  # Keep on top of the main window
            self.quit_dialog.grab_set()  # Make the dialog modal
            action = self.quit_dialog.show_dialog()  # Wait until the dialog is closed
        else:
            # If the quit dialog exists, bring it to focus
            self.quit_dialog.focus()

        if action == 'cancel':
            return
        elif action == 'save':
            self.controller.save_project_file()


    def _show_pane(self, pane):
        '''
        Show the specified pane and update sidebar buttons accordingly.
        '''

        # Loop through widgets, hide them, and update sidebar button colors
        for i, widget in enumerate(self.sidebar_frames_list):
            widget.grid_forget()
            # Change color of sidebar buttons
            self.sidebar_buttons[i + 2].configure(fg_color=self.sidebar_button_settings['fg_color'])

        # Change color of currently opened pane sidebar button
        self.sidebar_buttons[self.sidebar_frames_list.index(pane) + 2].configure(fg_color=self.option_menu_settings['button_color'])

        # Display the specified pane
        pane.grid(row=0, column=1, sticky='nsew')


    def update_appearance(self):
        customtkinter.set_appearance_mode(self.appearance_mode_option_menu.get())


    def update_scaling(self, *args):
        '''
        Changes the scaling factor of widgets and updates appearance accordingly.
        '''
        # Convert percentage string to float
        new_scaling_float = int(self.scaling_option_menu.get().replace('%', '')) / 100

        # Apply new scaling
        customtkinter.set_widget_scaling(new_scaling_float)

        # Hide progress bars
        self.progress_bar.grid_remove()
        self.animation_saving_progress_bar.grid_remove()



    def _get_table_values(self, column: str):
        '''
        Get values from the specified column of the table.
        '''
        values = []
        for item_id in self.table.get_children():
            
            weight, centre_ref, skeleton, colour = self.table.item(item_id, 'values')

            if column == 'weights':
                if weight == '' or not isfloat(weight):
                    weight = 0.0
                value = float(weight)

            elif column == 'centre_ref':
                if centre_ref == '' or not isfloat(centre_ref):
                    centre_ref = 0.0
                value = float(centre_ref)
                
            elif column == 'skeleton':
                value = ''
                if isposint(skeleton):
                    value = int(skeleton)
                    
            elif column == 'colour':
                value = str(colour)

            values.append(value)

        return values


    def update_file_path_label(self, data_file_path_list):
        # Update the file path label
        self.file_path_label.configure(state='normal')
        self.file_path_label.delete('0.0', 'end')
        self.file_path_label.insert('0.0', self._format_file_paths(data_file_path_list))
        self.file_path_label.configure(state='disabled')
        

    def _show_bug_plot(self):
        '''Display 3D animation of bug movement and Frequency Spectrum plot.'''
        print('Refresh 3D Animation')
        
        if self.controller.current_subject_id.get() not in self.controller.subj_pca_models:
            return
        
        if self.controller.current_subj.bug_plot_data.empty:
            return

        self.BugPlot.cla()

        frame_count = len(self.controller.current_subj.bug_plot_data)

        self.current_full_skeleton = self.controller.current_subj.get_skeleton()
        self.set_3d_axis_limits(self.BugPlot,  self.controller.current_subj.bug_plot_data.iloc[:].to_numpy(), 0.95)
        self.BugPlot.set_xlabel('X')
        self.BugPlot.set_ylabel('Y')
        self.BugPlot.set_zlabel('Z')
                                
        # Initialise lines for insect legs in each stick figure plot
        self.lines2 = [self.BugPlot.plot([], [], [], 'o-', color=c, lw=1)[0]
                       for c in self.controller.current_subj.line_colors]
        
        def _init_animation():
            '''Initialize the bug animation.'''
            for line in self.lines2:
                line.set_data([], [])
                line.set_3d_properties([])
            return self.lines2

        def _animate(frame_number):
            '''Animate the bug movement.'''
            if self.controller.current_subj.bug_plot_data.empty or self.pca_busy:
                for line in self.lines2:
                    line.set_data([], [])
                    line.set_3d_properties([])
                self.figAnim3D.suptitle(f'{self.controller.current_subject_id.get()} | Row: -  | Time: -')
                self.figAnim3D.canvas.draw()
                return self.lines2
            
            else:
                coords = self.controller.current_subj.bug_plot_data.iloc[frame_number].to_numpy().reshape(-1, 3)
                
                for i, (start, end) in enumerate(self.current_full_skeleton):
                    if len(coords) > start and len(coords) > end:  # Only plot as many skeleton lines as there are data points
                        x_values = [coords[start, 0], coords[end, 0]]
                        y_values = [coords[start, 1], coords[end, 1]]
                        z_values = [coords[start, 2], coords[end, 2]]
                        self.lines2[i].set_data(x_values, y_values)
                        self.lines2[i].set_3d_properties(z_values)
                
                fs = self.controller.current_subj.sample_freq
                self.figAnim3D.suptitle(f'{self.controller.current_subject_id.get()} | Row: {frame_number}/{frame_count} | Time: {frame_number/fs:.2f} s')
                self.figAnim3D.canvas.draw()
                return self.lines2
        
        # Set up the Tkinter canvas
        self._destroy_all_canvas()
        self.canvas = FigureCanvasTkAgg(self.figAnim3D, master=self.plot_3d_subframe)
        self.canvas.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)#.grid(row=0, column=0, sticky='nsew', **self.grid_settings)
        self.figAnim3D.suptitle(self.controller.current_subject_id.get())

        
        fs = self.controller.current_subj.sample_freq
        
        # Create animation
        self.bug_plot_anim = animation.FuncAnimation(
            self.figAnim3D, _animate, frames=frame_count,
            init_func=_init_animation, repeat=True, blit=True, interval=0#int(1000 / fs)
        )
        
        self.canvas.draw()

        # Frequency Spectrum plot
        df = self.controller.current_subj.results
        if df.empty:
            raise TerminatingError("No PCA results file paths found. Run analysis and save the project.")
            
        time_series_PP1 = np.array(df.loc[('PP Time Series (Position)')].iloc[:, 0])

        self.plot_welch.clear()
        self.plot_welch.magnitude_spectrum(np.array(time_series_PP1), Fs=fs, scale='dB')
        self.plot_welch.set_ylabel('PSD [dB]')
        self.plot_welch.set_xlabel('Frequency [Hz]')
        self.plot_welch.title.set_text(f'Frequency Spectrum of PP1 [{self.controller.current_subject_id.get()}]')
        self.plot_welch.set_xlim(0)
        self.plot_welch.grid(True)
        #self.fig_welch.tight_layout()
        self.canvas_welch.draw()

        self.plot_orientation_series.clear()
        self.plot_orientation_freq.clear()
        orientation_plot_original = self.controller.current_subj.orientation_original
        orientation_plot_drift = self.controller.current_subj.orientation_drift
        orientation_plot_resultant = self.controller.current_subj.orientation_resultant
        
        if self.align_orientation_option_menu.get() != 'Off' and len(orientation_plot_original) > 1:
            # Orientation Plot
            xs = np.array(range(len(orientation_plot_original))) / fs
            if self.align_orientation_option_menu.get() == 'Align Orientation (X-Axis)':
                self.plot_orientation_series.axhline(y=np.pi/2, color='k', linestyle='--', linewidth=1)
            else:
                self.plot_orientation_series.axhline(y=0, color='k', linestyle='--', linewidth=1)
            
            self.plot_orientation_series.plot(xs, orientation_plot_original, label='Original Orientation')
            self.plot_orientation_series.plot(xs, orientation_plot_drift, label='Estimated Offset', linewidth=2)
            self.plot_orientation_series.plot(xs, orientation_plot_resultant, label='Resultant Orientation', linewidth=2)
            self.plot_orientation_series.set_xlim(0, xs[-1])
            self.plot_orientation_series.title.set_text(f'Original vs Corrected Orientation Plot [{self.controller.current_subject_id.get()}]')
            self.plot_orientation_series.set_ylabel('Angle (rad)')
            self.plot_orientation_series.set_xlabel('Time (s)')
            self.plot_orientation_series.legend()
            
            self.plot_orientation_freq.magnitude_spectrum(orientation_plot_original, Fs=fs, color='tab:blue', scale='dB', label='Original Orientation')
            self.plot_orientation_freq.magnitude_spectrum(orientation_plot_resultant, Fs=fs, color='tab:green', scale='dB', label='Filtered Orientation')
            self.plot_orientation_freq.set_xlim(0, 6)
            self.plot_orientation_freq.title.set_text(f'Frequency Spectrum of Orientation [{self.controller.current_subject_id.get()}]')
            self.plot_orientation_freq.set_ylabel('Magnitude')
            self.plot_orientation_freq.set_xlabel('Frequency (Hz)')
            self.plot_orientation_freq.legend()
            #self.fig_orientation.tight_layout()

        self.canvas_orientation.draw()

        self.plot_loocv.clear()
        self.plot_loocv_2.clear()
        if self.loocv_checkbox.get() and 'PRESS Naive' in df.index and 'PRESS Approx' in df.index:
            # Plotting the results
            PRESS_naive = np.array(df.loc[('PRESS Naive')])[0]
            PRESS_naive = PRESS_naive[~np.isnan(PRESS_naive)]
            PRESS_approx = np.array(df.loc[('PRESS Approx')])[0]
            PRESS_approx = PRESS_approx[~np.isnan(PRESS_approx)]
            
            # Plot Naive Reconstruction Error
            self.plot_loocv.plot(PRESS_naive, 'k.--', label='Naive Reconstruction Error')
            self.plot_loocv.set_xlabel('Number of PCs')
            self.plot_loocv.set_xticks(np.arange(0, len(PRESS_naive)))
            self.plot_loocv.set_xticklabels(np.arange(1, len(PRESS_naive) + 1))
            self.plot_loocv.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)
            self.plot_loocv.set_ylabel('Standardized Cross-validation error')
            self.plot_loocv.get_yaxis().set_ticks([])
            original_ylim = self.plot_loocv.get_ylim()
            self.plot_loocv.set_ylim(original_ylim[0], original_ylim[1] * 2)
            if 'pca_all_together' in self.controller.current_subj.results_file_path:
                self.plot_loocv.title.set_text(f'Leave-One-Out Cross-Validation [All Together]')
            else:
                self.plot_loocv.title.set_text(f'Subject Specific Leave-One-Out Cross-Validation [{self.controller.current_subject_id.get()}]')

            # Plot Approximate Reconstruction Error on the secondary y-axis
            self.plot_loocv_2.plot(PRESS_approx, 'r.-', label='Approximate Reconstruction Error')
            self.plot_loocv_2.tick_params(axis='y', left=False, labelleft=False, right=False, labelright=False)
            self.plot_loocv_2.get_yaxis().set_ticks([])

            self.fig_loocv.legend()
            #self.fig_loocv.tight_layout()

        self.canvas_loocv.draw()


    def _show_plots(self):
        '''Refresh and display plots based on the current results CSV.'''
        print('Refresh Plots')
        self._destroy_all_canvas()

        df = self.controller.current_subj.results
        if df.empty:
            raise TerminatingError("No PCA results file paths found. Run analysis and save the project.")
        
        num_of_PCs_to_show = int(string_to_list(self.controller.ev_num_of_pcs_entry.get(), [12])[0])
        self._plot_ev(df, num_of_PCs_to_show)
        self._plot_scores(df, num_of_PCs_to_show)
            

    def _plot_ev(self, df, num_of_PCs_to_show):
        '''Plot eigenvalue-related charts.'''
        plot_configs = [
            ('Explained VAR Ratio', self.plot_ev_1, 'Eigenvalues - VAR', 'Explained Variance by Component [%]', 'bar'),
            ('Explained STD Ratio', self.plot_ev_2, 'Eigenvalues - STD', 'Explained STD by Component [%]', 'bar'),
            ('Explained Cumulative VAR', self.plot_ev_3, 'Cumulative Eigenvalues - VAR', 'Explained Variance [%]', 'line'),
            ('Explained Cumulative STD', self.plot_ev_4, 'Cumulative Eigenvalues - STD', 'Explained STD [%]', 'line')
        ]

        for label, plot, title, ylabel, plot_type in plot_configs:
            data = df.loc[label].dropna(axis=1).to_numpy()[0][:num_of_PCs_to_show] * 100
            plot.clear()
            if plot_type == 'bar':
                plot.bar(range(1, len(data) + 1), data)
            elif plot_type == 'line':
                plot.plot(range(1, len(data) + 1), data, marker='o', markersize=3)
            plot.title.set_text(title)
            plot.set_ylabel(ylabel)
            plot.set_xlabel('PC - Component Nr.')
            plot.set_ylim([0, 100])
            plot.set_xticks(range(1, len(data) + 1, len(data) // 21 + 1))

        self.fig_ev.suptitle(f'Eigenvalues Plots [{self.controller.current_subject_id.get()}]')
        self.fig_ev.tight_layout()
        self.canvas_ev.draw()


    def _plot_scores(self, df, num_of_PCs_to_show):

        self.score_plot_axes = [[], [], []]
        num_of_rows = num_of_PCs_to_show // 4 + (1 if (num_of_PCs_to_show % 4) > 0 else 0)
        num_of_cols = min(num_of_PCs_to_show, 4)

        for i in range(3):
            self.fig_score_plots[i].clear()
            self.canvas_score_plots[i].draw()
            self.fig_score_plots[i].set_figheight(num_of_rows*3)
            self.canvas_score_plots[i].get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.X, expand=1)
            for k in range(num_of_PCs_to_show):
                self.score_plot_axes[i].append(self.fig_score_plots[i].add_subplot(num_of_rows, num_of_cols, int(k // 4 * num_of_cols + k % 4 + 1)))
            
        '''Plot score plots.'''
        plot_configs = [
            ('PP Time Series (Position)', 'PP Metrics', 'r', True),  # Level 0, Level 1, Plot colour, Plot sin approx.
            ('PV Time Series (Velocity)', 'PV Metrics', 'b', False),
            ('PA Time Series (Acceleration)', 'PA Metrics', 'g', False)
        ]

        for i, (time_series_label, metrics_label, color, show_sin_approx) in enumerate(plot_configs):
            self.fig_score_plots[i].suptitle(f'Score Plots [{self.controller.current_subject_id.get()}]')
            for k in range(len(self.score_plot_axes[i])):
                self._plot_single_score(self.score_plot_axes[i][k], df.loc[(time_series_label)], df.loc[(metrics_label)], k, color, show_sin_approx)

            self.fig_score_plots[i].tight_layout()
            self.canvas_score_plots[i].draw()


    def _plot_single_score(self, ax, scores, metrics, plot_index, color, show_sin_approx):
        '''Plot a single score plot.'''
        ax.clear()
        fs = self.controller.current_subj.sample_freq
        tt = np.linspace(0, scores.iloc[:, plot_index].shape[0] / fs, scores.iloc[:, plot_index].shape[0])
        ax.plot(tt, scores.iloc[:, plot_index].to_numpy(), color=color, lw=0.75)

        if self.sine_approx_checkbox.get() and show_sin_approx:
            def sinfunc(t, A, w, p):
                return A * np.sin(w * t + p)

            A = float(metrics.loc[('Sin Approx. Amplitude')].iloc[plot_index])
            w = float(metrics.loc[('Sin Approx. Omega')].iloc[plot_index])
            p = float(metrics.loc[('Sin Approx. Phase')].iloc[plot_index])
            ax.plot(tt, sinfunc(tt, A, w, p), 'k-', label='y fit curve', lw=0.75, alpha=0.7)
            if not np.isnan(w):
                ax.text(0.01, 0.01, f'Freq: {round(w / (2. * np.pi), 2)}, Phi: {round(p, 2)}', fontsize=12, ha='left', va='bottom', transform=ax.transAxes)
        
        ax.title.set_text(f'PP{plot_index + 1}')
        ax.set_xlabel('Time [s]')
        ax.set_xlim(0, tt[-1])
        

    def _video_plot_tab_selection(self, selection=None, *args):
        '''Switch video plot tab based on selection.'''

        df = self.controller.current_subj.results
        if df.empty:
            raise TerminatingError("No PCA results file paths found. Run analysis and save the project.")
        
        self.initial_synchronization_done = False
        
        p_0 = df.loc[('Data Mean')].to_numpy()  # Input data mean
        p = df.loc[('Loadings')].to_numpy()  # PC Loadings
        self.controller.current_subj.d_norm = float(df.loc['Normalisation Factor'].to_numpy()[0, 0])
        self.controller.current_subj.weight_vector = df.loc['Weight Vector'].to_numpy()[0]

        # Custom PC scaling from user input
        amplification_factors = string_to_list(self.amp_factors_entry.get())
        amplification_factors = amplification_factors[:8] + [1] * max(0, 8 - len(amplification_factors))

        scores = {
            'PP': df.loc['PP Time Series (Position)'].to_numpy(),
            'PV': df.loc['PV Time Series (Velocity)'].to_numpy(),
            'PA': df.loc['PA Time Series (Acceleration)'].to_numpy()
        }


        no_of_pcs = min(scores['PP'].shape)
        scores = {key: value[:, :no_of_pcs] for key, value in scores.items()}

        """frame_count = 103
        sin_approx_score = np.zeros((frame_count, 7))
        def sinfunc(t, A, w, phi):
                return A * np.sin(w * t + phi)

        for i in range(7):
            A = float(self.controller.current_subj.results.loc[('PP Metrics')].loc[('Sin Approx. Amplitude')].iloc[i])
            w = float(self.controller.current_subj.results.loc[('PP Metrics')].loc[('Sin Approx. Omega')].iloc[i])
            phi = float(self.controller.current_subj.results.loc[('PP Metrics')].loc[('Sin Approx. Phase')].iloc[i])
            tt = np.linspace(0, 1.89274, frame_count)
            sin_approx_score[:, i] = sinfunc(tt, A, w, phi)"""
        
        if not selection:
            selection = self.video_visualisation.get()

        if selection == 'Subject Time Series' or selection == 1:
            self._show_subject_time_series(scores, p_0, p, amplification_factors)
        elif selection == 'PC Loadings' or selection == 2:
            self._show_pc_loadings(p_0, p, amplification_factors)
        else:
            self._show_pc_reconstruction(scores['PP'], p_0, p, amplification_factors)  #sin_approx_score


    def _show_subject_time_series(self, scores, p_0, p, amplification_factors):
        '''Refresh and display the subject time series animations.'''
        print('Refresh Time Series')
        
        # List to store reconstructed data for each principal component
        reconstructed_data_pcs = [
            self.controller.current_subj.reconstruct_data(
                                    p_0, (amplification_factors[k] * scores['PP'])[:, [k]], p[:, [k]],
                                    self.coordinate_transformation_option_menu.get())
            for k in range(8)
        ]

        # Highest frame count is the lowest row length of what is being plotted
        frame_count = np.amin([reconstructed_data_pcs[0].shape[0], scores['PP'].shape[0], scores['PV'].shape[0], scores['PA'].shape[0]])
        max_activity = {k: np.max(np.abs(ts)) for k, ts in scores.items()}

        for k, ax in enumerate(self.stick_figures):
            ax.cla()
            self._configure_axes(ax, 0.15)
            self.set_3d_axis_limits(ax, reconstructed_data_pcs[k], 0.95)
            ax.set_title(f'PC{k + 1}')

        self.current_full_skeleton = self.controller.current_subj.get_skeleton()

        # Initialize lines for insect legs in each stick figure plot (3D)
        self.lines = [
            [ax.plot([], [], [], 'o-', color=c, lw=1, ms=1.5)[0]
            for c in self.controller.current_subj.line_colors]
            for ax in self.stick_figures
        ]

        def _init_animation():
            '''Initialize the animation.'''
            for k, ax in enumerate(self.stick_figures):
                for line in self.lines[k]:
                    line.set_data([], [])
                    line.set_3d_properties([])
            return np.concatenate(self.lines)

        # Animation function. This is called sequentially.
        def _animate(frame_number):
            # Update the bar chart
            for k, (label, container) in enumerate(zip(['PP', 'PV', 'PA'], [bar_container1, bar_container2, bar_container3])):
                for value, rect in zip(scores[label][frame_number, :], container):
                    rect.set_height((value / max_activity[label]) * 100)

            # Update stick figure plots
            for k, (lines, ax) in enumerate(zip(self.lines, self.stick_figures)):
                coords = reconstructed_data_pcs[k][frame_number].reshape(-1, 3)
                for i, (start, end) in enumerate(self.current_full_skeleton):
                    if len(coords) > start and len(coords) > end:  # Only plot as many skeleton lines as there are data points
                        x_values = [coords[start, 0], coords[end, 0]]
                        y_values = [coords[start, 1], coords[end, 1]]
                        z_values = [coords[start, 2], coords[end, 2]]
                        lines[i].set_data(x_values, y_values)
                        lines[i].set_3d_properties(z_values)

            return [*bar_container1, *bar_container2, *bar_container3, *np.concatenate(self.lines)]

        # Set up the Tkinter canvas
        self._destroy_all_canvas()
        self.canvas_video_time_series = FigureCanvasTkAgg(self.fig_video_plots, master=self.time_series_scroll_frame) #self.video_visualisation.tab('Subject Time Series'))
        self.canvas_video_time_series.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)
        self.fig_video_plots.suptitle(f'Time Series [{self.controller.current_subject_id.get()}]')
        self.fig_video_plots.canvas.mpl_connect('motion_notify_event', self._synchronize_rotation)
        #self.fig_video_plots.tight_layout()
        self.fig_video_plots.subplots_adjust(top=0.95, hspace=0.225, left=0.1, right=0.95)

        plots = [
            (self.pp_video_plot, 'PP Activity'),
            (self.pv_video_plot, 'PV Activity'),
            (self.pa_video_plot, 'PA Activity')
        ]

        for plot, title in plots:
            plot.cla()
            plot.set_ylim(-100, 100)
            plot.set_xlim(0.5, 20.5)  # First 20 PCs
            plot.set_xticks(range(1, 21))
            plot.set_title(title)
            plot.set_ylabel('Relative Amplitude [%]')

        bar_container1 = self.pp_video_plot.bar(range(1, 21), np.zeros(20), align='center', alpha=0.6, color='r')  ## PP Plots
        bar_container2 = self.pv_video_plot.bar(range(1, 21), np.zeros(20), align='center', alpha=0.6, color='b')  ## PV Plots
        bar_container3 = self.pa_video_plot.bar(range(1, 21), np.zeros(20), align='center', alpha=0.6, color='g')  ## PA Plots

        for ax in self.stick_figures:
            self._set_predefined_view(ax, self.view_type_option_menu.get())
        
        # Create animation
        fs = self.controller.current_subj.sample_freq
        self.time_series_anim = animation.FuncAnimation(
            self.fig_video_plots, _animate, frames=frame_count,
            init_func=_init_animation, repeat=True, blit=True, interval=1000 / fs)

        self.canvas_video_time_series.draw()
  

    def _show_pc_loadings(self, p_0, p, amplification_factors):
        '''Refresh and animate PC loadings visualization.'''
        print('Refresh PC Loadings')
        
        frame_count = 25  # Number of frames animation should last (lower number increases sweep speed)

        # Generate a range of values to sweep over
        sweep = np.tile(2 * signal.sawtooth(np.linspace(0, 2 * np.pi, frame_count + 1), width=0.5)[:-1, np.newaxis], (1, len(p)))  # Triangle wave

        # Reconstruct data for each principal component
        reconstructed_data_pcs = [
            self.controller.current_subj.reconstruct_data(
                                    p_0, (amplification_factors[k] * sweep)[:, [k]], p[:, [k]],
                                    self.coordinate_transformation_option_menu.get())
            for k in range(8)
        ]

        for k, ax in enumerate(self.stick_figures_pc_loadings):
            ax.cla()
            self._configure_axes(ax, 0.15)
            self.set_3d_axis_limits(ax, reconstructed_data_pcs[k], 0.95)
            ax.set_title(f'PC{k + 1}')

        self.current_full_skeleton = self.controller.current_subj.get_skeleton()
                       
        # Initialize lines for insect legs in each stick figure plot
        self.lines_loadings = [
            [ax.plot([], [], [], 'o-', color=c, lw=1, ms=1.5)[0]
            for c in self.controller.current_subj.line_colors]
            for ax in self.stick_figures_pc_loadings
        ]

        def _init_animation():
            '''Initialize the animation.'''
            for k, ax in enumerate(self.stick_figures_pc_loadings):
                for line in ax.get_lines():
                    line.set_data([], [])
                    line.set_3d_properties([])
            return np.concatenate(self.lines_loadings)

        # Animation function. This is called sequentially.
        def _animate(frame_number):
            # Update stick figure plots
            for k, (lines, ax) in enumerate(zip(self.lines_loadings, self.stick_figures_pc_loadings)):
                coords = reconstructed_data_pcs[k][frame_number].reshape(-1, 3)
                for i, (start, end) in enumerate(self.current_full_skeleton):
                    if len(coords) > start and len(coords) > end:  # Only plot as many skeleton lines as there are data points
                        x_values = [coords[start, 0], coords[end, 0]]
                        y_values = [coords[start, 1], coords[end, 1]]
                        z_values = [coords[start, 2], coords[end, 2]]
                        lines[i].set_data(x_values, y_values)
                        lines[i].set_3d_properties(z_values)
            return np.concatenate(self.lines_loadings)

        # Set up the Tkinter canvas
        self._destroy_all_canvas()
        self.canvas_pc_loadings = FigureCanvasTkAgg(self.fig_pc_loadings, master=self.video_visualisation.tab('PC Loadings'))
        self.canvas_pc_loadings.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)
        self.fig_pc_loadings.canvas.mpl_connect('motion_notify_event', self._synchronize_rotation)
        self.fig_pc_loadings.suptitle(f'PC Loadings [{self.controller.current_subject_id.get()}]')
        self.fig_pc_loadings.tight_layout()
        self.fig_pc_loadings.subplots_adjust(top=0.95, hspace=0.2)

        for ax in self.stick_figures_pc_loadings:
            self._set_predefined_view(ax, self.view_type_option_menu.get())

        # Create animation
        fs = self.controller.current_subj.sample_freq
        self.loadings_anim = animation.FuncAnimation(
            self.fig_pc_loadings, _animate, frames=frame_count,
            init_func=_init_animation, repeat=True, blit=True, interval=1000/fs
        )

        self.canvas_pc_loadings.draw()


    def _show_pc_reconstruction(self, pp_score, p_0, p, amplification_factors):
        '''Refresh and animate PC reconstruction visualization.'''
        print('Refresh PC Reconstruction')
        
        no_of_pcs = min(pp_score.shape)
        frame_count = pp_score.shape[0]  # Number of frames animation should last (lower number increases sweep speed)

        # Reconstruct data for each principal component
        reconstructed_data_pcs = [
            self.controller.current_subj.reconstruct_data(
                                    p_0, (amplification_factors[i] * pp_score)[:, :k], p[:, :k],
                                    self.coordinate_transformation_option_menu.get())
            for i, k in enumerate([1, 2, 3, 4, 5, 6, 7, no_of_pcs])
        ]

        for k, ax in enumerate(self.stick_figures_pc_reconstruction):
            ax.cla()
            self._configure_axes(ax, 0.15)
            self.set_3d_axis_limits(ax, reconstructed_data_pcs[k], 0.9)
            if k == 0:
                ax.set_title(f'PC1')
            elif k < 7:
                ax.set_title(f'PC1-{k + 1}')
            else:
                ax.set_title(f'Original')

        self.current_full_skeleton = self.controller.current_subj.get_skeleton()
                       
        # Initialize lines for insect legs in each stick figure plot
        self.lines_reconstruction = [
            [ax.plot([], [], [], 'o-', color=c, lw=1, ms=1.5)[0]
            for c in self.controller.current_subj.line_colors]
            for ax in self.stick_figures_pc_reconstruction
        ]

        def _init_animation():
            '''Initialize the animation.'''
            for k, ax in enumerate(self.stick_figures_pc_reconstruction):
                for line in ax.get_lines():
                    line.set_data([], [])
                    line.set_3d_properties([])
            return np.concatenate(self.lines_reconstruction)

        # Animation function. This is called sequentially.
        def _animate(frame_number):
            # Update stick figure plots
            for k, (lines, ax) in enumerate(zip(self.lines_reconstruction, self.stick_figures_pc_reconstruction)):
                coords = reconstructed_data_pcs[k][frame_number].reshape(-1, 3)
                for i, (start, end) in enumerate(self.current_full_skeleton):
                    if len(coords) > start and len(coords) > end:  # Only plot as many skeleton lines as there are data points
                        x_values = [coords[start, 0], coords[end, 0]]
                        y_values = [coords[start, 1], coords[end, 1]]
                        z_values = [coords[start, 2], coords[end, 2]]
                        lines[i].set_data(x_values, y_values)
                        lines[i].set_3d_properties(z_values)
            return np.concatenate(self.lines_reconstruction)

        # Set up the Tkinter canvas
        self._destroy_all_canvas()
        self.canvas_pc_reconstruction = FigureCanvasTkAgg(self.fig_pc_reconstruction, master=self.video_visualisation.tab('PC Reconstruction'))
        self.canvas_pc_reconstruction.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)
        self.fig_pc_reconstruction.canvas.mpl_connect('motion_notify_event', self._synchronize_rotation)
        self.fig_pc_reconstruction.suptitle(f'PC Reconstruction [{self.controller.current_subject_id.get()}]')
        self.fig_pc_reconstruction.tight_layout()
        self.fig_pc_reconstruction.subplots_adjust(top=0.95, hspace=0.2)

        for ax in self.stick_figures_pc_reconstruction:
            self._set_predefined_view(ax, self.view_type_option_menu.get())

        # Create animation
        fs = self.controller.current_subj.sample_freq
        self.reconstruction_anim = animation.FuncAnimation(
            self.fig_pc_reconstruction, _animate, frames=frame_count,
            init_func=_init_animation, repeat=True, blit=True, interval=1000/fs
        )

        self.canvas_pc_reconstruction.draw()


    def set_3d_axis_limits(self, ax_3d, subject_data, axes_scale_3d=1.0):
        data_array = subject_data.reshape(-1, 3)
        center = np.mean([np.min(data_array, axis=0), np.max(data_array, axis=0)], axis=0)
        min_max_range = np.ptp(data_array, axis=0).max() * axes_scale_3d
        ax_3d.set_xlim(center[0] - min_max_range/2, center[0] + min_max_range/2)
        ax_3d.set_ylim(center[1] - min_max_range/2, center[1] + min_max_range/2)
        ax_3d.set_zlim(center[2] - min_max_range/2, center[2] + min_max_range/2)


    def _show_eigenwalker_space(self):

        self.controller.calc_eigenwalkers()

        if self.controller.current_eigenwalker_group_id.get() == '':
            return
        
        template_subj, eigenwalker_pca = self.controller.eigenwalker_pca_runs.get(self.controller.current_eigenwalker_group_id.get())

        df = template_subj.results
        # TODO: Make this on a per TrojePCA basis VVVV
        # if df.empty:
        #     raise TerminatingError("No PCA results file paths found. Run analysis and save the project.")

        self.eigenwalker_axes.cla()
        self.slider_ax_1.cla()
        self.slider_ax_2.cla()

        # INPUTS: W_0, V, K, num_of_eigenposture_features
        W_0 = eigenwalker_pca.W_0
        V = eigenwalker_pca.V
        K = eigenwalker_pca.K

        if W_0.shape[0] == 0 or V.shape[0] == 0 or K.shape[0] == 0:
            raise TerminatingError('Unable to plot eigenwalkers. Run PCA.')

        Vt = V.T
        slider_1_value = 0.0
        slider_2_value = 0.0

        min_slider_vals = np.min(K, axis=0)
        max_slider_vals = np.max(K, axis=0)

        # Adjust the subplots for the figure
        self.eigenwalker_figAnim3D.subplots_adjust(left=0.1, bottom=0.25)

        slider_1 = Slider(self.slider_ax_1, 'PC 1', min_slider_vals[0], max_slider_vals[0], valinit=slider_1_value)
        slider_2 = Slider(self.slider_ax_2, 'PC 2', min_slider_vals[1], max_slider_vals[1], valinit=slider_2_value)

        num_of_eigenposture_features = template_subj.results.loc['Loadings'].shape[0]

        # Initial plot
        bug_data = eigenwalker_pca.reconstruct(eigenwalker_pca.transform_k_to_w(np.zeros_like(V[0])), num_of_eigenposture_features,
                                               float(self.controller.freq_entry.get()),
                                               float(df.loc['Normalisation Factor'].to_numpy()[0, 0]),
                                               df.loc['Weight Vector'].to_numpy()[0],
                                               self.coordinate_transformation_option_menu.get()
                                            )

        self.current_full_skeleton = template_subj.get_skeleton()
        self.set_3d_axis_limits(self.eigenwalker_axes, bug_data, 0.95)
                                
        # Initialise lines for insect legs in each stick figure plot
        self.lines3 = [self.eigenwalker_axes.plot([], [], [], 'o-', color=c, lw=1)[0]
                  for c in template_subj.line_colors]

        def _init_animation():
            '''Initialize the bug animation.'''
            for line in self.lines3:
                line.set_data([], [])
                line.set_3d_properties([])
            return self.lines3

        def _animate(frame_number):
            '''Animate the bug movement.'''
            k = np.array([slider_1.val, slider_2.val])
            k = np.pad(k, (0, V.shape[1] - len(k)), 'constant')
            bug_data = eigenwalker_pca.reconstruct(eigenwalker_pca.transform_k_to_w(k), num_of_eigenposture_features,
                                               float(self.controller.freq_entry.get()),
                                               float(df.loc['Normalisation Factor'].to_numpy()[0, 0]),
                                               df.loc['Weight Vector'].to_numpy()[0],
                                               self.coordinate_transformation_option_menu.get()
                                            )
            coords = bug_data[frame_number % len(bug_data)].reshape(-1, 3)
            for i, (start, end) in enumerate(self.current_full_skeleton):
                if len(coords) > start and len(coords) > end:  # Only plot as many skeleton lines as there are data points
                    x_values = [coords[start, 0], coords[end, 0]]
                    y_values = [coords[start, 1], coords[end, 1]]
                    z_values = [coords[start, 2], coords[end, 2]]
                    self.lines3[i].set_data(x_values, y_values)
                    self.lines3[i].set_3d_properties(z_values)
            
            self.eigenwalker_figAnim3D.suptitle(f'[{self.controller.current_eigenwalker_group_id.get()}] Row: {frame_number % len(bug_data)}/{len(bug_data)}')
            self.eigenwalker_figAnim3D.canvas.draw()
            return self.lines3

        self._destroy_all_canvas()
        self.eigenwalker_canvas = FigureCanvasTkAgg(self.eigenwalker_figAnim3D, master=self.eigenwalkers_subframe)
        self.eigenwalker_canvas.get_tk_widget().pack(side=tkinter.LEFT, fill=tkinter.BOTH, expand=True)#.grid(row=0, column=0, sticky='nsew', **self.grid_settings)

        # Create animation
        anim3D = animation.FuncAnimation(
            self.eigenwalker_figAnim3D, _animate, frames=10000,
            init_func=_init_animation, repeat=True, blit=True, interval=0
        )

        self.eigenwalker_canvas.draw()



    def _synchronize_rotation(self, event):
        #if event.name == 'button_release_event':
        elev, azim = event.canvas.figure.axes[0].elev, event.canvas.figure.axes[0].azim  # Get elevation and azimuth angles from the triggered subplot
        xlim, ylim, zlim = event.canvas.figure.axes[0].get_xlim(), event.canvas.figure.axes[0].get_ylim(), event.canvas.figure.axes[0].get_zlim()
        for ax in event.canvas.figure.axes[1:8]:
            if ax.elev != elev or ax.azim != azim:
                ax.view_init(elev=elev, azim=azim)  # Apply same angles to axes with different view

            if not np.allclose(ax.get_xlim(), xlim) or not np.allclose(ax.get_ylim(), ylim) or not np.allclose(ax.get_zlim(), zlim):
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.set_zlim(zlim)  # Synchronize limits if different

        # Redraw the canvas once after initial setup to avoid ghosting
        if not self.initial_synchronization_done:
            event.canvas.draw_idle()
            self.initial_synchronization_done = True


    # Set predefined views
    def _set_predefined_view(self, ax, view_type):
        if view_type == 'Frontal':
            ax.view_init(elev=0, azim=0)
        elif view_type == 'Sagittal':
            ax.view_init(elev=0, azim=-90)
        elif view_type == 'Transverse':
            ax.view_init(elev=90, azim=-90)
        elif view_type == 'Orthographic':
            ax.view_init(elev=45, azim=-45)


    def _destroy_all_canvas(self):
        '''Destroy all canvas widgets.'''
        if self.canvas_video_time_series:
            self.canvas_video_time_series.get_tk_widget().destroy()
        if self.canvas_pc_loadings:
            self.canvas_pc_loadings.get_tk_widget().destroy()
        if self.canvas_pc_reconstruction:
            self.canvas_pc_reconstruction.get_tk_widget().destroy()
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        if  self.eigenwalker_canvas:
            self.eigenwalker_canvas.get_tk_widget().destroy()


    def _update_subject_selection(self, switching_to_subject):
        if switching_to_subject and switching_to_subject != 'None':
            print(f"Switching to {switching_to_subject}")
            self.subject_del_rows_entry.configure(textvariable=self.controller.subj_pca_models[switching_to_subject].subject_UI_settings['rows_to_del'])
            self.subject_del_markers_entry.configure(textvariable=self.controller.subj_pca_models[switching_to_subject].subject_UI_settings['markers_to_del'])
            self.subject_flip_x_checkbox.configure(variable=self.controller.subj_pca_models[switching_to_subject].subject_UI_settings['flip_x'])
            self.subject_flip_y_checkbox.configure(variable=self.controller.subj_pca_models[switching_to_subject].subject_UI_settings['flip_y'])
            self.subject_flip_z_checkbox.configure(variable=self.controller.subj_pca_models[switching_to_subject].subject_UI_settings['flip_z'])
            self.subject_eigenwalker_group_entry.configure(textvariable=self.controller.subj_pca_models[switching_to_subject].subject_UI_settings['eigenwalker_group'])
            
            self.controller.change_current_subj(switching_to_subject)


    def _set_subject_specific_UI_state(self, state):
        self.subject_del_rows_entry.configure(state=state)
        self.subject_del_markers_entry.configure(state=state)
        self.subject_flip_x_checkbox.configure(state=state)
        self.subject_flip_y_checkbox.configure(state=state)
        self.subject_flip_z_checkbox.configure(state=state)
        self.subject_eigenwalker_group_entry.configure(state=state)


    def prepare_ui(self):
        self.pca_busy = True
        self.run_pca_button.grid_remove()
        self.progress_bar.set(0.0)
        self.progress_bar.grid()

        self._set_subject_specific_UI_state('disabled')
        self.prev_subject_id = self.controller.current_subject_id.get()
        
        for w in self.winfo_children():
            self._disable_widgets(w)
            

    def finalise_ui(self):
        for w in self.winfo_children():
            self._enable_widgets(w)

        # Restore the subject selection to the user-selected one before processing started
        self.controller.current_subject_id.set(self.prev_subject_id)
        self._update_subject_selection(self.prev_subject_id)
        self._set_subject_specific_UI_state('normal')

        self.progress_bar.grid_remove()
        self.run_pca_button.grid()
        self.pca_busy = False
        if self.pca_analysis_frame.grid_info():
            self._show_bug_plot()


    def update_pca_file_path_textbox(self, file_paths_str_list):
        self.pca_file_path_textbox.configure(state='normal')
        self.pca_file_path_textbox.delete('0.0', 'end')
        self.pca_file_path_textbox.insert('0.0', self._format_file_paths(file_paths_str_list))
        self.pca_file_path_textbox.configure(state='disabled')

        self.pca_file_path_textbox_2.configure(state='normal')
        self.pca_file_path_textbox_2.delete('0.0', 'end')
        self.pca_file_path_textbox_2.insert('0.0', self._format_file_paths(file_paths_str_list))
        self.pca_file_path_textbox_2.configure(state='disabled')


    # Function to set properties for ticks, grids, axis lines, and background
    def _configure_axes(self, ax, alpha):
        # Adjust tick labels alpha
        ax.tick_params(axis='x', which='both', labelcolor=(0, 0, 0, alpha*4))
        ax.tick_params(axis='y', which='both', labelcolor=(0, 0, 0, alpha*4))
        ax.tick_params(axis='z', which='both', labelcolor=(0, 0, 0, alpha*4))
        
        # Update grid properties
        ax.xaxis._axinfo['grid'].update(color=(0, 0, 0, alpha), linestyle='-', linewidth=0.5)
        ax.yaxis._axinfo['grid'].update(color=(0, 0, 0, alpha), linestyle='-', linewidth=0.5)
        ax.zaxis._axinfo['grid'].update(color=(0, 0, 0, alpha), linestyle='-', linewidth=0.5)
        
        # Set alpha for axis lines
        ax.xaxis.line.set_alpha(alpha)
        ax.yaxis.line.set_alpha(alpha)
        ax.zaxis.line.set_alpha(alpha)
        
        # Set background color and alpha for each axis
        ax.xaxis.set_pane_color((0.95, 0.95, 0.95), alpha=alpha)
        ax.yaxis.set_pane_color((0.95, 0.95, 0.95), alpha=alpha)
        ax.zaxis.set_pane_color((0.95, 0.95, 0.95), alpha=alpha)


    def _format_file_paths(self, file_paths):
        '''
        Format a list of file paths to a presentable string for the UI, grouped by common base directory.

        Parameters:
        - file_paths: list containing file paths

        Returns:
        - str: Formatted string with file paths grouped by common base directory
        '''
        # Find the common base directory path
        common_base_dir = os.path.commonpath(file_paths)
        common_base_dir_formatted = common_base_dir.replace(' ', '\xa0')
        formatted_str = f'{common_base_dir_formatted}:\n'
        
        # Process each file path in the list
        for file_path in file_paths:
            # Extract the relative path from the common base directory
            relative_path = os.path.relpath(file_path, common_base_dir)
            
            # Add formatted path to the string
            formatted_str += f'        {relative_path}\n'
        
        return formatted_str.strip()


    def _disable_widgets(self, widget):
        """
        Recursively disable all widgets in a Tkinter widget hierarchy.
        
        Args:
            widget (tk.Widget): The root widget from which to start the disabling process.
        """
        if isinstance(widget, (tkinter.Canvas, tkinter.Frame, customtkinter.CTkFrame, customtkinter.CTkCanvas, customtkinter.CTkScrollableFrame, customtkinter.CTkScrollbar, customtkinter.CTkTabview)):
            for child in widget.winfo_children():
                self._disable_widgets(child)
        try:
            widget.configure(state='disabled')
            if isinstance(widget, (customtkinter.CTkEntry)):
                widget.configure(text_color=self.entry_settings['text_color_disabled'])
            if isinstance(widget, (customtkinter.CTkCheckBox)):
                widget.configure(fg_color=self.checkbox_fg_color_disabled)

        except Exception:
            pass


    def _enable_widgets(self, widget):
        """
        Recursively disable all widgets in a Tkinter widget hierarchy.
        
        Args:
            widget (tk.Widget): The root widget from which to start the disabling process.
        """
        if isinstance(widget, (tkinter.Canvas, tkinter.Frame, customtkinter.CTkFrame, customtkinter.CTkCanvas, customtkinter.CTkScrollableFrame, customtkinter.CTkScrollbar, customtkinter.CTkTabview)):
            for child in widget.winfo_children():
                self._enable_widgets(child)
        try:
            widget.configure(state='normal')
            if isinstance(widget, (customtkinter.CTkEntry)):
                widget.configure(text_color=self.entry_settings['text_color'])
            if isinstance(widget, (customtkinter.CTkCheckBox)):
                widget.configure(fg_color=self.checkbox_settings['fg_color'])

        except Exception:
            pass


class Tableview(tkinter.ttk.Treeview):
    '''
    Custom subclass of ttk.Treeview that enhances functionality with double-click event handling.
    This class extends ttk.Treeview to support double-click events on rows, which opens a read-only EntryPopup to view and potentially edit cell values.
    Methods:
    - _onDoubleClick(event):
        Event handler executed when a row is double-clicked.
        Retrieves the row and column where the double-click occurred and opens an EntryPopup to display the cell value.
        If an EntryPopup is already open, it closes the existing one before opening a new one for the clicked cell.

    Source: https://stackoverflow.com/questions/18562123/how-to-make-ttk-treeviews-rows-editable
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bind('<Double-1>', self._onDoubleClick)  # Bind double-click to event handler

    def _onDoubleClick(self, event):
        '''Executed when a row is double-clicked to open a read-only EntryPopup.'''

        # Close previous popups if any
        if hasattr(self, 'entryPopup'):
            self.entry_popup.destroy()

        rowid = self.identify_row(event.y)  # Identify clicked row
        column = self.identify_column(event.x)  # Identify clicked column

        if not rowid: return  # Ignore if header is double-clicked

        x, y, width, height = self.bbox(rowid, column)  # Get column position info

        if int(column[1:]) > 0:
            # place Entry popup properly
            text = self.item(rowid, 'values')[int(column[1:]) - 1]
            self.entry_popup = EntryPopup(self, rowid, int(column[1:]) - 1, text, justify='center')
            self.entry_popup.place(x=x + 1, y=y + (height // 2), width=width, height=height, anchor='w')


class EntryPopup(tkinter.ttk.Entry):
    '''
    EntryPopup extends tkinter.ttk.Entry to create a popup widget for editing treeview rows.

    Methods:
    - _on_return(event): Handles updating the treeview item with the entered value.
    - _select_all(*ignore): Selects all text in the entry widget.

    Source: https://stackoverflow.com/questions/18562123/how-to-make-ttk-treeviews-rows-editable
    '''
    def __init__(self, parent, iid, column, text, **kw):
        # Configure custom style
        tkinter.ttk.Style().configure('pad.TEntry', padding='1 1 1 1')
        super().__init__(parent, style='pad.TEntry', **kw)
        self.tv, self.iid, self.column = parent, iid, column

        self.insert(0, text)  # Insert initial text
        self['exportselection'] = False  # Disable exportselection
        self.focus_force()  # Focus on this widget
        self._select_all()  # Select all text

        # Bind keys to methods
        self.bind('<Return>', self._on_return)
        self.bind('<Control-a>', self._select_all)
        self.bind('<Escape>', lambda *ignore: self.destroy())
        self.bind('<FocusOut>', self._on_return)

    def _on_return(self, event):
        vals = self.tv.item(self.iid, 'values')
        vals = list(vals)
        if self.column == 0 or self.column == 1:
            vals[self.column] = self.get() if isfloat(self.get()) else 0.0  # Only allow floats
        elif self.column == 2:
            vals[self.column] = self.get() if isposint(self.get()) else ''  # Only allow ints
        else:
            vals[self.column] = self.get() if is_color_like(self.get()) else '0'

        self.tv.item(self.iid, values=vals) # Update treeview item
        self.destroy() # Destroy widget

    def _select_all(self, *ignore):
        self.selection_range(0, 'end')  # Select all text
        return 'break'  # Interrupt default key-bindings


class QuitDialog(customtkinter.CTkToplevel):
    '''
    Dialog window for confirming user actions related to saving changes.
    Centers the dialog relative to the parent instance and provides options to save changes, discard changes, or cancel.
    '''
    def __init__(self, instance, return_action=False):
        super().__init__(instance)
        
        self.return_action = return_action
        self.selected_action = 'cancel'
        self.instance = instance
        
        # Center the dialog
        self.geometry('480x100+{}+{}'.format(
            self.instance.winfo_x() + self.instance.winfo_width() // 2 - 240,
            self.instance.winfo_y() + self.instance.winfo_height() // 2 - 50
        ))
        
        self.focus_force()
        self.grid_columnconfigure(1, weight=1, minsize=100)
        self.grid_rowconfigure(0, weight=1, minsize=100)
        self.resizable(False, False)
        
        # Update label text if a project is open
        label_text = 'Do you want to save changes?'
        if self.instance.controller.project_name:
            label_text = f'Do you want to save changes to {os.path.basename(self.instance.controller.project_name)}?'
        self.label = customtkinter.CTkLabel(self, text=label_text)
        self.label.pack(padx=10, pady=(10, 5))

        # Create buttons with their respective commands
        customtkinter.CTkButton(self, text='Save', command=self._save_action, **instance.button_settings).pack(padx=10, pady=5, side='left')
        customtkinter.CTkButton(self, text='Don\'t Save', command=self._dont_save_action, **instance.button_settings).pack(padx=10, pady=5, side='left')
        customtkinter.CTkButton(self, text='Cancel', command=self._cancel_action, **instance.button_settings).pack(padx=10, pady=5, side='left')

    def _save_action(self):
        if self.return_action:
            self.selected_action = 'save'
            self.destroy()
        else:
            self.instance.controller.save_project_file()
            self.destroy()
            self.instance.destroy()

    def _dont_save_action(self):
        if self.return_action:
            self.selected_action = 'dont_save'
            self.destroy()
        else:
            self.destroy()
            self.instance.destroy()

    def _cancel_action(self):
        if self.return_action:
            self.selected_action = 'cancel'
            self.destroy()
        else:
            self.destroy()

    def show_dialog(self):
        self.wait_window()  # Wait for the dialog to be closed
        return self.selected_action


class ToolTip(object):
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None
        self.id = None
        self.x = self.y = 0
        self.enabled = True

    def showtip(self, text):
        "Display text in tooltip window"
        self.text = text
        if not self.enabled or self.tipwindow or not self.text:
            return
        x, y, cx, cy = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 0
        y = y + cy + self.widget.winfo_rooty() + 5
        self.tipwindow = tw = tkinter.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry("+%d+%d" % (x, y))
        label = tkinter.Label(tw, text=self.text, justify=tkinter.LEFT,
                              background="#e0e0e0", relief=tkinter.FLAT, borderwidth=0,
                              font=("tahoma", "8", "normal"))
        label.pack(ipadx=2, ipady=2)

    def hidetip(self):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False



def CreateToolTip(widget, text):
    toolTip = ToolTip(widget)
    def enter(event):
        toolTip.showtip(text)
    def leave(event):
        toolTip.hidetip()
    widget.bind('<Enter>', enter)
    widget.bind('<Leave>', leave)
    return toolTip


def string_to_list(index_string, no_str_result = []):
    index_string = index_string.replace('[', '').replace(']', '').replace(' ', '')
    if not index_string:
        return no_str_result
    
    result_list = []
    for item in index_string.split(','):
        if item:
            if isfloat(item):
                result_list.append(float(item))
            else:
                raise TerminatingError(f"Invalid literal for float: '{item}'")
    
    if len(result_list) == 0:
        return no_str_result
    return result_list


def isfloat(num):
    '''Check if a string represents a float number.'''
    try:
        float(num)
        return True
    except ValueError:
        return False

def isposint(num):
    '''Check if a string represents a float number.'''
    try:
        if int(num) >= 0:
            return True
        else:
            return False
    except ValueError:
        return False

class TerminatingError(Exception):
    '''
    Exception raised for terminating errors in the application, displaying an error message using Tkinter.

    Source: https://stackoverflow.com/questions/7957436/error-exception-must-derive-from-baseexception-even-when-it-does-python-2-7
    '''
    def __init__(self, message):
        self.message = 'ERROR: ' + message
        tkinter.messagebox.showerror('Terminating Error', self.message)

    def __str__(self):
        return self.message
