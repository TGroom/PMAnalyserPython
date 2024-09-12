import itertools
import os

import numpy as np
import pandas as pd
from scipy import optimize
from scipy import signal
from scipy.linalg import svd


class SVD:
    def __init__(self):
        self.components = None
        self.eigenvalues = None


    def fit(self, data_pca):
        """
        Fit the model to the data.

        Parameters:
        data_pca (np.ndarray): The input data matrix for SVD.
        """
        # Scale by square root of (n-1)
        Y = data_pca / np.sqrt(data_pca.shape[0] - 1)

        # Perform SVD
        U, S, Vt = svd(Y, full_matrices=False)

        # Enforce deterministic output
        # Source: https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/utils/extmath.py#L848
        max_abs_v_rows = np.argmax(np.abs(Vt), axis=1)
        shift = np.arange(Vt.shape[0])
        indices = max_abs_v_rows + shift * Vt.shape[1]
        signs = np.sign(np.take(np.reshape(Vt, (-1,)), indices, axis=0))
        if U is not None:
            U *= signs[np.newaxis, :]
        Vt *= signs[:, np.newaxis]

        self.components = Vt

        # Calculate eigenvalues
        self.eigenvalues = S ** 2  # (S ** 2) / (data.shape[0] - 1) for explained variance ratio
    

    def project(self, data_pca):

        return np.dot(data_pca, self.components.T)


class PCA_Model(SVD):
    def __init__(self, sample_freq):
        super().__init__()

        self.sample_freq = sample_freq

        self.d_norm = 1.0
        self.weight_vector = np.ones(1000)  #self.df.shape[1]
        self.orientation_original = np.zeros(1)
        self.orientation_drift = np.zeros(1)
        self.orientation_resultant = np.zeros(1)

        self.results = pd.DataFrame()


    def remove_markers(self, df, idxs_to_remove):
        '''
            Remove specified markers from the DataFrame.

            Args:
                df (pd.DataFrame): Input data.
                idxs_to_remove (list of int): Indices of markers to remove.
        '''
        # Calculate the columns to delete based on marker indices
        cols_to_del = [marker_index * 3 + i for marker_index in idxs_to_remove for i in range(3)]
        # Drop the columns from the DataFrame
        df.drop(df.columns[cols_to_del], axis=1, inplace=True)


    def remove_rows(self, df, idxs_to_remove):
        '''
            Remove specified rows from the DataFrame.

            Args:
                df (pd.DataFrame): Input data.
                idxs_to_remove (list of int): Indices of rows to remove.
        '''
        # Drop the rows from the DataFrame
        df.drop(idxs_to_remove, axis=0, inplace=True)
        # Reset the index of the DataFrame
        df.reset_index(drop=True, inplace=True)


    def fill_gaps(self, df, fill_method='Off'):
        '''
        Fill the gaps in the provided data based on the selected method.

        Args:
            df (pd.DataFrame): Input data.
            fill_method (str): Chosen interpolation method, one of ['Off', 'Forward Fill', 'Backward Fill', 'Linear', 'Quadratic', 'Cubic', '1st Order Spline', '2nd Order Spline']
        '''

        if fill_method == 'Forward Fill':
            df.ffill(axis=0, inplace=True)

        elif fill_method == 'Backward Fill':
            df.bfill(axis=0, inplace=True)

        elif fill_method == 'Linear':
            df.interpolate(method='linear', axis=0, order=0, inplace=True)

        elif fill_method == 'Quadratic':
            df.interpolate(method='quadratic', axis=0, order=0, inplace=True)
            df.ffill(axis=0, inplace=True)
            df.bfill(axis=0, inplace=True)
        
        elif fill_method == 'cubic':
            df.interpolate(method='cubic', axis=0, order=0, inplace=True)
            df.ffill(axis=0, inplace=True)
            df.bfill(axis=0, inplace=True)

        elif fill_method == '1st Order Spline':
            df.interpolate(method='spline', axis=0, order=1, inplace=True)

        elif fill_method == '2nd Order Spline':
            df.interpolate(method='spline', axis=0, order=2, inplace=True)

        #  Define custom fill functions here
        #  elif fill_method == '...':


    def flip_axes(self, df, axes_to_flip=[False, False, False]):
        '''
            Flip specified axes in the provided data.

            Args:
                df (pd.DataFrame): Input data.
                axes_to_flip (list of bool): List indicating which axes to flip.
        '''
        for axis_index, should_flip in enumerate(axes_to_flip):
            if should_flip:
                df.iloc[:, axis_index::3] *= -1

    
    def filter(self, df, filter_method='Off', order=1, cutoff=10):
        '''
        Apply selected filtering method to the input data

        Args:
            df (pd.DataFrame): Input data.
            method (str): The method to use between ['Off', 'Butterworth']
            order (int): The order of the butterworth
            cutoff (float): The cutoff frequency of the butterworth

        Source:
            https://dsp.stackexchange.com/questions/49460/apply-low-pass-butterworth-filter-in-python
        '''
        # Butterworth Filtering
        if filter_method == 'Butterworth':
            for column_name, column in df.items():

                # Check for valid inputs
                if order < 1:
                    raise Exception('Butterworth filter order must be a non-negative non-zero integer')
                
                if cutoff <= 0.0:
                    raise Exception('Butterworth filter cutoff must be a non-negative non-zero integer')
                
                try:
                    # Butterworth filter parameters
                    b, a = signal.butter(order, cutoff, btype='low', analog=False, output='ba', fs=self.sample_freq)
                    # Apply filter
                    column = signal.filtfilt(b, a, column)

                except ValueError as ve:
                    raise Exception(f'ValueError occurred during Butterworth filtering: {ve}')
                except Exception as e:
                    raise Exception(f'An unexpected error occurred during Butterworth filtering: {e}')

                # Check for NaNs in filtered column
                if np.isnan(column).any():
                    # If NaNs found, raise an error
                    raise Exception(f'Unable to perform butterworth filtering on column {column_name} '
                                        f'as it contains NaNs. Try using a gap filling method.')
                # Update original DataFrame with filtered column
                df.loc[:, column_name] = column

        #  Define custom filtering functions here
        #  elif filter_method == '...':


    def _apply_weighted_mean_subtraction(self, df, weights):
        '''
        Apply weighted mean subtraction to data.

        Args:
            df (pd.DataFrame): The data to be centered.
            weights (numpy.array): The weights for centering.
        '''
        if not sum(weights) > 0:
            raise Exception('Sum of weights used for centring (weights column or centre ref. column) is <= 0.')
        total_weights_per_xyz = sum(weights) / 3
        num_of_markers = df.shape[1] // 3
        for index, row in df.iterrows():
            weighted_sum = ((row.to_numpy(dtype=float) * weights).reshape(-1, 3)).sum(axis=0)  # Calculate the weighted sum of every three elements in the row
            weighted_mean_arr = np.tile(weighted_sum / total_weights_per_xyz, num_of_markers)  # Calculate the weighted mean array and repeat it to match the original length of the row
            df.iloc[index] -= weighted_mean_arr  # Subtract the weighted mean array from the original row values


    def centre(self, df, centre_method='Off', weights=[], centre_refs=[]):
        '''
        Center the data based on different options.

        Args:
            df (pd.DataFrame): The data to be centered.
        '''
        if centre_method == 'Mean Marker Position':
            ones = [1.0] * df.shape[1]
            self._apply_weighted_mean_subtraction(df, ones)

        elif centre_method == 'Mean Marker Pos. (Weights)':
            self._apply_weighted_mean_subtraction(df, np.repeat(weights, 3))

        elif centre_method == 'Mean Marker Pos. (Centre Ref.)':
            self._apply_weighted_mean_subtraction(df, np.repeat(centre_refs, 3))

        #  Define custom centring functions here
        #  elif centre_method == '...':


    def align_orientation(self, df, align_orientation_method='Off', centre_refs:np.ndarray=[], cutoff=1.0):

        def _rotate_dataframe(df, heading):
            '''
            Rotate each set of 3D points in a DataFrame about the Z unit vector based on a 2D (x, y) unit vector heading.
            Heading [1, 0] implies no rotation.
            '''
            # Convert DataFrame to numpy array
            points = df.to_numpy(dtype=float).reshape(-1, len(df.columns) // 3, 3)
            
            # Ensure heading is a numpy array
            heading = np.asarray(heading)
            
            # Check if heading is a single point [x, y]
            if heading.ndim == 1:
                # Single point case: Convert to array of shape (2, 1)
                heading = heading[:, np.newaxis]
            
            # Determine rotation angles around Z axis for each point
            angles_rad = np.arctan2(heading[1], heading[0])
            
            # Compute rotation matrices around Z axis for each point
            rotation_matrices = np.stack([
                np.cos(angles_rad), -np.sin(angles_rad), np.zeros_like(angles_rad),
                np.sin(angles_rad), np.cos(angles_rad), np.zeros_like(angles_rad),
                np.zeros_like(angles_rad), np.zeros_like(angles_rad), np.ones_like(angles_rad)
            ], axis=-1).reshape((-1, 3, 3))
            
            # Apply rotation to each set of points inplace
            rotated_points = np.einsum('ijk,ikl->ijl', points, rotation_matrices)

            # Flatten back to original DataFrame shape and update df inplace
            df[:] = rotated_points.reshape(-1, len(df.columns))

        def _calc_unit_vectors(df):
            points = df.to_numpy(dtype=float)
            points = points.reshape(points.shape[0], -1, 3)
            points = points.transpose((1, 0, 2))
            largest_indices = centre_refs.argsort()[-2:]

            # Vector from point1 to point2
            vector = np.array(points[largest_indices[1]].T) - np.array(points[largest_indices[0]].T)
            unit_vector = vector / np.linalg.norm(vector)
            return unit_vector

        def _vector_to_angle(vectors):
            # Separate the points into x and y components
            x_points = vectors[0, :]
            y_points = vectors[1, :]

            prev_theta = np.zeros(len(x_points))

            for i in range(len(x_points)):
                x = x_points[i]
                y = y_points[i]
                
                r = np.sqrt(x**2 + y**2)
                theta = np.arctan2(y, x)
                
                # Adjust phi for continuity
                theta_adjusted = theta + np.round((prev_theta[i - 1] - theta) / (2 * np.pi)) * (2 * np.pi)
                prev_theta[i] = theta_adjusted
            
            return prev_theta

        if 'Soft Align Z' in align_orientation_method:
            if not sum(centre_refs) > 0:
                raise Exception('Sum of centre ref. column used for orientation alignment to axes is <= 0.')

            def _calc_normal_vectors(df):
                unit_vector = _calc_unit_vectors(df)
                nx = -unit_vector[1]
                ny = unit_vector[0]
                normal_vectors = np.array([nx, ny])
                normal_vectors /= np.linalg.norm(normal_vectors)
                return normal_vectors

            original_normal_vectors = _calc_normal_vectors(df)

            if cutoff > 0.0:
                filter_order = 5
                b, a = signal.butter(filter_order, cutoff, btype='low', fs=self.sample_freq)
                normal_vector_drift = signal.filtfilt(b, a, original_normal_vectors)
            else:
                # Find the mean
                normal_vector_drift = np.tile(np.mean(original_normal_vectors, axis=1, keepdims=True), (1, original_normal_vectors.shape[1]))

            if '(X-Axis)' in align_orientation_method:
                _rotate_dataframe(df, np.array([normal_vector_drift[1], -normal_vector_drift[0]]))  # 90 deg rotation
            else:
                _rotate_dataframe(df, normal_vector_drift)

            self.orientation_original = _vector_to_angle(original_normal_vectors) + np.pi/2
            self.orientation_drift = _vector_to_angle(normal_vector_drift) + np.pi/2
            self.orientation_resultant = _vector_to_angle(_calc_normal_vectors(df))
        

        if 'Body-Fixed' in align_orientation_method:
            if not sum(centre_refs) > 0:
                raise Exception('Sum of centre ref. column used for orientation alignment to axes is <= 0.')

            def body_fix_coordinate_system(points, plane_points):
                # Extract the three points defining the plane
                p1, p2, p3 = plane_points
                
                # Define the new x-axis as the vector from p1 to p2
                new_x = p2 - p1
                new_x /= np.linalg.norm(new_x)  # Normalize
                
                # Define the new y-axis as the vector from p1 to p3, projected to be orthogonal to new_x
                new_y = p3 - p1
                new_y -= new_x * np.dot(new_y, new_x)
                new_y /= np.linalg.norm(new_y)  # Normalize
                new_z = np.cross(new_x, new_y)  # Define the new z-axis as the cross product of new_x and new_y (normal to the plane)
                
                transformation_matrix = np.vstack([new_x, new_y, new_z])  # Create the transformation matrix
                translated_points = points - p1  # Subtract the origin point (p1) from all points to translate them to the new origin
                transformed_points = translated_points @ transformation_matrix.T  # Apply the transformation matrix to the translated points
                
                return transformed_points
            
            points = df.to_numpy(dtype=float)
            points = points.reshape(points.shape[0], -1, 3)
            largest_indices = centre_refs.argsort()[-3:]
            plane_points_list = points[:, largest_indices[0:3]]

            points_list = df.to_numpy(dtype=float).reshape(df.shape[0], -1, 3)

            if cutoff > 0.0:
                filter_order = 5
                b, a = signal.butter(filter_order, cutoff, btype='low', fs=self.sample_freq)
                plane_points_list = signal.filtfilt(b, a, plane_points_list, axis=0)

            transformed_points_list = []
            for points, plane_points in zip(points_list, plane_points_list):
                transformed_points = body_fix_coordinate_system(points, plane_points)
                transformed_points_list.append(transformed_points)
            
            data_fixed = np.asarray(transformed_points_list).reshape(-1, len(df.columns))

            if not cutoff > 0.0:
                # Add a small offset to the first row so that these features have some variance
                data_fixed[0, largest_indices[0]*3:largest_indices[0]*3+3] += 0.00001
                data_fixed[0, largest_indices[1]*3:largest_indices[1]*3+3] += 0.00001
                data_fixed[0, largest_indices[2]*3:largest_indices[2]*3+3] += 0.00001

            df[:] = data_fixed

        #  Define custom alignment functions here
        #  elif align_orientation_method == '...':


    def _centre_on_feature_mean(self, df):
        ''' Center each feature to its mean '''
        df -= np.mean(df, axis=0)


    def _normalise(self, df, method='Off', centre_refs=[]):
        '''
        Normalise the given data based on different options.
        '''

        data_np = df.to_numpy(dtype=float)
        self.d_norm = 1.0

        if method == 'MED (Mean Euclidean Distance)':  # Normalize data to mean euclidean distance MED
            self.d_norm = np.mean(np.sqrt(np.sum(df**2, axis=1)))  

        elif method == 'Mean Dist. 2 Markers (Centre Ref.)':  # Mean between 2 markers
            if not sum(centre_refs) > 0:
                raise Exception('Sum of centre ref. column used for normalisation is <= 0.')
            largest_indices = centre_refs.argsort()[-2:]
            marker_a = largest_indices[1]
            marker_b = largest_indices[0]

            self.d_norm = np.mean(np.linalg.norm(data_np[:, marker_a*3:marker_a*3+3] - data_np[:, marker_b*3:marker_b*3+3], axis=1))  # Computes the mean Euclidean distance
            #self.d_norm = np.mean(data_np[:, (marker_a*3 + 2)] - data_np[:, (marker_b*3 + 2)])  # Computes the mean height only (z axis)

        elif method == 'Maximum Range (1st Coords)':  # Normalize to maximum range of 1st coordinate (usually x)
            self.d_norm = np.max(np.max(data_np[:, 0::3], axis=1) - np.min(data_np[:, 0::3], axis=1))

        elif method == 'Maximum Range (2nd Coords)':  # Normalize to maximum range of 2nd coordinate (usually y)
            self.d_norm = np.max(np.max(data_np[:, 1::3], axis=1) - np.min(data_np[:, 1::3], axis=1))

        elif method == 'Maximum Range (3rd Coords)':  # Normalize to maximum range of 3rd coordinate (usually z)
            self.d_norm = np.max(np.max(data_np[:, 2::3], axis=1) - np.min(data_np[:, 2::3], axis=1))
        
        df /= self.d_norm


    def _assign_weight(self, df, method='Off', weights=[]):
        '''
            Assign weights to the columns of the DataFrame based on the specified method.

            Args:
                df (pd.DataFrame): Input data.
                method (str): Method to assign weights, one of ['Off', 'Manual Weight Vector'].
                            'Off' means no weights are applied.
                            'Manual Weight Vector' applies the provided weights.
                weights (list of float): List of weights to be applied if method is 'Manual Weight Vector'.
        '''

        self.weight_vector = np.ones(df.shape[1])
        weights = np.repeat(weights, 3)

        if method == 'Manual Weight Vector':
            if np.sum(weights) <= 0 or len(weights) != len(self.weight_vector):
                raise Exception('Invalid weights')
            self.weight_vector = weights
    
        df *= self.weight_vector


    def coordinate_transformation(self, df, coord_transform_method='Off'):
        '''
            Transform the coordinates of the input data frame from Cartesian to another coordinate system.

            Args:
                df_cartesian (pd.DataFrame): Input data in Cartesian coordinates.
                coord_transform_method (str): Method for coordinate transformation. Supported methods:
                    - 'Off': No transformation.
                    - 'Cartesian -> Spherical (3D)': Transform Cartesian coordinates to Spherical coordinates in 3D.
        '''

        if coord_transform_method == 'Cartesian -> Spherical (3D)':
            num_columns = len(df.columns)
            num_points = num_columns // 3  # Each point has x, y, z columns
            
            cartesian_values = df.to_numpy(dtype=float)
            cartesian_values = cartesian_values.reshape(-1, num_points, 3)
            
            prev_theta = np.zeros(num_points)
            prev_phi = np.zeros(num_points)
            
            for i in range(cartesian_values.shape[0]):
                x = cartesian_values[i, :, 0]
                y = cartesian_values[i, :, 1]
                z = cartesian_values[i, :, 2]
                
                r = np.sqrt(x**2 + y**2 + z**2)
                theta = np.arcsin(z / r)
                phi = np.arctan2(y, x)
                
                theta_adjusted = theta + np.round((prev_theta - theta) / (2 * np.pi)) * (2 * np.pi)
                phi_adjusted = phi + np.round((prev_phi - phi) / (2 * np.pi)) * (2 * np.pi)
                
                prev_theta = theta_adjusted
                prev_phi = phi_adjusted
                
                cartesian_values[i, :, 0] = phi_adjusted
                cartesian_values[i, :, 1] = theta_adjusted
                cartesian_values[i, :, 2] = r
            
            df.iloc[:, :] = cartesian_values.reshape(-1, num_columns)

        #  Define custom coordinate transformation functions here
        #  elif coord_transform_method == '...':

    def reverse_coordinate_transformation(self, df, coord_transform_method='Off'):
        if coord_transform_method == 'Cartesian -> Spherical (3D)':
            num_columns = len(df.columns)
            num_points = num_columns // 3  # Each point has r, theta, phi columns
            
            spherical_values = df.to_numpy(dtype=float)
            spherical_values = spherical_values.reshape(-1, num_points, 3)
            
            x = spherical_values[:, :, 2] * np.cos(spherical_values[:, :, 1]) * np.cos(spherical_values[:, :, 0])
            y = spherical_values[:, :, 2] * np.cos(spherical_values[:, :, 1]) * np.sin(spherical_values[:, :, 0])
            z = spherical_values[:, :, 2] * np.sin(spherical_values[:, :, 1])
            
            cartesian_values = np.stack((x, y, z), axis=-1).reshape(-1, num_columns)
            df.iloc[:, :] = cartesian_values

        #  Define custom reverse coordinate transformation functions here
        #  elif coord_transform_method == '...':


    def reconstruct_data(self, p_0, c, p, coord_transform_method='Off'):
        '''
        Reconstruct data post PCA.

        Args:
            p_0: Average posture.
            c: principal component scores
            p: Eigenpostures / Loadings.
            coord_transform_method: Method of coordinate transformation to reverse.
            
        Returns:
            p: Posture
        '''

        p = np.swapaxes(p, 0, 1)
        posture = pd.DataFrame(c @ p)
        posture *= self.d_norm # Reverse normalisation of the data
        self.weight_vector = self.weight_vector[:posture.shape[1]]
        weight_vector_inv = np.where(self.weight_vector == 0.0, 0.0, 1.0 / self.weight_vector)  # Prevent division by zero by replacing zero weights with zero
        posture *= weight_vector_inv
        posture += p_0  # Reverse centring of each feature
        self.reverse_coordinate_transformation(posture, coord_transform_method)  # Reverse coordinate transformation is applicable
        return posture.to_numpy(dtype=float)
    

    def check_data_format(self, df, file_name):
        '''
        Check the format and integrity of the provided data.

        Args:
            df (pd.DataFrame): The input data to be checked.
            file_name (str): The name of the data file for reference in error messages.

        Raises:
            Exception: If the input data is not a pandas DataFrame.
            Exception: If there are columns with only NaN values.
            Exception: If there are rows with only NaN values.
            Exception: If there are any NaN values in the data.
            Exception: If there are infinite values in the data.
            Exception: If there are columns with constant values (no variance).
            Exception: If the data contains non-numeric values.
            Exception: If there are duplicated columns.
            Exception: If there are duplicated rows.
            Exception: If the data has fewer than 5 rows.
            Exception: If there are columns with non-float values.
        '''

        # Check if data is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise Exception(f'Data set "{file_name}" is not a pandas DataFrame.')

        # Check for columns with only NaN values
        nan_columns = (np.sum(np.isnan(df), axis=0) == df.shape[0])
        num_nan_columns = np.sum(nan_columns)
        if num_nan_columns != 0:
            nan_columns_indices = np.where(nan_columns)[0] + 1
            raise Exception(
                f'Data set "{file_name}" has {num_nan_columns} columns with only NaNs (Columns {nan_columns_indices}). '
            )

        # Check for rows with only NaN values
        nan_rows = (np.sum(np.isnan(df), axis=1) == df.shape[1])
        num_nan_rows = np.sum(nan_rows)
        if num_nan_rows != 0:
            nan_rows_indices = np.where(nan_rows)[0] + 1
            raise Exception(
                f'Data set "{file_name}" has {num_nan_rows} rows with only NaNs (Rows {nan_rows_indices})!'
            )

        # Check for any NaN values in the data
        cols_with_nan = df.isna().any(axis=0)
        rows_with_nan = df.isna().any(axis=1)
        cols_with_nan_indices = cols_with_nan[cols_with_nan].index.tolist()
        rows_with_nan_indices = rows_with_nan[rows_with_nan].index.tolist()
        if cols_with_nan_indices or rows_with_nan_indices:
            raise Exception(
                f'Data set "{file_name}" has NaN values in columns {cols_with_nan_indices} and rows {rows_with_nan_indices}. '
                'Try using gap filling methods.'
            )

        # Check for infinite values in the data
        if np.isinf(df.to_numpy(dtype=float)).any():
            inf_columns_indices = np.where(np.isinf(df.to_numpy(dtype=float)).any(axis=0))[0] + 1
            inf_rows_indices = np.where(np.isinf(df.to_numpy(dtype=float)).any(axis=1))[0] + 1
            raise Exception(
                f'Data set "{file_name}" has infinite values in columns {inf_columns_indices} and rows {inf_rows_indices}. '
                'Ensure all data is finite before proceeding.'
            )

        # Check for constant columns (no variance)
        constant_columns = df.nunique() <= 1
        constant_columns_indices = constant_columns[constant_columns].index.tolist()
        if constant_columns_indices:
            raise Exception(
                f'Data set "{file_name}" has columns with no variance (constant values): {constant_columns_indices}. '
                'These columns should be removed or handled before PCA.'
            )

        # Check for non-numeric data
        if not np.issubdtype(df.to_numpy(dtype=float).dtype, np.number):
            raise Exception(
                f'Data set "{file_name}" contains non-numeric values. Ensure all data entries are numeric before filtering, '
                'normalising, and performing PCA.'
            )
        
        # Check for duplicated columns
        duplicated_columns = df.columns[df.columns.duplicated()].tolist()
        if duplicated_columns:
            raise Exception(
                f'Data set "{file_name}" contains duplicated columns: {duplicated_columns}. '
                'Ensure all columns have unique names before proceeding.'
            )

        # Check for duplicated rows
        duplicated_rows = df.index[df.index.duplicated()].tolist()
        if duplicated_rows:
            raise Exception(
                f'Data set "{file_name}" contains duplicated rows: {duplicated_rows}. '
                'Ensure all rows have unique indices before proceeding.'
            )
        
        # Check if the data has only very few rows
        if df.shape[0] < 5:
            raise Exception(
                f'Data set "{file_name}" has less than 5 rows of data. Hint: Check delimiter is set correctly and row deletion is set correctly.'
            )

        # Check for non-float columns
        non_float_cols = [col for col in df.columns if not pd.api.types.is_float_dtype(df[col])]
        if non_float_cols:
            raise Exception(
                f'Data set "{file_name}" contains the following columns with non-float values: {non_float_cols}.'
            )
    

    def norm_weight_centre(self, df, cent_norm_weigh_order, weight_method='Off', weights=[], normalisation_method='Off', centre_refs=[]):
        '''
            Apply centring, weighting, and normalisation to the provided data frame in a specified order.

            Args:
                df (pd.DataFrame): The input data frame to process.
                cent_norm_weigh_order (int): The order of operations to apply:
                    [0: Centre -> weight -> normalise]
                    [1: Centre -> normalise -> weight]
                    [2: Weight -> centre -> normalise]
                    [3: Weight -> normalise -> centre]
                    [4: Normalise -> centre -> weight]
                    [5: Normalise -> weight -> centre]
                weight_method (str, optional): The method used for weighting. Defaults to 'Off'.
                weights (list, optional): The weights to apply if weighting is enabled. Defaults to an empty list.
                normalisation_method (str, optional): The method used for normalisation. Defaults to 'Off'.
                centre_refs (list, optional): Reference values for centring. Defaults to an empty list.
        '''

        self.mean_of_data = np.mean(df, axis=0)  # Taking the mean here allows us not to deal with the order when reversing the operations post PCA

        match cent_norm_weigh_order:
            case 0:
                self._centre_on_feature_mean(df)
                self._assign_weight(df, weight_method, weights)
                self._normalise(df, normalisation_method, centre_refs)

            case 1:
                self._centre_on_feature_mean(df)
                self._normalise(normalisation_method, centre_refs)
                self._assign_weight(weight_method, weights)

            case 2:
                self._assign_weight(df, weight_method, weights)
                self._centre_on_feature_mean(df)
                self._normalise(df, normalisation_method, centre_refs)

            case 3:
                self._assign_weight(df, weight_method, weights)
                self._normalise(df, normalisation_method, centre_refs)
                self._centre_on_feature_mean(df)
            
            case 4:
                self._normalise(df, normalisation_method, centre_refs)
                self._centre_on_feature_mean(df)
                self._assign_weight(df, weight_method, weights)

            case 5:
                self._normalise(df, normalisation_method, centre_refs)
                self._assign_weight(df, weight_method, weights)
                self._centre_on_feature_mean(df)
    

    def calculate_explained_variance(self, pp_time_series, eigenvalues):
        '''
            Calculate the explained variance and related metrics from the provided preprocessed time series and eigenvalues.

            Args:
                pp_time_series (pd.DataFrame): Preprocessed time series data where rows represent observations and columns represent variables.
                eigenvalues (array-like): Eigenvalues obtained from PCA.

            Returns:
                tuple: A tuple containing:
                    - eigenvalues (pd.DataFrame): DataFrame of eigenvalues with original columns.
                    - eigenvalue_rvar (pd.DataFrame): DataFrame of relative variances for the eigenvalues.
                    - PP_rVAR (pd.DataFrame): DataFrame of relative variances for the preprocessed time series.
                    - PP_rSTD (pd.DataFrame): DataFrame of relative standard deviations for the preprocessed time series.
                    - PP_CUM_rVAR (pd.DataFrame): DataFrame of cumulative relative variances for the preprocessed time series.
                    - PP_CUM_rSTD (pd.DataFrame): DataFrame of cumulative relative standard deviations for the preprocessed time series.
            '''
        PP_STD = pd.DataFrame(np.std(pp_time_series, axis=0), index=pp_time_series.columns).T
        PP_VAR = PP_STD ** 2

        # Compute relative standard deviations and variances
        PP_rSTD = PP_STD / PP_STD.sum().sum()
        PP_rVAR = PP_VAR / PP_VAR.sum().sum()

        # Calculate cumulative relative standard deviations and variances
        PP_CUM_rSTD = PP_rSTD.cumsum(1)
        PP_CUM_rVAR = PP_rVAR.cumsum(1)

        eigenvalues = pd.DataFrame(eigenvalues, index=pp_time_series.columns).T
        eigenvalue_rvar = eigenvalues / eigenvalues.sum().sum()

        return eigenvalues, eigenvalue_rvar, PP_rVAR, PP_rSTD, PP_CUM_rVAR, PP_CUM_rSTD


    def postprocess_pca_results(self, columns, transformed_data, components, eigenvalues, pp_filter, pv_filter, pa_filter, order, cutoff, freq_harmonics, **kwargs):
        # Generate column headings for Principal Components
        pc_column_headings = [f'PC{i+1}' for i in range(components.shape[1])]

        pp_time_series = pd.DataFrame(data=transformed_data, columns=pc_column_headings[:transformed_data.shape[1]])

        def _apply_filter(data, filter_type, order, cutoff):
            if filter_type == 'Low Pass Butterworth':
                self.filter(data, 'Butterworth', order, cutoff, self.sample_freq)

        def _diff_central(input_data):
            central_diff = input_data.iloc[2:].reset_index(drop=True) - input_data.iloc[:-2].reset_index(drop=True)
            return central_diff.mul(self.sample_freq / 2)  # Multiply by sampling frequency

        # Apply filters to time series data based on user options
        _apply_filter(pp_time_series, pp_filter, order, cutoff)
        pv_time_series = _diff_central(pp_time_series)
        _apply_filter(pv_time_series, pv_filter, order, cutoff)
        pa_time_series = _diff_central(pv_time_series)
        _apply_filter(pa_time_series, pa_filter, order, cutoff)

        # Initialize time series metrics calculator
        scores_metrics = TimeSeriesMetrics(self.sample_freq)
        freq_harmonics = np.pad(freq_harmonics, (0, pp_time_series.shape[1] - len(freq_harmonics)), mode='constant', constant_values=0.0)
        set_freqs = np.zeros_like(freq_harmonics)
        # Initialize DataFrames for sine approximation and PCA metrics
        sin_approx_df = pd.DataFrame(index=['Sin Approx. Amplitude', 'Sin Approx. Omega', 'Sin Approx. Phase', 'Sin Approx. Frequency'])
        pm_metrics_dfs = [pd.DataFrame() for _ in range(3)]
        
        # Calculate metrics for each time series
        for i, scores in enumerate([pp_time_series, pv_time_series, pa_time_series]):
            pm_metrics_temp = {}
            for k in range(pp_time_series.shape[1]):
                pk_scores = scores.iloc[:, k].to_numpy(dtype=float)
                pm_metrics_temp[f'PC{k+1}'] = scores_metrics.get_all(pk_scores)
                
                # Only fit sin curves to pp_time_series (i == 0)
                if i == 0 and k < 8:
                    sin_approx_df[f'PC{k+1}'] = scores_metrics.fit_sin(pk_scores, set_freqs[k])

                if k == 0 and freq_harmonics[0] > 0:
                    set_freqs = (freq_harmonics / freq_harmonics[0]) * sin_approx_df[f'PC1'].loc['Sin Approx. Frequency']

            pm_metrics_dfs[i] = pd.DataFrame(pm_metrics_temp, index=[
                'mean [AU]', 'std [AU]', 'meanPos [AU]', 'meanNeg [AU]', 'stdPos [AU]', 'stdNeg [AU]',
                'NoZC [#]', 'NoPeaks [#]', 'meanTbZC [s]', 'meanTbPeaks [s]', 'stdTbZC [s]', 'stdTbPeaks [s]',
                'ratioZC/Peaks [AU]', 'averagePower [AU]', 'RMS [AU]', 'averageActivity [AU]'
            ])

        # Calculate explained variance metrics
        eigenvalues, eigenvalue_rvar, PP_rVAR, PP_rSTD, PP_CUM_rVAR, PP_CUM_rSTD = self.calculate_explained_variance(pp_time_series, eigenvalues)
        
        # Generate DataFrames for PCA loadings and input data mean
        loadings = pd.DataFrame(components.T, index=columns, columns=pc_column_headings[:transformed_data.shape[1]])

        kwargs['Data Mean'] = pd.DataFrame(np.array(self.mean_of_data).reshape(1, -1), columns=pc_column_headings)
        kwargs['Total Var (Eigenvalues)'] = eigenvalues 
        kwargs['Overall Explained VAR Ratio'] = eigenvalue_rvar
        kwargs['Explained VAR Ratio'] = PP_rVAR
        kwargs['Explained STD Ratio'] = PP_rSTD
        kwargs['Explained Cumulative VAR'] = PP_CUM_rVAR
        kwargs['Explained Cumulative STD'] = PP_CUM_rSTD
        kwargs['Normalisation Factor'] = pd.DataFrame([self.d_norm], columns=['PC1'])
        kwargs['Weight Vector'] = pd.DataFrame([self.weight_vector], columns=pc_column_headings)

        # Prepare results for saving
        concat_list = (
            [value for key, value in kwargs.items()] +
            [
                loadings, sin_approx_df, pm_metrics_dfs[0], pp_time_series,
                pm_metrics_dfs[1], pv_time_series, pm_metrics_dfs[2], pa_time_series
            ]
        )
        
        concat_keys = (
            [key.replace('_', ' ') for key, value in kwargs.items()] + 
            [
                'Loadings', 'PP Metrics', 'PP Metrics', 'PP Time Series (Position)',
                'PV Metrics', 'PV Time Series (Velocity)', 'PA Metrics',
                'PA Time Series (Acceleration)'
            ]
        )

        self.results = pd.concat(concat_list, keys=concat_keys, axis=0).rename_axis(['level_0', 'level_1'])

        return self.results


def loocv(X, max_pcs=25, subset_size=50):
    '''
        Perform Leave-One-Out Cross-Validation.
        The larger the subset_size, the greater the approximation reliability.
        
        Source:
            https://stats.stackexchange.com/questions/93845/how-to-perform-cross-validation-for-pca-to-determine-the-number-of-principal-com
    '''
    n_samples, n_features = X.shape
    max_pcs = min(n_features, max_pcs)
    subset_size = min(n_samples, subset_size)
    
    # Randomly select a subset of samples
    subset_indices = np.random.choice(n_samples, subset_size, replace=False)
    X_subset = X[subset_indices, :]

    error_naive = np.zeros((subset_size, max_pcs))
    error_approx = np.zeros((subset_size, max_pcs))

    for i, sample_idx in enumerate(subset_indices):
        # Leave-one-out: create training data by excluding the ith sample
        X_train = np.delete(X_subset, i, axis=0)
        _, _, Vt = svd(X_train, full_matrices=False)
        V = Vt.T
        X_test = X_subset[i, :]

        for j in range(1, max_pcs + 1):
            P = V[:, :j] @ V[:, :j].T
            err_naive = X_test @ (np.eye(n_features) - P)
            err_approx = X_test @ (np.eye(n_features) - P + np.diag(np.diag(P)))

            error_naive[i, j-1] = np.sum(err_naive**2)
            error_approx[i, j-1] = np.sum(err_approx**2)

    PRESS_naive = np.sum(error_naive, axis=0)
    PRESS_approx = np.log(np.sum(error_approx, axis=0))
    
    index_labels = [f'PC{i+1}' for i in range(PRESS_naive.shape[0])]
    PRESS_naive = pd.DataFrame([PRESS_naive], columns=index_labels)
    PRESS_approx = pd.DataFrame([PRESS_approx], columns=index_labels)

    return PRESS_naive, PRESS_approx


class TimeSeriesMetrics:
    def __init__(self, sample_freq):
        self.sample_freq = sample_freq

    def get_all(self, pk_scores):
        metrics_funcs = [
            self.mean, self.std, self.mean_pos, self.mean_neg, self.std_pos,
            self.std_neg, self.no_zc, self.no_p, self.mean_tb_zc,
            self.mean_tb_p, self.std_tb_zc, self.std_tb_p, self.ratio_zcp,
            self.average_power, self.rms, self.average_activity
        ]

        return [func(pk_scores) for func in metrics_funcs]

    def mean(self, time_series):
        '''Calculate the mean of the time series.'''
        return np.mean(time_series)

    def std(self, time_series):
        '''Calculate the standard deviation of the time series.'''
        return np.std(time_series, axis=0)

    def mean_pos(self, time_series):
        '''Calculate the mean of positive values in the time series.'''
        positives = time_series[time_series >= 0]
        return np.mean(positives) if len(positives) > 0 else np.nan

    def mean_neg(self, time_series):
        '''Calculate the mean of negative values in the time series.'''
        negatives = time_series[time_series <= 0]
        return np.mean(negatives) if len(negatives) > 0 else np.nan

    def std_pos(self, time_series):
        '''Calculate the standard deviation of positive values in the time series.'''
        positives = time_series[time_series >= 0]
        return np.std(positives, axis=0) if len(positives) > 0 else np.nan

    def std_neg(self, time_series):
        '''Calculate the standard deviation of negative values in the time series.'''
        negatives = time_series[time_series <= 0]
        return np.std(negatives, axis=0) if len(negatives) > 0 else np.nan

    def no_zc(self, time_series):
        '''Count the number of zero crossings in the time series.'''
        sign_changes = np.sign(time_series)
        locations_zc = np.where(np.diff(sign_changes))[0]
        return len(locations_zc)

    def no_p(self, time_series):
        '''Count the number of peaks in the time series.'''
        # Find positive peaks
        peak_pos, _ = signal.find_peaks(time_series)
        # Find negative peaks (troughs) by inverting the time series
        peak_neg, _ = signal.find_peaks(-time_series)
        # Combine and sort the indices of peaks and troughs
        locations_extrp = np.sort(np.concatenate((peak_pos, peak_neg)))
        # Return the total number of peaks and troughs
        return len(locations_extrp)

    def mean_tb_zc(self, time_series):
        '''Calculate the mean time between zero crossings in the time series.'''
        # Find zero crossings
        sign_changes = np.sign(time_series)
        zero_crossings = np.where(np.diff(sign_changes))[0]
        # Calculate time between zero crossings
        time_between_zc = np.diff(zero_crossings) / self.sample_freq
        # Compute mean time between zero crossings
        return np.mean(time_between_zc) if len(time_between_zc) > 0 else np.nan

    def mean_tb_p(self, time_series):
        '''Calculate the mean time between peaks in the time series.'''
        # Find positive peaks
        peak_pos, _ = signal.find_peaks(time_series)
        # Find negative peaks (troughs) by inverting the time series
        peak_neg, _ = signal.find_peaks(-time_series)
        # Combine and sort the indices of peaks and troughs
        locations_extrp = np.sort(np.concatenate((peak_pos, peak_neg)))
        # Calculate time differences between consecutive peaks/troughs
        time_between_peaks = np.diff(locations_extrp) / self.sample_freq
        # Compute mean time between peaks/troughs
        return np.mean(time_between_peaks) if len(time_between_peaks) > 0 else np.nan

    def std_tb_zc(self, time_series):
        '''Calculate the standard deviation of time between zero crossings in the time series.'''
        # Find zero crossings
        sign_changes = np.sign(time_series)
        zero_crossings = np.where(np.diff(sign_changes))[0]
        # Calculate time between zero crossings
        time_between_zc = np.diff(zero_crossings) / self.sample_freq
        # Compute standard deviation of time between zero crossings
        return np.std(time_between_zc, axis=0) if len(time_between_zc) > 0 else np.nan

    def std_tb_p(self, time_series):
        '''Calculate the standard deviation of time between peaks in the time series.'''
        # Find positive peaks
        peak_pos, _ = signal.find_peaks(time_series)
        # Find negative peaks (troughs) by inverting the time series
        peak_neg, _ = signal.find_peaks(-time_series)
        # Combine and sort the indices of peaks and troughs
        locations_extrp = np.sort(np.concatenate((peak_pos, peak_neg)))
        # Calculate time differences between peaks and troughs
        time_between_peaks = np.diff(locations_extrp) / self.sample_freq
        # Compute standard deviation of time between peaks
        return np.std(time_between_peaks, axis=0) if len(time_between_peaks) > 0 else np.nan

    def ratio_zcp(self, time_series):
        '''Calculate the ratio of zero crossings to peaks in the time series.'''
        peaks_troughs_cnt = self.no_p(time_series)
        zero_crossings_cnt = self.no_zc(time_series)
        return (zero_crossings_cnt / peaks_troughs_cnt) if peaks_troughs_cnt > 0 else np.nan

    def instantaneous_power(self, time_series):
        '''Calculate the instantaneous power of the time series.'''
        return np.square(time_series)

    def average_power(self, time_series):
        '''Calculate the average power of the time series.'''
        return np.mean(np.square(time_series))

    def rms(self, time_series):
        '''Calculate the root mean square of the time series.'''
        return np.sqrt(np.mean(np.square(time_series)))

    def instantaneous_activity(self, time_series):
        '''Calculate the instantaneous activity of the time series.'''
        return np.abs(time_series)

    def average_activity(self, time_series):
        '''Calculate the average activity of the time series.'''
        return np.mean(np.abs(time_series))

    def fit_sin(self, yy, set_freq=0):
        '''
        Find sine wave approximations of time series: 'amp', 'omega', 'phase', 'offset', 'freq', 'period' and 'fitfunc'
        
        Source: https://stackoverflow.com/questions/16716302/how-do-i-fit-a-sine-curve-to-my-data-with-pylab-and-numpy
        '''
        cutoff = 2
        yy = np.array(yy)
        tt = np.linspace(0, len(yy) / self.sample_freq, len(yy))
        guess_amp = np.std(yy) * np.sqrt(2)
        
        if set_freq == 0:
            ff = np.fft.rfftfreq(len(tt), (tt[1] - tt[0]))   # assume uniform spacing
            Fyy = abs(np.fft.rfft(yy))
            max__freq = self.sample_freq / 5.5  # nyquist criterion with some margin
            auto_guess_freq = min(abs(ff[np.argmax(Fyy[cutoff:]) + cutoff]), max__freq)  # excluding the zero frequency 'peak', which is related to offset and also very low frequencies
            freq = 2. * np.pi * auto_guess_freq
            param_bounds = ([guess_amp * 0.5, freq * 0.95, -2 * np.pi], [guess_amp * 2.0, freq * 1.05, 2 * np.pi])
            def sinfunc(t, A, w, p):  return A * np.sin(w * t + p)
            guess = np.array([guess_amp, freq, 0.])
        else:
            param_bounds = ([guess_amp * 0.5, -2 * np.pi], [guess_amp * 2.0, 2 * np.pi])
            def sinfunc(t, A, p):  return A * np.sin(2 * np.pi * set_freq * t + p)
            guess = np.array([guess_amp, 0.])

        try:
            popt, _ = optimize.curve_fit(sinfunc, tt, yy, p0=guess, bounds=param_bounds, maxfev=1000)
        except RuntimeError as e:
            print(f'Unable to fit sin wave to time series data: {e}')
            return [np.nan, np.nan, np.nan, np.nan]

        if set_freq == 0:
            A, w, p = popt
            freq = w / (2. * np.pi)
        else:
            A, p = popt
            w = 2. * np.pi * set_freq
            freq = set_freq
            
        freq = max(freq, 1e-7)  # Ensure freq is not zero

        return [A, w, p, freq] # 'period': 1./f, 'fitfunc': lambda t: A * np.sin(w * t + p) , 'maxcov': np.max(pcov), 'rawres': (guess,popt,pcov)}


def flip_amp_with_phi(amplitudes, phi):
    phi_diffs = phi - phi[0]
    wrapped_phi_diffs = (phi_diffs + np.pi) % (2 * np.pi) - np.pi

    for i in range(1, len(wrapped_phi_diffs)):
        if wrapped_phi_diffs[i] < 0:
            wrapped_phi_diffs[i] += np.pi
            amplitudes[i] *= -1
    
    return amplitudes, wrapped_phi_diffs


class EigenwalkerPCA(SVD):
    def __init__(self, num_PCs_to_use=2):
        super().__init__()
        self.num_PCs_to_use = num_PCs_to_use

        self.W_prime = np.array([])
        self.W_0 = np.array([])
        self.V = np.array([])
        self.K = np.array([])

        self.results = pd.DataFrame()


    def preprocess(self, dfs: pd.DataFrame, wtype='full') -> np.ndarray:
        if len(dfs) == 0:
            raise IndexError("No data provided")
        
        num_of_features = dfs[0].loc['Loadings'].shape[0]
        num_of_subjects = len(dfs)
        mean_postures = np.empty((num_of_subjects, num_of_features))
        eigenpostures = np.empty((num_of_subjects, (self.num_PCs_to_use) * num_of_features))
        phis = np.empty((num_of_subjects, self.num_PCs_to_use - 1))
        omegas = np.empty((num_of_subjects, 1))
        groups = []

        # Get loadings from first PC as a reference
        Vt_ref = dfs[0].loc['Loadings'].to_numpy(dtype=float)[:, :self.num_PCs_to_use]
        
        for i, df in enumerate(dfs):
            ### Start
            phi = df.loc['PP Metrics', 'Sin Approx. Phase'].to_numpy(dtype=float)[:self.num_PCs_to_use]
            omega = df.loc['PP Metrics', 'Sin Approx. Omega'].to_numpy(dtype=float)[0]
            amplitudes = df.loc['PP Metrics', 'Sin Approx. Amplitude'].to_numpy(dtype=float)[:self.num_PCs_to_use]
            mean_posture = df.loc['Data Mean'].to_numpy(dtype=float)[0]
            Vt = df.loc['Loadings'].to_numpy(dtype=float)[:, :self.num_PCs_to_use]
            group = str(df.loc['Group'].index.get_level_values('level_1').tolist()[0])
            groups.append(group)

            # Flip scores (and hence loadings) to keep them consistent accross all subjects
            Vt_dist = np.linalg.norm(Vt_ref - Vt, axis=0)
            Vt_dist_flipped = np.linalg.norm(Vt_ref - (Vt * -1), axis=0)
            flip_pcx = np.argmax(np.array([Vt_dist, Vt_dist_flipped]), axis=0)
            amplitudes *= np.sign(flip_pcx - 0.5)
            phi += np.pi * flip_pcx

            # Take Phi out here if only Eigenwalker consistency matters

            amplitudes, phi = flip_amp_with_phi(amplitudes, phi)

            # Absorb the amplitude into the loadings and concatenate the mean posture
            p = (Vt.T * amplitudes[:, np.newaxis]).ravel()
            mean_postures[i] = mean_posture
            eigenpostures[i] = p
            phis[i] = phi[1]
            omegas[i] = omega

        full_walkers = np.concatenate((mean_postures, eigenpostures, phis, omegas), axis=1)
        self.average_walker = np.mean(full_walkers, axis=0)

        if wtype == 'full':
            W = full_walkers

        elif wtype == 'structural':
            W = mean_postures

        elif wtype == 'dynamic':
            W = np.concatenate((eigenpostures, phis, omegas), axis=1)
            
        unique_groups = np.unique(groups)
        group_masks = {unique_group: np.zeros(len(groups), dtype=int) for unique_group in unique_groups}
        for idx, group in enumerate(groups):
            group_masks[group][idx] = 1
        grouped_average_walker = {}
        for unique_group, mask in group_masks.items():
            mask = np.array(mask, dtype=bool) # NEXT
            grouped_average_walker[unique_group] = np.mean(W[mask], axis=0)

        return W, groups, grouped_average_walker
    
    def project_walkers(self, W: np.ndarray) -> np.ndarray:
        return (np.linalg.pinv(self.V) @ (W.T - self.W_0))[:-1]  # Pseudo-inverse of V to handle non-square matrix, remove last dimension as the number of dimensions is always 1 less than the number of samples

    def fit_walkers(self, W: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # "Whitening" the data (Same as this except without mean centring which is done by norm_weight_centre regardless: scaler = StandardScaler() W = scaler.fit_transform(W))
        u = np.std(W, axis=0)
        self.W_prime = W @ np.diag(1 / u)
        self.W_prime -= np.mean(self.W_prime, axis=0)  # Feature centring
        #print(f'Vector W_prime shape: {self.W_prime.shape}')
        self.fit(self.W_prime)  # Fit PCA on combined data of all subjects

        V_prime = self.components.T # Technically only [:, :-1] meaningful PCs (as constrainted by the number of samples/subjects the last PC is meaningless)
        self.V = np.diag(u) @ V_prime  # These are the eigenwalkers [loadings, pcs 1-N]
        self.W_0 = np.mean(W.T, axis=1).reshape(-1, 1)  # Average walker in each column
        self.K = self.project_walkers(W)

        return self.W_0, self.V, self.K, self.eigenvalues

    def transform_k_to_w(self, k):
        return self.W_0.reshape(-1) + np.dot(k, self.V.T)

    def reconstruct(self, w, num_of_eigenposture_features, sample_freq, d_norm, weight_vec, coord_transform, wtype='full'):
        #print(w.shape)
        #print(num_of_eigenposture_features * (self.num_PCs_to_use + 1))
        temp = self.average_walker.copy()
        if wtype == 'structural':
            temp[:len(w)] = w
            w = temp
        elif wtype == 'dynamic':
            temp[num_of_eigenposture_features:] = w
            w = temp

        # Deconstruct W
        eigenpostures = w[0:num_of_eigenposture_features * (self.num_PCs_to_use + 1)].reshape(self.num_PCs_to_use + 1, -1)
        mean_posture, eigenpostures = eigenpostures[0], eigenpostures[1:]
        phis = np.insert(w[-self.num_PCs_to_use:-1], 0, 0.0)
        omega = w[-1]

        tt = np.arange(0, 2*np.pi/omega, 0.01)
        scores = np.array([np.sin(tt * omega + phi) for phi in phis])

        pca = PCA_Model(sample_freq)
        pca.d_norm = d_norm
        pca.weight_vector = weight_vec
        reconstructed_eigenwalker = pca.reconstruct_data(mean_posture, scores.T, eigenpostures.T, coord_transform)
        
        return reconstructed_eigenwalker
    
    def process_results(self, W):
        columns = [f'PC{i+1}' for i in range(self.W_0.shape[0])]
        df0 = pd.DataFrame(self.W_0.T, index=['W_0'], columns=columns)
        df1 = pd.DataFrame(W, index=[f'W_{i+1}' for i in range(W.shape[0])], columns=columns[:W.shape[1]])
        df2 = pd.DataFrame(self.V.T, index=[f'V_{i+1}' for i in range(self.V.shape[1])], columns=columns)
        df3 = pd.DataFrame(self.K.T, index=[f'K_{i+1}' for i in range(self.K.shape[1])], columns=columns[:self.K.shape[0]])
        df4 = pd.DataFrame(self.eigenvalues.reshape(1, -1), index=['Eigenvalues'], columns=columns[:self.V.shape[1]])
        df3 = df3.reindex(columns=columns)
        df4 = df4.reindex(columns=columns)
        self.results = pd.concat([df0, df1, df2, df3, df4])
        return self.results

