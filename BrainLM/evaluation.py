import pandas as pd
import numpy as np
import nibabel as nib
from nilearn import image, plotting
from pathlib import Path
from nilearn.plotting import plot_roi, view_img
from nilearn.plotting import plot_stat_map
# from nimare.decode import CorrelationDecoder
# from nimare.dataset import Dataset
import random
from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import os
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from llava.psy_llava_utils.general_utils import sanitize_string
from llava.psy_llava_utils.constants import NEUROQUERY_LITERATURE_DIR
from llava.explain.visualize import save_colorbar

coords_ds_path = '/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/BrainLM/toolkit/atlases/A424_Coordinates.dat'
coords_ds = pd.read_csv(coords_ds_path, delimiter='\t', names=['idx', 'X', 'Y', 'Z'])
coords_labels_path = os.path.join(os.path.dirname(coords_ds_path), 'A424.dlabel.nii')
networks_coords_path ='/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/BrainLM/toolkit/atlases/Schaefer2018_1000Parcels_7Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'
networks_coords = pd.read_csv(networks_coords_path)
coords_labels = nib.load(coords_labels_path)
main_atlas_path = os.path.join(os.path.dirname(coords_ds_path), 'A424.nii.gz')
main_atlas = nib.load(main_atlas_path)


class MetaAnalyze():
    def __init__(self, main_atlas, coords_labels, main_coords, network_coords, do_network_validation=False):
        self.atlas_nimg = main_atlas
        self.atlas_volume = self.atlas_nimg.get_fdata()
        self.coords_labels = coords_labels
        self.parcel_dict = self.create_parcel_dict(self.coords_labels, main_coords, network_coords)
        if do_network_validation:
            self.plot_yeo_network_projection(interactive=True)

    def create_parcel_dict(self, coords_labels, atlas_coords, network_coords):
        my_coords = atlas_coords[['X', 'Y', 'Z']].values
        yeo_coords = network_coords[['R', 'A', 'S']].values
        yeo_network_names = network_coords['ROI Name'].apply(self.extract_network_name)

        label_table = coords_labels.header.get_axis(0).label[0]
        data = coords_labels.get_fdata()

        total_voxels = (data > 0).sum()  # Count all non-zero voxels

        parcel_dict = {}

        for key, value in label_table.items():
            if key == 0:
                print('skipping background...')
                continue
            parcel_name = value[0]
            parcel_color = value[1]
            parcel_voxels = (data == key).sum()  # Count voxels for this parcel
            relative_size = parcel_voxels / total_voxels if total_voxels > 0 else 0

            x,y,z = my_coords[key-1] #-1 because the first is index 1 but its index 0
            distances = np.linalg.norm(yeo_coords - np.array([x, y, z]), axis=1)
            closest_idx = np.argmin(distances)

            # Extract the network name
            network_name = yeo_network_names.iloc[closest_idx]


            parcel_dict[key] = {
                'parcel_name': parcel_name,
                'color': parcel_color,
                'total_voxels': parcel_voxels,
                'relative_size': relative_size,
                'full_network_name':network_name,
                'network_abbreviation':network_name[:4]}

        return parcel_dict

    def plot_yeo_network_projection(self, output_path='yeo_networks.png',
                                    interactive=False, display_mode='ortho', cut_coords=None):
        all_networks = [
            parcel_info['full_network_name']
            for parcel_info in self.parcel_dict.values()
        ]
        unique_networks = sorted(set(all_networks))
        network_to_int = {
            net: i + 1 for i, net in enumerate(unique_networks)
        }
        # +1 so background can stay 0

        # 2) Create an empty volume the same shape as the atlas
        network_volume = np.zeros_like(self.atlas_volume, dtype=int)

        # 3) Fill the new volume based on each voxel's parcel --> network label
        for parcel_index, parcel_data in self.parcel_dict.items():
            network_name = parcel_data['full_network_name']
            net_id = network_to_int[network_name]
            # every voxel that matches `parcel_index` in the atlas goes to net_id
            network_volume[self.atlas_volume == parcel_index] = net_id

        # Make a NIfTI image
        network_img = nib.Nifti1Image(network_volume.astype(np.int16), affine=self.atlas_nimg.affine)

        brain_networks = {
            'Default Mode': 'red',
            'Dorsal Attention': 'green',
            'Executive Control': 'orange',
            'Limbic': 'yellow',
            'Salience Ventral Attention': 'pink',
            'Somato Motor': 'blue',
            'Visual': 'purple'
        }
        color_list = [mcolors.to_rgb(color) for color in brain_networks.values()]

        # Create a discrete colormap
        # We'll insert a color for the background (0) as well:
        # e.g., black or white for label=0
        cmap_values = [(0, 0, 0)] + color_list  # 0 is black, etc.
        discrete_cmap = ListedColormap(cmap_values)

        # 5) Plot it
        if interactive:
            # Create an interactive view
            html_view = view_img(
                network_img,
                threshold=0,  # show everything
                cmap=discrete_cmap,
                title="Yeo Network Projection",
                symmetric_cmap=False,
            )
            # Save HTML
            html_file = os.path.splitext(output_path)[0] + ".html"
            html_view.save_as_html(html_file)
            print(f"Interactive HTML plot saved to {html_file}")

            # (Legends are not natively supported inside the interactive viewer,
            #  so you can optionally generate a separate legend .png if desired.)
            fig_legend, ax_legend = plt.subplots(figsize=(2, 2))
            legend_patches = []
            for net_name, color in zip(unique_networks, color_list):
                legend_patches.append(mpatches.Patch(color=color, label=net_name))
            ax_legend.axis('off')
            ax_legend.legend(
                handles=legend_patches,
                loc='upper left',
                bbox_to_anchor=(0, 1),
                frameon=False
            )
            legend_png = os.path.splitext(output_path)[0] + "_legend.png"
            fig_legend.savefig(legend_png, bbox_inches='tight', dpi=150)
            plt.close(fig_legend)
            print(f"Legend PNG saved to {legend_png}")

        else:
            # Produce a static image via plot_roi
            display = plot_roi(
                roi_img=network_img,
                display_mode=display_mode,
                cut_coords=[25, -5, 0],  # Example values; adjust as needed
                cmap=discrete_cmap,
                alpha=0.7,
                title="Yeo Network Projection",
                colorbar=False
            )

            # We create a custom legend
            # Grab the current figure from the display
            fig = plt.gcf()
            legend_patches = []
            for net_name, color in zip(unique_networks, color_list):
                legend_patches.append(mpatches.Patch(color=color, label=net_name))
            # Add the legend outside the main axes
            plt.legend(
                handles=legend_patches,
                bbox_to_anchor=(1.05, 1),
                loc='upper left',
                borderaxespad=0.
            )

            # Save figure
            display.savefig(output_path, dpi=150)
            display.close()
            print(f"Static figure saved to {output_path}")

    def project_to_statistical_map(self, parcelled_array):
        normalized_scores = (parcelled_array - parcelled_array.min()) / (parcelled_array.max() - parcelled_array.min())
        statistical_map = np.zeros_like(self.atlas_volume)
        parcel_info = {}
        for index, score in enumerate(normalized_scores):
            index += 1
            statistical_map_indices = self.atlas_volume == index
            statistical_map[statistical_map_indices] = score
            parcel_data = self.parcel_dict[index]
            parcel_name = parcel_data['parcel_name']
            if parcel_name not in parcel_info:
                parcel_info[parcel_name] = {'value':[score],'network':parcel_data['full_network_name']}
            else:
                parcel_info[parcel_name]['value'].append(score)
        parcel_info = {k:{'value':np.mean(v['value']),'network':v['network']} for k,v in parcel_info.copy().items()}

        statistical_nimg = nib.Nifti1Image(statistical_map, self.atlas_nimg.affine, self.atlas_nimg.header)
        sorted_parcel_info = dict(sorted(parcel_info.items(), key=lambda item: item[1]['value'], reverse=True))
        return statistical_nimg, sorted_parcel_info

    def extract_network_name(self, roi_name):
        if 'Cont' in roi_name:
            return 'Executive Control'
        elif 'Vis' in roi_name:
            return 'Visual'
        elif 'Default' in roi_name:
            return 'Default Mode'
        elif 'SomMot' in roi_name:
            return 'Somato Motor'
        elif 'DorsAttn' in roi_name:
            return 'Dorsal Attention'
        elif 'SalVentA' in roi_name:
            return 'Salience Ventral Attention'
        elif 'Limbic' in roi_name:
            return 'Limbic'
        else:
            return 'Unknown'

    def scale_to_unit_range(self, map):
        """
        Scales the voxel values of a 3D map to the range [0, 1].

        Parameters:
        - map: np.array file of the brain map.

        Returns:
        - map: np.array of the brain map after min-max scaling.
        """
        data = map.get_fdata()
        data_reshaped = data.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        data_scaled = scaler.fit_transform(data_reshaped).reshape(data.shape)
        rescaled_map = nib.Nifti1Image(data_scaled, map.affine)
        return rescaled_map

    def compute_distance(self, map1, map2):
        """
        Computes the mean of the Euclidean distance of each voxel,
        in 2 NIfTI brain maps after scaling to [0, 1].

        Parameters:
        - map1_path: first NIfTI file.
        - map2_path: second NIfTI file.

        Returns:
        - float: Mean Euclidean distance between the two maps.
        """
        map1_data = map1.get_fdata()
        map2_data = map2.get_fdata()
        distance_map = np.sqrt((map1_data - map2_data) ** 2)
        mean_distance = distance_map.mean()
        return mean_distance

    def create_and_save_overlay_map(self, results_dir, map1_img, map2_img, interactive, mode='ortho', cut_coords = None, threshold=95):
        """
        Create an overlay map highlighting the intersection and strong voxels in two fMRI maps.

        Parameters:
        - map1_path: the first NIfTI image.
        - map2_path: the second NIfTI image.
        """
        map1_data = map1_img.get_fdata()
        map2_data = map2_img.get_fdata()

        # Compute percentile thresholds
        thr_map1 = np.percentile(map1_data[~np.isnan(map1_data)], threshold)
        thr_map2 = np.percentile(map2_data[~np.isnan(map2_data)], threshold)

        # Create masks based on thresholds
        map1_mask = map1_data > thr_map1
        map2_mask = map2_data > thr_map2

        # Define regions
        intersection_mask = map1_mask & map2_mask
        map1_only_mask = map1_mask & ~map2_mask
        map2_only_mask = map2_mask & ~map1_mask

        # Compute intersection mean values
        intersection_data = np.zeros_like(map1_data, dtype=float)
        intersection_data[intersection_mask] = (map1_data[intersection_mask] + map2_data[intersection_mask]) / 2.0

        # Map1 only data
        map1_only_data = np.zeros_like(map1_data, dtype=float)
        map1_only_data[map1_only_mask] = map1_data[map1_only_mask]

        # Map2 only data
        map2_only_data = np.zeros_like(map2_data, dtype=float)
        map2_only_data[map2_only_mask] = map2_data[map2_only_mask]

        # Create NIfTI images
        intersection_img = nib.Nifti1Image(intersection_data, affine=map1_img.affine)
        map1_only_img = nib.Nifti1Image(map1_only_data, affine=map1_img.affine)
        map2_only_img = nib.Nifti1Image(map2_only_data, affine=map2_img.affine)

        for color_map, data, label in zip(['hot','Greens','Blues'],
                                          [intersection_data, map1_only_data, map2_only_data],
                                          ['intersection','gradients','neuroquery']):
            save_colorbar(color_map,data.min(),data.max(),os.path.join(results_dir,f'overlay_{label}_colormap.png'),label)

        if interactive:
            from nilearn.plotting import view_img
            for color_map, data, label in zip(['hot', 'Greens', 'Blues'],
                                              [intersection_img, map1_only_img, map2_only_img],
                                              ['intersection', 'gradients', 'neuroquery']):
                output_file = os.path.join(results_dir, f'{label}.html')
                view = view_img(
                    intersection_img,
                    threshold=0,
                    cmap=color_map,
                    title=f'{label} Map'
                )
                view.save_as_html(output_file)
        else:
            output_file = os.path.join(results_dir, 'overlay_gradients_and_neuroquery.png')
            # Plot intersection in 'hot' colormap
            display = plotting.plot_stat_map(
                intersection_img,
                display_mode=mode,
                cut_coords=cut_coords,
                cmap='hot',
                alpha=0.9,
                black_bg=True,
                colorbar=False,
                title=f'Intersection (≥{threshold}th pct) mean values'
            )

            # Add map1-only voxels (green-ish)
            display.add_overlay(
                map1_only_img,
                cmap='Greens',
                alpha=0.9
            )

            # Add map2-only voxels (blue-ish)
            display.add_overlay(
                map2_only_img,
                cmap='Blues',
                alpha=0.9
            )

            # Save figure
            display.savefig(output_file,dpi=300)
            display.close()

    def compare_maps(self, gradient_map, neuroquery_map):
        """
        compares the distance between
        gradient map to the neuroquery map (which represnts the question)
        to the distance between
        random map to the neuroquery map
        in order to verify that the gradients are indeed "closer" to that specific question.

        Parameters:
        - gradient_map_path: Path to the gradient map NIfTI file.
        - random_map_path: Path to the random map NIfTI file.
        - neuroquery_map_path: Path to the neuroquesry map NIfTI file.

        Returns:
        - a dict of the comparison results.
        """
        # assert (gradient_map.affine == neuroquery_map.affine) and (gradient_map.shape == neuroquery_map.shape)
        gradient_data = gradient_map.get_fdata().flatten()
        neuroquery_data = neuroquery_map.get_fdata().flatten()
        pearson_matrix = np.corrcoef(gradient_data, neuroquery_data)
        assert np.isclose(pearson_matrix[0, 1], pearson_matrix[1, 0]), f"Correlation matrix is not symmetric!\n matrix: {pearson_matrix}"
        pearson_corr = pearson_matrix[0, 1]
        spearman_corr = spearmanr(gradient_data, neuroquery_data)[0]
        gradient_data = gradient_data.reshape(1, -1)
        neuroquery_data = neuroquery_data.reshape(1, -1)
        cosine_corr = cosine_similarity(gradient_data, neuroquery_data)[0][0]
        L2_dist = self.compute_distance(gradient_map, neuroquery_map)
        analysis_results = {
            "L2 distance": L2_dist,
            "Pearson correlation": pearson_corr,
            "Spearman correlation": spearman_corr,
            "Cosine correlation": cosine_corr,
        }
        return analysis_results

    def get_neuroquery_map(self, question, target_img):
        directory = Path(NEUROQUERY_LITERATURE_DIR)
        assert directory.exists(), f"The directory {directory} does not exist! run preprocess_questions.py"
        question_sanitized = sanitize_string(question)
        question_neuroquery_map_path = directory.joinpath(f"{question_sanitized}_neuroquery_statistical_map.nii.gz")
        try:
            neuroquery_map = nib.load(question_neuroquery_map_path)
        except FileNotFoundError:
            neuroquery_map = None
            for file in directory.iterdir():
                if question_sanitized.lower() in str(file).lower():
                    neuroquery_map = nib.load(file)
                    break
            if neuroquery_map is None:
                raise FileNotFoundError
        if target_img is not None:
            resampled_neuroquery_map = image.resample_to_img(neuroquery_map, target_img, interpolation="nearest",
                                                         force_resample=True, copy_header=True)
            return resampled_neuroquery_map
        else:
            return neuroquery_map

    def neuroquery_question_analysis(self, results_dir, question, statistical_map, num_permutations=10, interactive=False):
        """
        compare the given question's neuroquery map to the statistical map.
        choose 10 random questions.
        compare each random question's neuroquery map to the statistical map.
        average the results.

        Parameters:
        - question: string of the question
        - statistical_map: string of the path to the NifTi file.
        - n: number of random iterations, default is 10.

        Returns:
        - a dict of the comparison results.
        """

        resampled_neuroquery_map = self.get_neuroquery_map(question, statistical_map)
        neuroquery_scaled = self.scale_to_unit_range(resampled_neuroquery_map)
        gradient_scaled = self.scale_to_unit_range(statistical_map)

        self.create_and_save_overlay_map(results_dir, gradient_scaled, neuroquery_scaled, interactive)

        question_analysis = self.compare_maps(gradient_scaled, neuroquery_scaled)
        other_questions = [f for f in directory.iterdir() if f != f"{question_sanitized}_neuroquery_statistical_map.nii.gz"]
        assert len(other_questions) >= num_permutations, "Not enough random questions available for permutations."
        random_questions = random.sample(other_questions, num_permutations)
        permutation_analysis = {metric: 0 for metric in question_analysis}
        for random_map_path in random_questions:
            random_map = nib.load(random_map_path)
            if len(np.unique(random_map.get_fdata())) == 1:
                print('found bad map, ignoring...')
                continue
            resampled_random_map = image.resample_to_img(random_map, statistical_map, interpolation="nearest", force_resample=True,copy_header=True)
            random_scaled = self.scale_to_unit_range(resampled_random_map)
            curr_random_analysis = self.compare_maps(gradient_scaled, random_scaled)
            for metric, value in curr_random_analysis.items():
                permutation_analysis[metric] += value
        permutation_analysis = {metric: value / num_permutations for metric, value in permutation_analysis.items()}
        final_analysis = {}
        for metric in question_analysis:
            if metric in ["Pearson correlation", "Spearman correlation", "Cosine correlation"]:
                # Higher correlation values mean greater similarity; we want positive differences
                final_analysis[metric] = question_analysis[metric] - permutation_analysis[metric]
            elif metric == "L2 distance":
                # Lower L2 distance means greater similarity; invert the sign for positive results
                final_analysis[metric] = permutation_analysis[metric] - question_analysis[metric]
            else:
                raise ValueError(f"Unhandled metric: {metric}")
        return final_analysis


    #def project_parcellation_to_meta(self, parcellated_array):
    #    statistical_nimg = self.project_to_statistical_map(parcellated_array)
#
    #    dset = Dataset.load('path_to_your_dataset.pkl.gz')
#
    #    # Initialize the CorrelationDecoder
    #    decoder = CorrelationDecoder()
    #    decoder.fit(dset)
    #    decoded_terms = decoder.transform(statistical_nimg)
if __name__=="__main__":
   meta_analyzer = MetaAnalyze(main_atlas, coords_labels, coords_ds, networks_coords)
   meta_analyzer.neuroquery_question_analysis(results_dir, question, statistical_map, num_permutations=10, interactive=False)