# To vectorize functions
import numpy as np
import pandas as pd
import xarray as xr
import skimage as sk
import matplotlib.pyplot as plt
import re
import scipy
import h5py
import math
import pickle
import matplotlib.colors as mcolors
from PIL import Image

###################################################################################################
# DATA MANIPULATION FUNCTIONS
###################################################################################################


def import_segmentation_data_numpy(file_path):
    """Function to import the segmentation data (as exported by Ilastik) from a numpy file

    Args:
        file_path (str): Path to the .npy file with the data

    Returns:
        np.array: returns an array corresponding to the image where 0 is the background and 1 is the particles
    """
    data = np.load(file_path)
    data = np.where(data == 0, 0, 1)
    return data

def import_segmentation_data_h5(file_path):
    """Function to import the segmentation data (as exported by Ilastik) from a h5df file

    Args:
        file_path (str): Path to the .h5 file with the data

    Returns:
        np.array: returns an array corresponding to the image where 0 is the background and 1 is the particles
    """
    with h5py.File(file_path, "r") as f:
        data =f["exported_data"][()]
    data = np.where(data !=3, 0, 1)
    data  = data[:, :, 0]
    return data

def import_and_display_segmentation(file_path_segmentation, file_path_image, title):
    """Function to display the segmentation and the raw SEM images next to eachother

    Args:
        file_path_segmentation (str): path of the .h5 segmentation file
        file_path_image (str): path of the SEM image
        title (str): Title of the figures
    """
    with h5py.File(file_path_segmentation, "r") as f:
        data =f["exported_data"][()]
    
    data  = data[:, :, 0]
    cmap = mcolors.ListedColormap(['yellow', 'blue', 'red'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
    ax1.imshow(data, cmap=cmap)
    data = np.where(data !=3, 0, 1)
    image = Image.open(file_path_image)
    ax2.imshow(image, cmap="grey")

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    fig.suptitle(title)
    plt.show()



def import_ecd_aspect_data(ecd_path, aspectratio_path):
    """Function to import the raw ecd and aspect ratio data and make an xarray with it

    Args:
        ecd_path (str): Path+filename of the ecd data (csv) as output by CEMIA
        aspectratio_path (str): Path+filename of the aspectratio data (csv) as output by CEMIA

    Returns:
        xarray: An xarray with dimensions properties (coords 'ECD' and 'Aspect Ratio) and particles
    """
    ecd_array = np.loadtxt(ecd_path)
    aspectratio_array = np.loadtxt(aspectratio_path)
    data = xr.DataArray(
        [ecd_array, aspectratio_array],
        dims=["properties", "particles"],
        coords={"properties": ["ECD", "Aspect Ratio"]},
    )
    return data





def identify_particle_properties(data, conversion_factor, split_value=0.5):
    """Function to create an xarray with the properties of the particles in an image

    Args:
        data (np.array): Array corresponding to a binary image
        conversion_factor: Value to convert the pixels to um
        split_value (float): Optional. value between 0 and 1, that acts as the limit between the area ratio values that 
        correspond to the alpha intermetallic and the beta intermetallic. Defaults to 0.5

    Returns:
        xarray: An xarray with the dimensions particles and properties (coords, 'area', 'area_convex', 
                'axis_major_length', 'axis_minor_length', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 
                'centroid-0', 'centroid-1', 'ECD', 'orientation', 'perimeter', 'Aspect Ratio', 'area_ratio', 'intermetallic). 
                The intermetallic column is 1 if it is alpha intermetallic and 2 if it's a beta intermetallic
    """
    labeled_image, num_labels = sk.measure.label(data, connectivity=2, return_num=True)
    print("lables", num_labels)
    properties = sk.measure.regionprops_table(
        labeled_image,
        properties=[
            "area",
            "area_convex",
            "axis_major_length",
            "axis_minor_length",
            "bbox",
            "centroid",
            "equivalent_diameter_area",
            "orientation",
            "perimeter", 
        ],
    )
    properties_df = pd.DataFrame.from_dict(properties)
    largest_dimensions= []
    for region in sk.measure.regionprops(labeled_image):
        convex_hull = sk.morphology.convex_hull_image(region.convex_image)

        if region.area < 5:  # Remove the particles that are too small
            largest_dimensions.append(np.nan)
            continue

        perimeter_coords = np.argwhere(convex_hull)

        if len(np.unique(perimeter_coords[:, 0])) == 1:  # All x values are the same
            largest_dimensions.append(np.nan)
            continue
        elif len(np.unique(perimeter_coords[:, 1])) == 1:  # All y values are the same
            largest_dimensions.append(np.nan)
            continue

        if len(perimeter_coords) > 1:
            perimeter_coords = np.column_stack(np.where(convex_hull))
            hull = scipy.spatial.ConvexHull(perimeter_coords)
            max_dist = 0
            for i in range(len(hull.vertices)):
                for j in range(i + 1, len(hull.vertices)):
                    p1 = perimeter_coords[hull.vertices[i]]
                    p2 = perimeter_coords[hull.vertices[j]]
                    dist = np.linalg.norm(p1 - p2)
                    if dist > max_dist:
                        max_dist = dist

            # Print the largest dimension length
            # print(f"Largest Dimension Length: {max_dist:.2f}")
            largest_dimensions.append(max_dist)

    largest_dimensions = [x / conversion_factor for x in largest_dimensions]
    properties_df["largest_dimension"] = largest_dimensions


    properties_df["area"]= properties_df["area"] / (conversion_factor * conversion_factor)
    properties_df["smallest_dim_area"] = properties_df["area"]/properties_df["largest_dimension"]
    properties_df["area_convex"] = properties_df["area_convex"] / (conversion_factor * conversion_factor)
    properties_df["axis_major_length"] = properties_df["axis_major_length"]/ conversion_factor
    properties_df["axis_minor_length"] = properties_df["axis_minor_length"] / conversion_factor
    properties_df["equivalent_diameter_area"] = properties_df["equivalent_diameter_area"] / conversion_factor
    properties_df["perimeter"] = properties_df["perimeter"]/ conversion_factor

    properties_df.rename(columns={"equivalent_diameter_area": "ECD"}, inplace=True)
    properties_df['Aspect Ratio'] = properties_df["axis_major_length"] / properties_df["axis_minor_length"]
    properties_df["area_ratio"] = properties_df["area"] / properties_df['area_convex']
    properties_df["intermetallic"] = np.where(properties_df["area_ratio"] < split_value, 1, 2)
    properties_df['orientation_deg'] =  90 - np.abs(properties_df['orientation']) * (180 / np.pi)
    properties_df["ECD"] = np.where(properties_df["ECD"]< 0.1, np.nan, properties_df["ECD"])
    # properties_df["orientation_deg"] = np.where(properties_df["Aspect Ratio"] == 1, 0, properties_df["orientation_deg"])
    properties_df = properties_df.dropna(axis=0)
    properties_df = properties_df.reset_index(drop=True)

    properties_xarray = xr.DataArray(properties_df, dims=["particles", "properties"])
    properties_xarray = properties_xarray.where(np.isfinite(properties_xarray), np.nan)
    return properties_xarray, labeled_image



# def identify_particle_properties(data, conversion_factor, split_value=0.5):
#     """Function to create an xarray with the properties of the particles in an image

#     Args:
#         data (np.array): Array corresponding to a binary image
#         conversion_factor: Value to convert the pixels to um
#         split_value (float): Optional. value between 0 and 1, that acts as the limit between the area ratio values that 
#         correspond to the alpha intermetallic and the beta intermetallic. Defaults to 0.5

#     Returns:
#         xarray: An xarray with the dimensions particles and properties (coords, 'area', 'area_convex', 
#                 'axis_major_length', 'axis_minor_length', 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 
#                 'centroid-0', 'centroid-1', 'ECD', 'orientation', 'perimeter', 'Aspect Ratio', 'area_ratio', 'intermetallic). 
#                 The intermetallic column is 1 if it is alpha intermetallic and 2 if it's a beta intermetallic
#     """
#     labeled_image, num_labels = sk.measure.label(data, connectivity=2, return_num=True)
#     print("lables", num_labels)
#     properties = sk.measure.regionprops_table(
#         labeled_image,
#         properties=[
#             "area",
#             "area_convex",
#             "axis_major_length",
#             "axis_minor_length",
#             "bbox",
#             "centroid",
#             "equivalent_diameter_area",
#             "orientation",
#             "perimeter", 
#         ],
#     )
#     properties_df = pd.DataFrame.from_dict(properties)
#     largest_dimensions= []
#     for region in sk.measure.regionprops(labeled_image):
#         convex_hull = sk.morphology.convex_hull_image(region.convex_image)

#         if region.area < 5:  # Remove the particles that are too small
#             largest_dimensions.append(np.nan)
#             continue

#         perimeter_coords = np.argwhere(convex_hull)

#         if len(np.unique(perimeter_coords[:, 0])) == 1:  # All x values are the same
#             largest_dimensions.append(np.nan)
#             continue
#         elif len(np.unique(perimeter_coords[:, 1])) == 1:  # All y values are the same
#             largest_dimensions.append(np.nan)
#             continue

#         if len(perimeter_coords) > 1:
#             perimeter_coords = np.column_stack(np.where(convex_hull))
#             hull = scipy.spatial.ConvexHull(perimeter_coords)
#             max_dist = 0
#             for i in range(len(hull.vertices)):
#                 for j in range(i + 1, len(hull.vertices)):
#                     p1 = perimeter_coords[hull.vertices[i]]
#                     p2 = perimeter_coords[hull.vertices[j]]
#                     dist = np.linalg.norm(p1 - p2)
#                     if dist > max_dist:
#                         max_dist = dist

#             # Print the largest dimension length
#             # print(f"Largest Dimension Length: {max_dist:.2f}")
#             largest_dimensions.append(max_dist)

#     largest_dimensions = [x / conversion_factor for x in largest_dimensions]
#     properties_df["largest_dimension"] = largest_dimensions


#     properties_df["area"]= properties_df["area"] / (conversion_factor * conversion_factor)
#     properties_df["smallest_dim_area"] = properties_df["area"]/properties_df["largest_dimension"]
#     properties_df["area_convex"] = properties_df["area_convex"] / (conversion_factor * conversion_factor)
#     properties_df["axis_major_length"] = properties_df["axis_major_length"]/ conversion_factor
#     properties_df["axis_minor_length"] = properties_df["axis_minor_length"] / conversion_factor
#     properties_df["equivalent_diameter_area"] = properties_df["equivalent_diameter_area"] / conversion_factor
#     properties_df["perimeter"] = properties_df["perimeter"]/ conversion_factor

#     properties_df.rename(columns={"equivalent_diameter_area": "ECD"}, inplace=True)
#     properties_df['Aspect Ratio'] = properties_df["axis_major_length"] / properties_df["axis_minor_length"]
#     properties_df["area_ratio"] = properties_df["area"] / properties_df['area_convex']
#     properties_df["intermetallic"] = np.where(properties_df["area_ratio"] < split_value, 1, 2)
#     properties_df['orientation_deg'] =  90 - np.abs(properties_df['orientation']) * (180 / np.pi)
#     properties_df["ECD"] = np.where(properties_df["ECD"]< 0.1, np.nan, properties_df["ECD"])
#     # properties_df["orientation_deg"] = np.where(properties_df["Aspect Ratio"] == 1, 0, properties_df["orientation_deg"])
#     properties_df = properties_df.dropna(axis=0)
#     properties_df = properties_df.reset_index(drop=True)

#     properties_xarray = xr.DataArray(properties_df, dims=["particles", "properties"])
#     properties_xarray = properties_xarray.where(np.isfinite(properties_xarray), np.nan)
#     return properties_xarray, labeled_image


def get_smallest_dimension(properties_data, image, conversion_factor):
    """Function to get the smallest dimension of particles and add it to the xarray with the properties

    Args:
        properties_data (xarray): An xarray with the dimensions particles and properties ( at least the following properties: 'bbox-0', 'bbox-1', 'bbox-2', 'bbox-3', 
                 'Aspect Ratio'). 
        image (np.array): Numpy array corresponding to the image with all the particles
        conversion_factor: Value to convert pixels to um

    Returns:
        xarray: The properties xarray that was the input with an extra column corresponding to the smallest distance
    """
    num_particles = properties_data.shape[0]
    smallest_size_list = []

    for i in range(num_particles):
        # Get a subimage for each particle
        bbox_0 = int(properties_data.loc[i, 'bbox-0'].values)
        bbox_1 = int(properties_data.loc[i, 'bbox-1'].values)
        bbox_2 = int(properties_data.loc[i, 'bbox-2'].values) 
        bbox_3 = int(properties_data.loc[i, 'bbox-3'].values) 
        subimage = image[bbox_0:bbox_2, bbox_1:bbox_3]
        subimage = np.pad(subimage, pad_width=1, mode='constant', constant_values=0)

        # Get the contour for each particle
        contours = sk.measure.find_contours(subimage, 0.5)

        # fig, ax = plt.subplots()
        # ax.imshow(subimage, cmap=plt.cm.gray)

        for contour in contours:
            # ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

            # Calculate distances and store the corresponding points (for plotting)
            distances_with_points = []
            num_points = len(contour)

            for j in range(num_points):
                for k in range(j + 1, num_points):
                    point1 = tuple(contour[j])
                    point2 = tuple(contour[k])
                    distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)
                    distances_with_points.append((distance, point1, point2))

            # Sort by distance (keeping the corresponding points)
            distances_with_points_sorted = sorted(distances_with_points, key=lambda x: x[0])


            # determine the percentage to use depending on the aspect ratio
            aspect_ratio = properties_data.sel(particles=i, properties='Aspect Ratio').item()
            area_ratio = properties_data.sel(particles=i, properties='area_ratio')
            if area_ratio < 0.5:
                percent_value = 0.05
            elif aspect_ratio > 4:
                percent_value = 0.1
            elif aspect_ratio > 2:
                percent_value = 0.2
            else:
                percent_value = 0.5

            idx_percent = int(percent_value * len(distances_with_points_sorted))
            distance_percent = distances_with_points_sorted[idx_percent][0]
            point1_percent = distances_with_points_sorted[idx_percent][1]
            point2_percent = distances_with_points_sorted[idx_percent][2]

            #Print the distance at 30% through the sorted list
            print(f"Distance at {100*percent_value}% through the sorted list: {distance_percent:.3f}")
            # print(f"Corresponding points: {point1_percent}, {point2_percent}")
            # print(f"aspect ratio of particle  { aspect_ratio}")

            #plot the line between the two points
            # ax.plot([point1_percent[1], point2_percent[1]],
            #         [point1_percent[0], point2_percent[0]],
            #         color='red', linewidth=2, linestyle='--')

        # add distance to list so that it can be added to the xarray with the properties
        smallest_size_list.append(distance_percent/conversion_factor)

        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.show()

    # Add the smallest distance to the property data array
    distance_percent_da = xr.DataArray(smallest_size_list, dims="particles")
    new_properties = xr.concat(
        [distance_percent_da],
        dim="properties"
    )
    new_properties = new_properties.assign_coords(
        properties=["smallest_dimension"]
    )

    properties_data = xr.concat([properties_data, new_properties], dim="properties")
    return properties_data

def get_largest_dimension(properties_data, image, conversion_factor):
    """Function to get the largest dimension of the particles

    Args:
        properties_data (xr.array): Data array with all the other properties of the particles
        image (np.array): The numpy array that represents the image
        conversion_factor (float): Value to convert the pixels to um 

    Returns:
        _type_: _description_
    """
    largest_dimensions= []
    labeled_image, num_labels = sk.measure.label(image, connectivity=2, return_num=True)
    props = sk.measure.regionprops(labeled_image)

    for region in sk.measure.regionprops(labeled_image):
        convex_hull = sk.morphology.convex_hull_image(region.convex_image)

        if region.area < 5:
            largest_dimensions.append(np.nan)
            continue

        perimeter_coords = np.argwhere(convex_hull)
        if len(perimeter_coords) > 0:
            perimeter_coords = np.column_stack(np.where(convex_hull))
            hull = scipy.spatial.ConvexHull(perimeter_coords)

            max_dist = 0

            for i in range(len(hull.vertices)):
                for j in range(i + 1, len(hull.vertices)):
                    p1 = perimeter_coords[hull.vertices[i]]
                    p2 = perimeter_coords[hull.vertices[j]]
                    dist = np.linalg.norm(p1 - p2)

                    # Update the max distance if this one is greater
                    if dist > max_dist:
                        max_dist = dist

            # Print the largest dimension length
            print(f"Largest Dimension Length: {max_dist:.2f}")
            largest_dimensions.append(max_dist)

    largest_dimensions = [x / conversion_factor for x in largest_dimensions]

    largest_dimensions_xr = xr.DataArray(largest_dimensions, dims="particles")
    new_properties = xr.concat(
        [largest_dimensions_xr],
        dim="properties"
    )
    new_properties = new_properties.assign_coords(
        properties=["largest_dimension"]
    )
    print(new_properties)
    print(properties_data)
    properties_data = xr.concat([properties_data, new_properties], dim="properties")

    print(properties_data.shape)
    return largest_dimensions

def get_image_size(filepath):
    """Function to read the size of the image from the given CEMIA output file
        (Properties_Filtered (IMC).txt)

    Args:
        filepath (str): The path to the CEMIA output file

    Raises:
        ValueError: File doesn't have the correct format (data not found on line expected)

    Returns:
        float: area of the image in um^2
    """
    file = open(filepath)
    file_content = file.readlines()
    area_image_line = file_content[-1]
    match = re.search(r"(\d+\.\d+|\d+)", area_image_line)
    if match:
        image_area = float(match.group(0))
        return image_area
    raise ValueError("The file provided doesn't have the correct format")

def get_area(image, conversionfactor):
    """Function to get the area of a segmentation image

    Args:
        image (np.array): Array that corresponds to an image, each value is for one pixel
        conversionfactor (float): Conversion factor (nb of pixels per um)

    Returns:
        float: Area of the image in um^2
    """
    length= np.shape(image)[0]
    width = np.shape(image)[1]
    length = length/conversionfactor
    width = width/conversionfactor
    area = length*width
    return area


def get_strain_list_center(filelist):
    """Function to get the list of strain values at the center of the sample at all rolling steps

    Args:
        filelist (list): A list with the paths to all of the files with the strain data in order

    Returns:
        list: a list of the strain values extracted from the files
    """
    strainlist = []
    for filename in filelist:
        file = pd.read_excel(
            filename, sheet_name="equ_strain", skiprows=45, header=None
        )
        center = int((file.shape[0] / 2) - 1)
        strain = file.iloc[center].max()
        strainlist.append(strain)
    return strainlist

def get_strain_list_edge(filelist):
    """Function to get the list of strain values at the edge of the sample at all rolling steps

    Args:
        filelist (list): A list with the paths to all of the files with the strain data in order

    Returns:
        list: a list of the strain values extracted from the files
    """
    strainlist = []
    for filename in filelist:
        file = pd.read_excel(
            filename, sheet_name="equ_strain", skiprows=45, header=None
        )
        edge = 0
        strain = file.iloc[edge].max()
        strainlist.append(strain)
    return strainlist


def save_to_pickle(data, filename):
    """Function to write data to pickle file

    Args:
        data (any type of data): The data that needs to be saved into a pickle file
        filename (str): A string that is the name of the file to which to save the data
    """
    filename = "pickles/"+filename
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def read_from_pickle(filename):
    """Function to import data from a pickle file

    Args:
        filename (str): Name of the pickle file containing the data

    Returns:
        any: The data contained in the pickle file
    """
    filename = "pickles/"+filename
    with open(filename, 'rb') as f:
        xarray_data = pickle.load(f)
    print(f"Xarray object loaded from {filename}")
    return xarray_data


def save_properties_csv(data, filename):
    """Function to save a particle properties xarray dataarray to a csv file

    Args:
        data (xarray): A dataarray with one dim "particles" and the other dim "properties"
        filename (str): Name of the csv file to be saved, (name without extension)
    """
    data.name = "particle_properties"
    data_df = data.to_dataframe()
    df_temp = data_df.reset_index()


    print(df_temp)
    properties_df = df_temp.pivot(index="particles", columns="properties", values= "particle_properties")
    properties_df.reset_index(inplace= True)
    properties_df.to_csv(f"CSV/{filename}.csv")


###################################################################################################
# CALCULATION FUNCTIONS
###################################################################################################


def get_stats(data):
    """Function to get some basic stats from an array containing particles

    Args:
        data (xarray): An xarray with one dimension properties (containing 'ECD' and 'Aspect Ratio')
                        and the other dimension particles

    Returns:
        tuple: The number of particles, the mean ECD, the standard deviation of the ECDs,
               the mean aspect ratio and the standard deviation of the aspect ratios
    """
    mean_ecd = data.sel(properties="ECD").mean(dim="particles").values
    std_ecd = data.sel(properties="ECD").std(dim="particles").values
    nb_particles = data.shape[1]
    print(f"number of particles : {nb_particles}")
    print(f"mean ecd : {mean_ecd}")
    print(f"standard deviation of ecd: {std_ecd}")
    mean_aspectratio = data.sel(properties="Aspect Ratio").mean(dim="particles").values
    std_aspectratio = data.sel(properties="Aspect Ratio").std(dim="particles").values
    print(f"mean aspect ratio : {mean_aspectratio}")
    print(f"standard deviation of aspect ratio : {std_aspectratio}")
    return nb_particles, mean_ecd, std_ecd, mean_aspectratio, std_aspectratio


def weibull_probability(
    ECD,
    aspect_ratio,
    equiv_strain,
    modulus=1.2,
    reference_ECD=4,
    reference_stress=16,
    shear_al=76.5,
):
    """function to calculate the weibull probability that a particle will crack

    Args:
        ECD (float): The equivalent circle diameter of the particle [μm]
        aspect_ratio (float): The aspect ratio of the particle (major length/ minor length)
        equiv_strain (float): equivalent strain, no unit
        modulus (int, optional): Weibull modulus m. Defaults to 3.
        reference_ECD (int, optional): Reference ECD value, choose value based on literature.
                                        Defaults to 4.
        reference_stress (int, optional): Reference stress value, choose value based on literature.
                                          Defaults to 16.
        shear_al (float, optional): Reference Al matrix shear modulus

    Returns:
        float: the probability that the particle will break under the specified conditions
    """
    s = 2 * 1.348442945 * 0.393 
    shear_al = 76.5  # GPa
    numerator = abs(s * aspect_ratio  * shear_al * equiv_strain)

    if numerator == 0.0:
        return 0
    denominator =  reference_stress
    weibull = 1 - np.exp(-((ECD/reference_ECD)**3)*(numerator / denominator) ** modulus)

    return weibull


weibull_probability_vec = np.vectorize(weibull_probability)
"""
Vectorized version of the Weibull_probability function, see that function's doc for more details
"""


def weibull_probas_array(
    data, equiv_strain, modulus=1.2, reference_ECD=4, reference_stress=16, shear_al=76.5
):
    """Function to calculate the weibull probabilities for an array of particles

    Args:
        data (xarray): An xarray with one dimension properties (containing 'ECD' and 'Aspect Ratio')
                        and the other dimension particles
        equiv_strain (float): Equivalent strain in the sample
        modulus (int, optional): Weibull modulus m. Defaults to 3.
        reference_ECD (int, optional): Reference ECD value, choose value based on literature.
                                        Defaults to 4.
        reference_stress (int, optional): Reference stress value, choose value based on literature.
                                          Defaults to 16.
        shear_al (float, optional): Reference Al matrix shear modulus

    Returns:
        xarray: The same xarray as the input data, with the coord 'Break proba' added to the
                properties dimension
    """
    temp_array = np.array(
            [
                weibull_probability_vec(
                    data.sel(properties="ECD").values,
                    data.sel(properties="Aspect Ratio").values,
                    equiv_strain,
                    modulus,
                    reference_ECD,
                    reference_stress,
                    shear_al,
                )
            ]
        )
    temp_array = temp_array.transpose()
    weibull_probas = xr.DataArray(
        temp_array
        ,
        dims=["particles", "properties"],
        coords={
            "properties": ["Break proba"],
        },
    )
    # merge the initial xarray with the xarray that contains the probabilities
    data_with_probas = xr.concat([data, weibull_probas], dim="properties")
    return data_with_probas


def compare_fct(proba, rand):
    """Function to compare the probability a particle will break and a randomly generated number
        to check if it will break

    Args:
        proba (float): Probability the particle will break, between 0 and 1
        rand (float): the randomly generated number

    Returns:
        int: 1 if the particle breaks, and 0 if the particle does not break
    """
    if rand <= proba:
        return 1
    else:
        return 0


compare_fct_vec = np.vectorize(compare_fct)
"""
Vectorized version of the compare_fct, see that function's doc for more details
"""


def check_if_break(
    data
):
    """Function that takes the data with the weibull probabilities calculated and determines which
        particles will break

    Args:
        data (xarray): Xarray with one dimension properties with the ECD, aspect ratios,
                        and probabilities of a break and the other dimension particles

    Returns:
        xarray: Xarray with the same format as the input data, but with a row containing random
                numbers and a row with whether the particles will break (1= break, 0= no break)
    """
    nb_particles = data.shape[0]  # the number of particles in the data array
    temp_array = np.random.rand(1, nb_particles)
    random_array = xr.DataArray(
        temp_array,
        dims=["properties", "particles"],
        coords={"properties": ["random number"]},
    )  # create new array
    data = xr.concat([data, random_array], dim="properties")
    break_data = xr.DataArray(
        np.array(
            [
                compare_fct_vec(
                    data.sel(properties="Break proba").values,
                    data.sel(properties="random number").values,
                    )
            ]
        ),
        dims=["properties", "particles"],
        coords={"properties": ["will break"]},
    )
    data_with_break = xr.concat([data, break_data], dim="properties")
    return data_with_break

### NORMAL VERSION (linear reorientation)
# def reorient_particle(orientation_deg, random_num):
#     """Function that determines if particles reorient and returns the orientation of the new particles

#     Args:
#         orientation_deg (float): orientation of the particle(s) in degrees
#         random_num (float): a random number between 0 and 1

#     Returns:
#         float: the new orientation if the particles reorient or that same orientation if it doesn't reorient
#     """
#     reorient = False
#     check_value = 90 * np.sqrt(random_num)
#     if orientation_deg > check_value:
#         reorient = True
#     if reorient:
#         neworientation = abs(orientation_deg - 0.77*orientation_deg)
#         return neworientation
#     else:
#         return orientation_deg

## Version of reorientation based on the power law, 
def reorient_particle(orientation_deg, random_num):
    """Function that determines if particles reorient and returns the orientation of the new particles

    Args:
        orientation_deg (float): orientation of the particle(s) in degrees
        random_num (float): a random number between 0 and 1

    Returns:
        float: the new orientation if the particles reorient or that same orientation if it doesn't reorient
    """
    reorient = False
    alpha = 0.4882
    check_value = np.power(random_num* (np.power(90, alpha + 1)), (1/(alpha +1)))
    if orientation_deg > check_value:
        reorient = True
    if reorient:
        neworientation = abs(orientation_deg - 0.4882*orientation_deg)
        return neworientation
    else:
        return orientation_deg
    


# def reorient_particle(orientation_deg, random_num):
#     """Function that determines if particles reorient and returns the orientation of the new particles

#     Args:
#         orientation_deg (float): orientation of the particle(s) in degrees
#         random_num (float): a random number between 0 and 1

#     Returns:
#         float: the new orientation if the particles reorient or that same orientation if it doesn't reorient
#     """
#     reorient = False
#     lambda_coeff = 0.004
#     check_value = (np.log(1 - random_num))/ (-lambda_coeff)
#     if orientation_deg > check_value:
#         reorient = True
#     if reorient:
#         neworientation = abs(orientation_deg - 0.5*orientation_deg)
#         return neworientation
#     else:
#         return orientation_deg
    
def new_dims_perfect_break(smallest_dim, major_length, n):
    """Function to determine the smallest and largest dimensions after a perfect break into n parts

    Args:
        smallest_dim (float): smallest dimension of the previous particle
        major_length (float): largest dimension of the previous particle
        n (int): number of new particles formed by breaking the particle

    Returns:
        tuple: new smallest dimension and new largest dimension
    """
    break_length = major_length/n 
    if break_length < smallest_dim:
        new_small_dim = break_length
        new_large_dim = smallest_dim
    else:
        new_small_dim = smallest_dim
        new_large_dim = break_length
    return (new_small_dim, new_large_dim)

def proba_distrib_break(largest_dim, smallest_dim,  alpha, random_num, area):
    """Function to break a particle based on a probability

    Args:
        largest_dim (float): Largest dimension of the parent particle
        smallest_dim (float): Smallest dimension of the parent particle
        alpha (float): smallest particle size
        random_num (float): random number between 0 and 1
        area (float): area of the parent particle

    Returns:
        tuple: dimensions of the two new particles formed ( largest dimension particle 1, largest dimension particle 2, smallest dimension particle 1, smallest dimension particle 2, area particle 1, area particle 2)
    """
    min_largest_dim = (np.pi * alpha**2)/(4*smallest_dim)
    
    new_part_len1 = min_largest_dim + np.sqrt(random_num) * (largest_dim/2 - min_largest_dim)
    new_part_len2 = largest_dim - new_part_len1

    proportion = new_part_len1/largest_dim
    new_area1 = proportion*area
    new_area2 = area - new_area1
    if new_part_len1 < smallest_dim:
        new_largest_dim1 = smallest_dim
        new_smallest_dim1 = new_part_len1
    else:
        new_largest_dim1 = new_part_len1
        new_smallest_dim1 = smallest_dim
    if new_part_len2 < smallest_dim:
        new_largest_dim2 = smallest_dim
        new_smallest_dim2 = new_part_len2
    else: 
        new_largest_dim2 = new_part_len2
        new_smallest_dim2 = smallest_dim
    return new_largest_dim1, new_largest_dim2, new_smallest_dim1, new_smallest_dim2, new_area1, new_area2


def probability_break_particle(
    ECD, aspect_ratio, will_break,  orientation, random_number, new_particles_array, smallest_dimension, largest_dimension, area, alpha=0.1, n=2
):
    """Function that applies a perfect break (into n equal parts) to a particle
        and adds the new particles with their properties to an array

    Args:
        ECD (float): The equivalent circle diameter of the particle (size)
        aspect_ratio (float): The aspect ratio of the particle (major length/minor length)
        will_break (int): 1 if particle breaks and 0 if particle doesn't break
        new_particles_array (xarray): array to store the new particles created ( one dimension is
            properties with coords ECD and Aspect Ratio, the other is particles)
        alpha (float, optional): Lower limit to particle ECD, if ECD < alpha -> doesn't break.
                                 Defaults to 0 (ie, no lower limit)
        n (int, optional): The number of new particles created by a break. Defaults to 2.

    Returns:
        xarray: The input new_particles_array to which the particles created by this break are added
    """

    neworientation = reorient_particle(orientation, random_number)

    if will_break == 0 or (area/2) < (np.pi * (alpha/2)*(alpha/2)):  # returns the existing particle ie. particle doesn't break
        temp_array = np.array([[ECD, aspect_ratio, neworientation, smallest_dimension, largest_dimension, area]])
        particle = xr.DataArray(
            temp_array,
            dims=[ "particles", "properties"],
            coords={"properties": ["ECD", "Aspect Ratio", "orientation_deg", "smallest_dim_area", "largest_dimension", "area"]},
        )

        new_particles_array = xr.concat(
            [new_particles_array, particle], dim="particles"
        )
        new_particles_array = new_particles_array.dropna( dim="particles")
        return new_particles_array
    
    neworientation = np.tile(neworientation, (n,1))
    new_largest_dim1, new_largest_dim2, new_smallest_dim1, new_smallest_dim2, new_area1,new_area2 = proba_distrib_break(largest_dimension, smallest_dimension, alpha, random_number, area)
    aspectratiovalue_1 = new_largest_dim1/ new_smallest_dim1
    aspectratiovalue_2 = new_largest_dim2 / new_smallest_dim2
    proportion = new_area1 / area
    # newECD_1 = proportion * ECD
    # newECD_2 = ECD - newECD_1

    newECD_1 = np.sqrt(new_area1/np.pi)
    newECD_2 = np.sqrt(new_area2/np.pi)

    newECD = np.array([[newECD_1], [newECD_2]])
    newAspectRatio = np.array([[aspectratiovalue_1], [aspectratiovalue_2]])
    new_smallest_dim = np.array([[new_smallest_dim1], [new_smallest_dim2]])
    new_largest_dim = np.array([[new_largest_dim1], [new_largest_dim2]])
    newArea = np.array([[new_area1], [new_area2]])


    # if ECD / n < alpha:
    #     particle = xr.DataArray(
    #         np.array([[ECD, aspect_ratio, orientation, smallest_dimension, largest_dimension, area]]),
    #         dims=[ "particles", "properties"],
    #         coords={"properties": ["ECD", "Aspect Ratio", "orientation_deg", "smallest_dimension", "largest_dimension", "area"]},
    #     )
    #     new_particles_array = xr.concat(
    #         [new_particles_array, particle], dim="particles"
    #     )
    #     return new_particles_array

    new_particles = xr.DataArray(
        np.concatenate((newECD, newAspectRatio, neworientation, new_smallest_dim, new_largest_dim, newArea), axis=1),
        dims=[ "particles", "properties"],
        coords={"properties": ["ECD", "Aspect Ratio", "orientation_deg", "smallest_dim_area", "largest_dimension", "area"]},
    )
    new_particles_array = xr.concat(
        [new_particles_array, new_particles], dim="particles"
    )
    new_particles_array = new_particles_array.dropna(dim="particles")
    return new_particles_array


def perfect_break_particle(
    ECD, aspect_ratio, will_break,  orientation, random_number, new_particles_array, smallest_dimension, largest_dimension, area, alpha=0, n=2
):
    """Function that applies a perfect break (into n equal parts) to a particle
        and adds the new particles with their properties to an array

    Args:
        ECD (float): The equivalent circle diameter of the particle (size)
        aspect_ratio (float): The aspect ratio of the particle (major length/minor length)
        will_break (int): 1 if particle breaks and 0 if particle doesn't break
        new_particles_array (xarray): array to store the new particles created ( one dimension is
            properties with coords ECD and Aspect Ratio, the other is particles)
        alpha (float, optional): Lower limit to particle ECD, if ECD < alpha -> doesn't break.
                                 Defaults to 0 (ie, no lower limit)
        n (int, optional): The number of new particles created by a break. Defaults to 2.

    Returns:
        xarray: The input new_particles_array to which the particles created by this break are added
    """
    neworientation = reorient_particle(orientation, random_number)


    if will_break == 0 or ECD < alpha:  # returns the existing particle ie. particle doesn't break
        temp_array = np.array([[ECD, aspect_ratio, neworientation, smallest_dimension, largest_dimension, area]])
        particle = xr.DataArray(
            temp_array,
            dims=[ "particles", "properties"],
            coords={"properties": ["ECD", "Aspect Ratio", "orientation_deg", "smallest_dim_area", "largest_dimension", "area"]},
        )

        new_particles_array = xr.concat(
            [new_particles_array, particle], dim="particles"
        )
        new_particles_array = new_particles_array.dropna( dim="particles")
        return new_particles_array
    neworientation = np.tile(neworientation, (n,1))
    newArea = np.tile(area/n, (n, 1))
    newECD = np.tile(np.sqrt((area/n)/np.pi), (n, 1))
    new_smallest_dim, new_largest_dim = new_dims_perfect_break(smallest_dimension, largest_dimension, n)
    aspectratiovalue = new_largest_dim/new_smallest_dim
    newAspectRatio = np.tile(aspectratiovalue, (n, 1))
    new_smallest_dim = np.tile(new_smallest_dim, (n, 1))
    new_largest_dim = np.tile(new_largest_dim, (n, 1))

    if ECD / n < alpha:
        particle = xr.DataArray(
            np.array([[ECD, aspect_ratio, orientation, smallest_dimension, largest_dimension, area]]),
            dims=[ "particles", "properties"],
            coords={"properties": ["ECD", "Aspect Ratio", "orientation_deg", "smallest_dim_area", "largest_dimension", "area"]},
        )
        new_particles_array = xr.concat(
            [new_particles_array, particle], dim="particles"
        )
        new_particles_array = new_particles_array.dropna( dim="particles")
        return new_particles_array
    new_particles = xr.DataArray(
        np.concatenate((newECD, newAspectRatio, neworientation, new_smallest_dim, new_largest_dim, newArea), axis=1),
        dims=[ "particles", "properties"],
        coords={"properties": ["ECD", "Aspect Ratio", "orientation_deg", "smallest_dim_area", "largest_dimension", "area"]},
    )
    new_particles_array = xr.concat(
        [new_particles_array, new_particles], dim="particles"
    )

    new_particles_array = new_particles_array.dropna( dim="particles")
    return new_particles_array


def apply_perfect_break(data_with_break, alpha=0, n=2):
    """Function that applies the perfect break to all of the particles in the data

    Args:
        data_with_break (xarray): Xarray with one dimension properties with the ECD, aspect ratios,
                                    probabilities of a break and will break (1=break, 0=no break)
                                    and the other dimension particles
        alpha (float, optional): Lower limit to particle ECD, if ECD < alpha -> doesn't break.
                                 Defaults to 0 (ie, no lower limit)
        n (int, optional): The number of new particles created by a break. Defaults to 2.
    Returns:
        xarray: An array containing all of the particles in the sample after applying the break
    """
    ## Will need to change this  when looking at reattributing the different properties at each step (shape of array, coords and the perfect_break_particles functions)
    empty_array = np.full((2, 6), np.nan)

    new_particles_array = xr.DataArray(
        empty_array,
        dims=["particles", "properties"],
        coords={"properties": ["ECD", "Aspect Ratio", "orientation_deg", "smallest_dim_area", "largest_dimension", "area"]},
    )

    for i in range(data_with_break.shape[0]):
        new_particles_array = perfect_break_particle(
            data_with_break.sel(properties="ECD").values[i],
            data_with_break.sel(properties="Aspect Ratio").values[i],
            data_with_break.sel(properties="will break").values[i],
            data_with_break.sel(properties="orientation_deg").values[i],
            data_with_break.sel(properties="random number").values[i],
            new_particles_array,
            data_with_break.sel(properties="smallest_dim_area").values[i], 
            data_with_break.sel(properties="largest_dimension").values[i],
            data_with_break.sel(properties="area").values[i],
            alpha,
            n,
        )
    return new_particles_array


def apply_probability_break(data_with_break, alpha=0.1, n=2):
    """Function that applies the probability break to all of the particles in the data

    Args:
        data_with_break (xarray): Xarray with one dimension properties with the ECD, aspect ratios,
                                    probabilities of a break and will break (1=break, 0=no break)
                                    and the other dimension particles
        alpha (float, optional): Lower limit of particle ECD, 
                                 Defaults to 0.1 
        n (int, optional): The number of new particles created by a break. Defaults to 2.
    Returns:
        xarray: An array containing all of the particles in the sample after applying the break
    """
    ## Will need to change this  when looking at reattributing the different properties at each step (shape of array, coords and the perfect_break_particles functions)
    empty_array = np.full((2, 6), np.nan)

    new_particles_array = xr.DataArray(
        empty_array,
        dims=["particles", "properties"],
        coords={"properties": ["ECD", "Aspect Ratio", "orientation_deg", "smallest_dim_area", "largest_dimension", "area"]},
    )

    for i in range(data_with_break.shape[0]):
        new_particles_array = probability_break_particle(
            data_with_break.sel(properties="ECD").values[i],
            data_with_break.sel(properties="Aspect Ratio").values[i],
            data_with_break.sel(properties="will break").values[i],
            data_with_break.sel(properties="orientation_deg").values[i],
            data_with_break.sel(properties="random number").values[i],
            new_particles_array,
            data_with_break.sel(properties="smallest_dim_area").values[i], 
            data_with_break.sel(properties="largest_dimension").values[i],
            data_with_break.sel(properties="area").values[i],
            alpha,
            n,
        )
    return new_particles_array



###################################################################################################
# FULL RUN FUNCTIONS
###################################################################################################


def perfectbreak_1step(
    data,
    n,
    alpha,
    equiv_strain,
    weibull_modulus=1.2,
    reference_ecd=4,
    reference_stress=16,
    shear_al=76.5,
):
    """Function to run all the steps of the perfect break model in one step

    Args:
        data (xarray): An xarray with one dimension "properties" containing at least the coords
                        'ECD' and 'Aspect Ratio' and the other dimension "particles"
        n (int): The number of new particles formed by breaking a particle
        alpha (float): lower limit of particle size
        equiv_strain (float): Equivalent strain in the sample
        weibull_modulus (float, optional): Weibull modulus. Defaults to 1.2.
        reference_ecd (float, optional): Reference value of the ECD. Defaults to 4.
        reference_stress (float, optional): Reference stress [GPa]. Defaults to 16.
        shear_al (float, optional): Shear modulus of the aluminum matrix. Defaults to 76.5.

    Returns:
        xarray: An xarray with the same format as the input array containing the particles after
                the breaking process
    """
    data_with_probas = weibull_probas_array(
        data, equiv_strain, weibull_modulus, reference_ecd, reference_stress, shear_al
    )
    data_with_break = check_if_break(data_with_probas)
    new_particles = apply_perfect_break(data_with_break, alpha, n)
    return new_particles


def probabilitybreak_1step(
    data,
    n,
    alpha,
    equiv_strain,
    weibull_modulus=1.2,
    reference_ecd=4,
    reference_stress=16,
    shear_al=76.5,
):
    """Function to run all the steps of the probability break model in one step

    Args:
        data (xarray): An xarray with one dimension "properties" containing at least the coords
                        'ECD' and 'Aspect Ratio' and the other dimension "particles"
        n (int): The number of new particles formed by breaking a particle
        alpha (float): lower limit of particle size
        equiv_strain (float): Equivalent strain in the sample
        weibull_modulus (float, optional): Weibull modulus. Defaults to 1.2.
        reference_ecd (float, optional): Reference value of the ECD. Defaults to 4.
        reference_stress (float, optional): Reference stress [GPa]. Defaults to 16.
        shear_al (float, optional): Shear modulus of the aluminum matrix. Defaults to 76.5.

    Returns:
        xarray: An xarray with the same format as the input array containing the particles after
                the breaking process
    """
    data_with_probas = weibull_probas_array(
        data, equiv_strain, weibull_modulus, reference_ecd, reference_stress, shear_al
    )
    data_with_break = check_if_break(data_with_probas)

    new_particles = apply_probability_break(data_with_break, alpha, n)
    return new_particles, data_with_break




def probabilitybreak_multistep_substeps(
    data,
    nb_steps,
    n,
    alpha,
    equiv_strains,
    substeps = 2,
    weibull_modulus=1.2,
    reference_ecd=4,
    reference_stress=16,
    shear_al=76.5,
    return_intermediate=False,
):
    """Function to run the perfect break multistep model

    Args:
        data (xarray): An xarray with one dimension "properties" containing at least the coords
                        'ECD' and 'Aspect Ratio' and the other dimension "particles"
        nb_steps (int): The number of perfect break steps to run
        n (int): The number of new particles formed by breaking a particle
        alpha (float): Lower limit of particle size
        equiv_strains (array): An array containing the equivalent strain for each step in order
        substeps(int, optional): Number of breaking steps in each rolling step. Defaults to 2
        weibull_modulus (float, optional): Weibull modulus. Defaults to 1.2.
        reference_ecd (float, optional): Reference value of the ECD. Defaults to 4.
        reference_stress (float, optional): Reference stress [GPa]. Defaults to 16.
        shear_al (float, optional): Shear modulus of the aluminum matrix. Defaults to 76.5.

    Returns:
        xarray: An xarray with the same format as the input array containing the particles after
                the breaking process
    """
    new_strain_list = [item for item in equiv_strains for x in range(substeps)]
    
    num_steps = len(new_strain_list)
    rows_with_zero = (data == 0).any(dim="properties")
    output_data_list = []
    output_data_probas_list = []
    data = data.where(~rows_with_zero, drop=True)
    for i in range(num_steps):
        print(f"data shape start of step {i}", data.shape)
        strain = new_strain_list[i]
        print("number of runs through perfect break", i)
        data, data_proba = probabilitybreak_1step(
            data,
            n,
            alpha,
            strain,
            weibull_modulus,
            reference_ecd,
            reference_stress,
            shear_al,
        )
        output_data_list.append(data)
        output_data_probas_list.append(data_proba)
    if return_intermediate is True:
        return output_data_list
    return data

# def perfectbreak_multistep(
#     data,
#     nb_steps,
#     n,
#     alpha,
#     equiv_strains,
#     weibull_modulus=1.2,
#     reference_ecd=4,
#     reference_stress=16,
#     shear_al=76.5,
#     return_intermediate=False
# ):
#     """Function to run the perfect break multistep model

#     Args:
#         data (xarray): An xarray with one dimension "properties" containing at least the coords
#                         'ECD' and 'Aspect Ratio' and the other dimension "particles"
#         nb_steps (int): The number of perfect break steps to run
#         n (int): The number of new particles formed by breaking a particle
#         alpha (float): Lower limit of particle size
#         equiv_strains (array): An array containing the equivalent strain for each step in order
#         weibull_modulus (float, optional): Weibull modulus. Defaults to 1.2.
#         reference_ecd (float, optional): Reference value of the ECD. Defaults to 4.
#         reference_stress (float, optional): Reference stress [GPa]. Defaults to 16.
#         shear_al (float, optional): Shear modulus of the aluminum matrix. Defaults to 76.5.

#     Returns:
#         xarray: An xarray with the same format as the input array containing the particles after
#                 the breaking process
#     """
#     if len(equiv_strains) < nb_steps:
#         raise ValueError(
#             "There are fewer equivalent strain values than the number of steps"
#         )
#     output_data_list = []
#     for i in range(nb_steps):
#         strain = equiv_strains[i]
#         data = perfectbreak_1step(
#             data,
#             n,
#             alpha,
#             strain,
#             weibull_modulus,
#             reference_ecd,
#             reference_stress,
#             shear_al,
#         )
#         output_data_list.append(data)
#     if return_intermediate is True:
#         return output_data_list
#     return data


def perfectbreak_multistep(
    data,
    nb_steps,
    n,
    alpha,
    equiv_strains,
    weibull_modulus=1.2,
    reference_ecd=4,
    reference_stress=16,
    shear_al=76.5,
    return_intermediate=False,
):
    """Function to run the perfect break multistep model

    Args:
        data (xarray): An xarray with one dimension "properties" containing at least the coords
                        'ECD' and 'Aspect Ratio' and the other dimension "particles"
        nb_steps (int): The number of perfect break steps to run
        n (int): The number of new particles formed by breaking a particle
        alpha (float): Lower limit of particle size
        equiv_strains (array): An array containing the equivalent strain for each step in order
        weibull_modulus (float, optional): Weibull modulus. Defaults to 1.2.
        reference_ecd (float, optional): Reference value of the ECD. Defaults to 4.
        reference_stress (float, optional): Reference stress [GPa]. Defaults to 16.
        shear_al (float, optional): Shear modulus of the aluminum matrix. Defaults to 76.5.

    Returns:
        xarray: An xarray with the same format as the input array containing the particles after
                the breaking process
    """
    if len(equiv_strains) < nb_steps:
        raise ValueError(
            "There are fewer equivalent strain values than the number of steps"
        )
    rows_with_zero = (data == 0).any(dim="properties")
    output_data_list = []
    data = data.where(~rows_with_zero, drop=True)
    for i in range(nb_steps):
        print(f"data shape start of step {i}", data.shape)
        strain = equiv_strains[i]
        print("number of runs through perfect break", i)
        data = perfectbreak_1step(
            data,
            n,
            alpha,
            strain,
            weibull_modulus,
            reference_ecd,
            reference_stress,
            shear_al,
        )
        output_data_list.append(data)
    if return_intermediate is True:
        return output_data_list
    return data


def probabilitybreak_multistep(
    data,
    nb_steps,
    n,
    alpha,
    equiv_strains,
    weibull_modulus=1.2,
    reference_ecd=4,
    reference_stress=16,
    shear_al=76.5,
    return_intermediate=False,
):
    """Function to run the perfect break multistep model

    Args:
        data (xarray): An xarray with one dimension "properties" containing at least the coords
                        'ECD' and 'Aspect Ratio' and the other dimension "particles"
        nb_steps (int): The number of perfect break steps to run
        n (int): The number of new particles formed by breaking a particle
        alpha (float): Lower limit of particle size
        equiv_strains (array): An array containing the equivalent strain for each step in order
        weibull_modulus (float, optional): Weibull modulus. Defaults to 1.2.
        reference_ecd (float, optional): Reference value of the ECD. Defaults to 4.
        reference_stress (float, optional): Reference stress [GPa]. Defaults to 16.
        shear_al (float, optional): Shear modulus of the aluminum matrix. Defaults to 76.5.

    Returns:
        xarray: An xarray with the same format as the input array containing the particles after
                the breaking process
    """
    if len(equiv_strains) < nb_steps:
        raise ValueError(
            "There are fewer equivalent strain values than the number of steps"
        )
    rows_with_zero = (data == 0).any(dim="properties")
    output_data_list = []
    output_data_probas_list = []
    data = data.where(~rows_with_zero, drop=True)
    for i in range(nb_steps):
        print(f"data shape start of step {i}", data.shape)
        strain = equiv_strains[i]
        print("number of runs through probability break", i)
        data, data_proba = probabilitybreak_1step(
            data,
            n,
            alpha,
            strain,
            weibull_modulus,
            reference_ecd,
            reference_stress,
            shear_al,
        )
        output_data_list.append(data)
        output_data_probas_list.append(data_proba)
    if return_intermediate is True:
        return output_data_list, output_data_probas_list
    return data



def multirun_full_code_probabreak(data,nb_steps,n,alpha,equiv_strains,nb_runs,weibull_modulus=1.2,reference_ecd=4,reference_stress=16,shear_al=76.5,return_intermediate=False,):
    """Function to run the probability break function multiple times on the same starting data

    Args:
        data (xarray): An x array with one dimension properties and the other dimension particles
        nb_steps (int): number of break steps for each run
        n (int): number of particles created by breaking a particle once
        alpha (float): smallest ECD a particle can have
        equiv_strains (list): list of the equivalent strain at every rolling step   
        nb_runs (int): number of times to 
        weibull_modulus (float, optional): weibull modulus. Defaults to 1.2.
        reference_ecd (float, optional): reference ecd. Defaults to 4.
        reference_stress (float, optional): reference stress. Defaults to 16.
        shear_al (float, optional): shear modulus of aluminum. Defaults to 76.5.
        return_intermediate (bool, optional): return the results at all intermediate steps. Defaults to False.

    Returns:
        tuple: List containing one array with the data for each break step of all the runs combined and  list of the probability data in the same format
    """
    list_output_data = []
    list_output_probas = []

    for i in range(nb_runs):
        if return_intermediate is True:
            output_data, output_probas = probabilitybreak_multistep(data, nb_steps, n, alpha, equiv_strains, weibull_modulus, reference_ecd, reference_stress, shear_al, return_intermediate)
            list_output_data.append(output_data)
            list_output_probas.append(output_probas)
        else:
            output_data = probabilitybreak_multistep(data, nb_steps, n, alpha, equiv_strains, weibull_modulus, reference_ecd, reference_stress, shear_al, return_intermediate)
            list_output_data.append(output_data)
        print(f"completed {i+1} full runs ")    

    transposed = list(zip(*list_output_data))

    # Concatenate arrays at the same index across the sub-lists
    list_output_data = [xr.concat([arr for arr in step], dim="particles") for step in transposed]

    transposed= list(zip(*list_output_probas))
    list_output_probas = [xr.concat([arr for arr in step], dim = "particles") for step in transposed]
    
    if return_intermediate is True:
        return (list_output_data, list_output_probas)
    else:
        return output_data
        


###################################################################################################
# GRAPHING FUNCTIONS
###################################################################################################


# def get_hist(data, name):
#     """Function that takes the particle data and bins it (bins of size 0.1μm)

#     Args:
#         data (xarray): xarray with one dimension properties containing at least the
#                     coords "name" (eg. 'ECD' or 'Aspect Ratio') and the other dimension particles
#         name (str): The name of the property to bin the particles along

#     Returns:
#         tuple: an array with the bin edges and an array with the histogram
#     """
#     maxvalue = float(data.sel(properties=name).max(dim="particles"))
#     print(maxvalue)
#     nb_bins = 30
#     bin_edges = np.linspace(0, int(np.ceil(maxvalue)), nb_bins)
#     bin_indices = np.digitize(data.sel(properties=name).values.flatten(), bin_edges)
#     hist = np.bincount(bin_indices)
#     while (
#         len(bin_edges) > len(hist) + 1
#     ):  # to make nb of edges equal to  the number of bins
#         bin_edges = bin_edges[:-1]
#     return (bin_edges, hist)




def get_hist(data, name, bin_width=0.1):
    """Function that takes the particle data and bins it with a specified bin width.

    Args:
        data (xarray): xarray with one dimension properties containing at least the
                        coords "name" (eg. 'ECD' or 'Aspect Ratio') and the other dimension particles
        name (str): The name of the property to bin the particles along
        bin_width (float): The width of each bin in the same units as the data (default is 0.1 μm)

    Returns:
        tuple: an array with the bin edges and an array with the histogram
    """

    # # Center
    if name == "orientation_deg":
        bin_width = 2
    if name == "smallest_dimension":
        bin_width = 0.1
    if name == "area":
        bin_width = 0.5
    if name == "largest_dimension":
        bin_width = 1
    if name == "ECD":
        bin_width = 0.4
    if name == "Aspect Ratio":
        bin_width = 1.5

    # Edge
    # if name == "orientation_deg":
    #     bin_width = 2
    # if name == "smallest_dim_area":
    #     bin_width = 0.25
    # if name == "area":
    #     bin_width = 1
    # if name == "largest_dimension":
    #     bin_width = 0.4
    # if name == "ECD":
    #     bin_width = 0.3
    # if name == "Aspect Ratio":
    #     bin_width = 1

    # Get the maximum value of the specified property (name) across all particles
    maxvalue = float(data.sel(properties=name).max(dim="particles"))
    minvalue = 0
    nb_bins = int(np.ceil((maxvalue - minvalue) / bin_width))
    bin_edges = np.linspace(minvalue, maxvalue, nb_bins + 1)
    bin_indices = np.digitize(data.sel(properties=name).values.flatten(), bin_edges)
    hist = np.bincount(bin_indices, minlength=len(bin_edges))
    return bin_edges, hist


def ecd_fit(sim_data, exp_data, sim_area, exp_area):
    """Function to calculate a difference value to compare experimental and simulation results

    Args:
        sim_data (xarray): xarray containing the simulation results
        exp_data (xarray): xarray containing the experimental results
        sim_area (float): the area of the sample corresponding to simulation
        exp_area (float): the area of the sample corresponding to the experimental results

    Returns:
        float: Value that measures the difference between the simulated and the experimental results
    """
    bin_width = 0.4
    
    maxvalue_sim = float(sim_data.sel(properties="ECD").max(dim="particles"))
    minvalue_sim = float(sim_data.sel(properties="ECD").min(dim="particles"))
    maxvalue_exp = float(exp_data.sel(properties="ECD").max(dim="particles"))
    minvalue_exp = float(exp_data.sel(properties="ECD").min(dim="particles"))
    
    minvalue = np.min([minvalue_exp, minvalue_sim])
    maxvalue = np.max([maxvalue_exp, maxvalue_sim])
    
    nb_bins = int(np.ceil((maxvalue - minvalue) / bin_width))
    bin_edges = np.linspace(minvalue, maxvalue, nb_bins + 1)
    
    bin_indices_sim = np.digitize(sim_data.sel(properties="ECD").values.flatten(), bin_edges) - 1
    bin_indices_exp = np.digitize(exp_data.sel(properties="ECD").values.flatten(), bin_edges) - 1
    bin_indices_sim = np.clip(bin_indices_sim, 0, nb_bins - 1)
    bin_indices_exp = np.clip(bin_indices_exp, 0, nb_bins - 1)
    
    hist_sim = np.bincount(bin_indices_sim, minlength=nb_bins)
    hist_exp = np.bincount(bin_indices_exp, minlength=nb_bins)

    hist_sim = hist_sim/ sim_area
    hist_exp = hist_exp/ exp_area
    
    abs_diff = np.abs(hist_sim - hist_exp)
    diff_value = np.sum(abs_diff)
    return diff_value
        

def orientation_fit(sim_data, exp_data, sim_area, exp_area):
    """Function to calculate a difference value to compare experimental and simulation results

    Args:
        sim_data (xarray): xarray containing the simulation results
        exp_data (xarray): xarray contianing the experimental results
        sim_area (float): the area of the sample corresponding to simulation
        exp_area (float): the area of the sample corresponding to the experimental results

    Returns:
        float: Value that measures the difference between the simulated and experimental results
    """
    bin_width = 0.4
    
    maxvalue_sim = float(sim_data.sel(properties="orientation_deg").max(dim="particles"))
    minvalue_sim = float(sim_data.sel(properties="orientation_deg").min(dim="particles"))
    maxvalue_exp = float(exp_data.sel(properties="orientation_deg").max(dim="particles"))
    minvalue_exp = float(exp_data.sel(properties="orientation_deg").min(dim="particles"))
    
    minvalue = np.min([minvalue_exp, minvalue_sim])
    maxvalue = np.max([maxvalue_exp, maxvalue_sim])
    
    nb_bins = int(np.ceil((maxvalue - minvalue) / bin_width))
    bin_edges = np.linspace(minvalue, maxvalue, nb_bins + 1)
    
    bin_indices_sim = np.digitize(sim_data.sel(properties="orientation_deg").values.flatten(), bin_edges) - 1
    bin_indices_exp = np.digitize(exp_data.sel(properties="orientation_deg").values.flatten(), bin_edges) - 1
    bin_indices_sim = np.clip(bin_indices_sim, 0, nb_bins - 1)
    bin_indices_exp = np.clip(bin_indices_exp, 0, nb_bins - 1)
    
    hist_sim = np.bincount(bin_indices_sim, minlength=nb_bins)
    hist_exp = np.bincount(bin_indices_exp, minlength=nb_bins)

    hist_sim = hist_sim/ sim_area
    hist_exp = hist_exp/ exp_area
    
    abs_diff = np.abs(hist_sim - hist_exp)
    diff_value = np.sum(abs_diff)
    return diff_value



def plot_hist_ECD(data, image_area, title=" ", norm=False):
    """Function that plots a histogram of the ECDs of the particles

    Args:
        data (xarray): _description_
        image_area (float): Area of the sample in um^2
        title (str, optional): Title of the graph, Use to identify sample. Defaults to " ".
        norm (bool, optional): False to have the data in number of particles,
                                True to have in particle density. Defaults to False.
    """
    bin_edges, hist = get_hist(data, "ECD")
    if norm is False:
        plt.bar(
            bin_edges[:-1],
            hist,
            width=np.diff(bin_edges),
            align="edge",
            edgecolor="black",
        )
        plt.ylabel("Number particles")
        # plt.ylim([0, 400])
    else:
        hist_norm = hist / image_area
        plt.bar(
            bin_edges[:-1],
            hist_norm,
            width=np.diff(bin_edges),
            align="edge",
            edgecolor="black",
        )
        plt.ylabel("Particle density #/μm^2")
        # plt.ylim([0, 0.002])

    mean = data.sel(properties="ECD").mean(dim="particles")
    std = data.sel(properties="ECD").std(dim="particles")
    nb_particles = hist.sum()

    print(f"mean:{float(mean)}, std:{float(std)}")
    plt.text(
        0.96,
        0.94,
        f"Mean = {mean:.2f}\nStd = {std:.2f}\n# particles = {nb_particles}",
        ha="right",
        va="top",
        transform=plt.gca().transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round,pad=1"},
    )

    plt.xlim([0, 20])

    plt.xlabel("ECD Value")
    plt.title(f"Histogram of ECD Values   {title}")
    plt.grid(True)
    plt.show()


def plot_particles_properties(data_xarray, property, propertyname, nb_bins=20):
    """Function to plot the histogram of a property of the particles 

    Args:
        data_xarray (xarray): Xarray with the properties of a particle
        property (_type_): _description_
        propertyname (_type_): _description_
        nb_bins (int, optional): _description_. Defaults to 20.
    """
    fig, ax = plt.subplots()
    ax.hist(data_xarray.sel(properties =property), bins = nb_bins)
    ax.set_title(f"Histogram of the {propertyname}")
    ax.set_xlabel(f"{propertyname}")
    ax.set_ylabel("number of particles")
    plt.show()

def plot_particles_properties_normalized(data_xarray, property, propertyname, area, title, nb_bins=20):
    """Function to plot the normalized histogram of a given particle property

    Args:
        data_xarray (xarray): Xarray data array containing the properties of the particles
        property (str): Name of the property to be plot (as titled in the array)
        propertyname (str): Name of the property as indicated on the x axis of the graph
        area (float): Area of the image corresponding to the particles
        nb_bins (int, optional): Number of bins in the histogram. Defaults to 20.
    """
    property_data = data_xarray.sel(properties=property)

    fig, ax = plt.subplots()
    counts, bin_edges = np.histogram(property_data, bins=nb_bins)
    


    bin_widths = bin_edges[1:] - bin_edges[:-1]
    
    # Normalize counts: divide by area and by bin width to get counts per unit area
    normalized_counts = counts / (area*bin_widths)
    
    # Re-plot the normalized histogram using ax.bar() to correctly handle widths
    ax.bar(bin_edges[:-1], normalized_counts, width=bin_widths, align='edge', edgecolor='black')
    
    
    # Re-plot the normalized histogram
    ax.bar(bin_edges[:-1], normalized_counts, width=bin_widths, align='edge',color="green", edgecolor='black')
    

    ax.set_title(title)
    ax.set_xlabel(f"{propertyname}")
    ax.set_ylabel("number of particles per um^2")
    
    exp_mean = data_xarray.sel(properties=property).mean()
    exp_std = data_xarray.sel(properties=property).std()
    exp_total_num = data_xarray.sel(properties=property).sum()

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.58, 0.97, f"Mean {exp_mean:.4f} \n Standard deviation {exp_std:.4f} \n Total nb density {exp_total_num:.4f} ", transform=ax.transAxes, fontsize=10,
        verticalalignment='top',   horizontalalignment='left', bbox=props)
    plt.show()

    # Print the results under the plot
    print(f"Experimental data (property: {property}):")
    print(f"  Average: {exp_mean:.4f}")
    print(f"  Standard Deviation: {exp_std:.4f}")
    print(f"  Total Number Density: {exp_total_num:.4f}")
    print()


def plot_sim_exp_ecd(sim_data, exp_data, sim_image_area, exp_image_area, title=" "):
    """Function to plot a histogram of the experimental ECD data and an overlay of the
    experimental results

    Args:
        sim_data (xarray): Xarray with the particles after running the simulation
        exp_data (xarray): X array with the data for the particles after hot rolling
        sim_image_area (float): area of the image used to calculated the simulated particles, um^2
        exp_image_area (float): area of the experimental image
        title (str, optional): Title for this graph. Defaults to " ".
    """
    sim_bin_edge, sim_hist = get_hist(sim_data, "ECD")
    exp_bin_edge, exp_hist = get_hist(exp_data, "ECD")
    sim_hist_norm = sim_hist / sim_image_area
    exp_hist_norm = exp_hist / exp_image_area
    plt.bar(
        exp_bin_edge[:-1],
        exp_hist_norm,
        width=np.diff(exp_bin_edge),
        align="edge",
        edgecolor="black",
    )
    plt.ylabel("Particle density #/μm^2")
    if sim_bin_edge.shape[0] == sim_hist.shape[0]:
        plt.plot(sim_bin_edge[:], sim_hist_norm[:], color="red", linewidth=1)
    if sim_bin_edge.shape[0] == sim_hist.shape[0] + 1:
        plt.plot(sim_bin_edge[:-1], sim_hist_norm[:], color="red", linewidth=1)
    plt.xlabel("ECD value")
    plt.title(f"Histogram of ECD Values  {title}")
    plt.grid(True)
    plt.show()


def plot_sim_exp_aspectratio(
    sim_data, exp_data, sim_image_area, exp_image_area, title=" "
):
    """Function to plot a histogram of the experimental aspect ratio data and an overlay of the
    simulated results

    Args:
        sim_data (xarray): Xarray with the particles after running the simulation
        exp_data (xarray): X array with the data for the particles after hot rolling
        sim_image_area (float): area of the image used to calculated the simulated particles, um^2
        exp_image_area (float): area of the experimental image
        title (str, optional): Title for this graph. Defaults to " ".
    """
    sim_bin_edge, sim_hist = get_hist(sim_data, "Aspect Ratio")
    exp_bin_edge, exp_hist = get_hist(exp_data, "Aspect Ratio")
    sim_hist_norm = sim_hist / sim_image_area
    exp_hist_norm = exp_hist / exp_image_area
    plt.bar(
        exp_bin_edge[:-1],
        exp_hist_norm,
        width=np.diff(exp_bin_edge),
        align="edge",
        edgecolor="black",
    )
    plt.ylabel("Particle density #/μm^2")
    if sim_bin_edge.shape[0] == sim_hist.shape[0]:
        plt.plot(sim_bin_edge[:], sim_hist_norm[:], color="red", linewidth=1)
    if sim_bin_edge.shape[0] == sim_hist.shape[0] + 1:
        plt.plot(sim_bin_edge[:-1], sim_hist_norm[:], color="red", linewidth=1)
    plt.xlabel("Aspect ratio value")
    plt.title(f"Histogram of Aspect ratio Values  {title}")
    plt.grid(True)
    plt.show()


def plot_intermetallic_split(labeled_image, properties_array, title):
    """Function to show the split of intermetallics 

    Args:
        labeled_image (np.array): Array corresponding to the labeled image
        properties_array (np.array): Array with all of the properties for the particles as produced by the function to characterize the particles
    """
    output_image = np.zeros((labeled_image.shape[0], labeled_image.shape[1], 3))
    for i in range(labeled_image.shape[0]): # interate over all the pixels in the image
        for j in range(labeled_image.shape[1]):
            label = labeled_image[i, j]
            if label != 0:  
                intermetallic_value = properties_array[label-1,15]
                if intermetallic_value == 1:
                    output_image[i, j] = [1, 0, 0]  
                elif intermetallic_value == 2:
                    output_image[i, j] = [0, 1, 0]

    plt.imshow(output_image)
    plt.title(f"Intermetallics split {title}")
    plt.text(400, 400, 'α = red, β = green', dict(size=12), color='white')
    plt.show()

def plot_sim_exp(sim_data, exp_data, sim_image_area, exp_image_area, property, x_axis_text, title=" "):
    """Function to plot experimental and simulated data together

    Args:
        sim_data (np.array): simulated data
        exp_data (np.array): Experimental data
        sim_image_area (float): area of the simulation data
        exp_image_area (float): area of the experimental data
        property (str): name of the property in the array to plot
        x_axis_text (str): text to use as label of the x axis 
        title (str, optional): Title for the graph. Defaults to " ".
    """
    sim_bin_edge, sim_hist = get_hist(sim_data, property)
    exp_bin_edge, exp_hist = get_hist(exp_data, property)

    sim_hist_norm = sim_hist / sim_image_area
    exp_hist_norm = exp_hist / exp_image_area

    if len(exp_bin_edge) > len(exp_hist_norm):
        exp_bin_edge = exp_bin_edge[:-1]  
    elif len(exp_hist_norm) > len(exp_bin_edge):
        exp_hist_norm = exp_hist_norm[:-1] 
    if len(sim_bin_edge) > len(sim_hist_norm):
        sim_bin_edge = sim_bin_edge[:-1]  
    elif len(sim_hist_norm) > len(sim_bin_edge):
        sim_hist_norm = sim_hist_norm[:-1]  



    x_lim = 10
    y_lim = 10

    # For center
    if property == "orientation_deg":
        x_lim = 90
        y_lim = 0.001
    if property == "smallest_dimension":
        x_lim = 5.5
        y_lim = 0.0009
    if property == "area":
        x_lim = 80
        y_lim = 0.003
    if property == "smallest_dim_area":
        x_lim = 5.5
        y_lim = 0.0009
    if property == "largest_dimension":
        x_lim = 30
        y_lim = 0.0021
    if property == "ECD":
        x_lim = 13
        y_lim = 0.0025
    if property == "Aspect Ratio":
        x_lim = 200
        y_lim = 0.005

    # For edge
    # if property == "orientation_deg":
    #     x_lim = 91
    #     y_lim = 0.002
    # if property == "smallest_dimension":
    #     x_lim = 15
    #     y_lim = 0.002
    # if property == "area":
    #     x_lim = 40
    #     y_lim = 0.006
    # if property == "smallest_dim_area":
    #     x_lim = 15
    #     y_lim = 0.004
    # if property == "largest_dimension":
    #     x_lim = 15
    #     y_lim = 0.003
    # if property == "ECD":
    #     x_lim = 8
    #     y_lim = 0.003
    # if property == "Aspect Ratio":
    #     x_lim = 30
    #     y_lim = 0.005

    fig, ax = plt.subplots()
    ax.bar(
        exp_bin_edge[:-1],
        exp_hist_norm[:-1],
        width=np.diff(exp_bin_edge),
        align="edge",
        edgecolor="black",
        label= "experimental"
    )

    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)

    bin_width = sim_bin_edge[1] - sim_bin_edge[0]

    ax.plot(sim_bin_edge + 0.5*bin_width, sim_hist_norm, color="red", linewidth=1, label="simulation")

    ax.set_xlabel(x_axis_text, fontsize=12)
    ax.set_ylabel("Particle density #/μm²  ", fontsize=12)
    ax.set_title(title, fontsize=18)

    exp_mean = exp_data.sel(properties=property).mean()
    exp_std = exp_data.sel(properties=property).std()
    # exp_total_num = exp_data.sel(properties=property).sum()
    exp_total_num = (np.shape(exp_data)[0])/exp_image_area

    # Calculate and print statistics for simulated data
    sim_mean = sim_data.sel(properties=property).mean()
    sim_std = sim_data.sel(properties=property).std()
    # sim_total_num = sim_data.sel(properties=property).sum()
    sim_total_num = np.shape(sim_data)[0]/sim_image_area
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.58, 0.97, f"Experimental data: \n Mean {exp_mean:.4f} \n Standard deviation {exp_std:.4f} \n Total nb density {exp_total_num:.4f} \n Simulation data: \n Mean {sim_mean:.4f} \n Standard deviation {sim_std:.4f} \n Total nb density {sim_total_num:.4f} ", transform=ax.transAxes, fontsize=10,
        verticalalignment='top',   horizontalalignment='left', bbox=props)

    ax.legend()
    # Print the results under the plot
    print(f"Experimental data (property: {property}):")
    print(f"  Average: {exp_mean:.4f}")
    print(f"  Standard Deviation: {exp_std:.4f}")
    print(f"  Total Number Density: {int(exp_total_num)}")
    print()
    
    print(f"Simulated data (property: {property}):")
    print(f"  Average: {sim_mean:.4f}")
    print(f"  Standard Deviation: {sim_std:.4f}")
    print(f"  Total Number Density: {int(sim_total_num)}")
    
    # Show the plot
    plt.show()


def plot_sim(sim_data, sim_image_area, property, x_axis_text, title=" "):
    """Function to plot experimental and simulated data together

    Args:
        sim_data (np.array): simulated data
        sim_image_area (float): area of the simulation data
        property (str): name of the property in the array to plot
        x_axis_text (str): text to use as label of the x axis 
        title (str, optional): Title for the graph. Defaults to " ".
    """
    sim_bin_edge, sim_hist = get_hist(sim_data, property)

    sim_hist_norm = sim_hist / sim_image_area

    if len(sim_bin_edge) > len(sim_hist_norm):
        sim_bin_edge = sim_bin_edge[:-1]  
    elif len(sim_hist_norm) > len(sim_bin_edge):
        sim_hist_norm = sim_hist_norm[:-1]  

    x_lim = 10
    y_lim = 10
    if property == "orientation_deg":
        x_lim = 90
        y_lim = 0.001
    if property == "smallest_dimension":
        x_lim = 5.5
        y_lim = 0.0009
    if property == "smallest_dim_area":
        x_lim = 5.5
        y_lim = 0.0009
    if property == "area":
        x_lim = 150
        y_lim = 0.003
    if property == "largest_dimension":
        x_lim = 80
        y_lim = 0.0025
    if property == "ECD":
        x_lim = 13
        y_lim = 0.003
    if property == "Aspect Ratio":
        x_lim = 100
        y_lim = 0.005


    fig, ax = plt.subplots()

    bin_width = sim_bin_edge[1] - sim_bin_edge[0]

    ax.plot(sim_bin_edge + 0.5*bin_width, sim_hist_norm, color="red", linewidth=1)
    ax.set_xlim(0, x_lim)
    ax.set_ylim(0, y_lim)
    ax.set_xlabel(x_axis_text, fontsize=12)
    ax.set_ylabel("Particle density #/μm^2", fontsize=12)
    ax.set_title(title, fontsize=18)

    # Calculate and print statistics for simulated data
    sim_mean = sim_data.sel(properties=property).mean()
    sim_std = sim_data.sel(properties=property).std()
    # sim_total_num = sim_data.sel(properties=property).sum()
    sim_total_num = np.shape(sim_data)[0]/sim_image_area
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    ax.text(0.58, 0.97, f"Simulation data: \n Mean {sim_mean:.4f} \n Standard deviation {sim_std:.4f} \n Total nb density {sim_total_num:.4f} ", transform=ax.transAxes, fontsize=10,
        verticalalignment='top',   horizontalalignment='left', bbox=props)

    print(f"Simulated data (property: {property}):")
    print(f"  Average: {sim_mean:.4f}")
    print(f"  Standard Deviation: {sim_std:.4f}")
    print(f"  Total Number Density: {int(sim_total_num)}")
    
    # Show the plot
    plt.show()






def multiplot(full_data, sim_image_area, exp_data15, exp_area15, exp_data16, exp_area16, property, x_axis_text, title=" ", step_list=[1, 2, 5, 10, 15, 16]):
    """Function to plot experimental and simulated data together at multiple different steps

    Args:
        sim_data (np.array): simulated data
        sim_image_area (float): area of the simulation data
        property (str): name of the property in the array to plot
        x_axis_text (str): text to use as label of the x axis 
        title (str, optional): Title for the graph. Defaults to " ".
    """
    x_lim = 10
    y_lim = 10
    # for center
    if property == "orientation_deg":
        x_lim = 90
        y_lim = 0.002
    if property == "smallest_dimension":
        x_lim = 5.5
        y_lim = 0.0009
    if property == "smallest_dim_area":
        x_lim = 8
        y_lim = 0.0009
    if property == "area":
        x_lim = 40
        y_lim = 0.003
    if property == "largest_dimension":
        x_lim = 40
        y_lim = 0.0025
    if property == "ECD":
        x_lim = 13
        y_lim = 0.003
    if property == "Aspect Ratio":
        x_lim = 30
        y_lim = 0.005


    # # For edge
    # if property == "orientation_deg":
    #     x_lim = 91
    #     y_lim = 0.002
    # if property == "smallest_dimension":
    #     x_lim = 15
    #     y_lim = 0.002
    # if property == "area":
    #     x_lim = 40
    #     y_lim = 0.006
    # if property == "smallest_dim_area":
    #     x_lim = 15
    #     y_lim = 0.004
    # if property == "largest_dimension":
    #     x_lim = 15
    #     y_lim = 0.003
    # if property == "ECD":
    #     x_lim = 8
    #     y_lim = 0.003
    # if property == "Aspect Ratio":
    #     x_lim = 30
    #     y_lim = 0.005


    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, i in zip(axes, step_list):
        sim_data = full_data[i-1]
        sim_bin_edge, sim_hist = get_hist(sim_data, property)

        sim_hist_norm = sim_hist / sim_image_area

        if len(sim_bin_edge) > len(sim_hist_norm):
            sim_bin_edge = sim_bin_edge[:-1]  
        elif len(sim_hist_norm) > len(sim_bin_edge):
            sim_hist_norm = sim_hist_norm[:-1]  


        bin_width = sim_bin_edge[1] - sim_bin_edge[0]

        if i==16:
            exp_bin_edge, exp_hist = get_hist(exp_data16, property)
            exp_hist_norm = exp_hist / exp_area16

            if len(exp_bin_edge) > len(exp_hist_norm):
                exp_bin_edge = exp_bin_edge[:-1]  
            elif len(exp_hist_norm) > len(exp_bin_edge):
                exp_hist_norm = exp_hist_norm[:-1] 
            if len(sim_bin_edge) > len(sim_hist_norm):
                sim_bin_edge = sim_bin_edge[:-1]  
            elif len(sim_hist_norm) > len(sim_bin_edge):
                sim_hist_norm = sim_hist_norm[:-1]  

            ax.set_xlim(0, x_lim)
            ax.set_ylim(0, y_lim)
            bin_width = sim_bin_edge[1] - sim_bin_edge[0]
            ax.bar(
                exp_bin_edge[:-1],
                exp_hist_norm[:-1],
                width=np.diff(exp_bin_edge),
                align="edge",
                edgecolor="black",
                label= "experimental"
            )
            ax.plot(sim_bin_edge + 0.5*bin_width, sim_hist_norm, color="red", linewidth=1, label="simulation")
            ax.set_xlabel(x_axis_text, fontsize=12)
            ax.set_ylabel("Particle density #/μm²  ", fontsize=12)
            exp_mean = exp_data16.sel(properties=property).mean()
            exp_std = exp_data16.sel(properties=property).std()
            # exp_total_num = exp_data.sel(properties=property).sum()
            exp_total_num = (np.shape(exp_data16)[0])/exp_area16

            # Calculate and print statistics for simulated data
            sim_mean = sim_data.sel(properties=property).mean()
            sim_std = sim_data.sel(properties=property).std()
            # sim_total_num = sim_data.sel(properties=property).sum()
            sim_total_num = np.shape(sim_data)[0]/sim_image_area
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            ax.text(0.58, 0.97, f"Experimental data: \n Mean {exp_mean:.4f} \n Std {exp_std:.4f} \n Nb density {exp_total_num:.4f} \n Simulation data: \n Mean {sim_mean:.4f} \n Std {sim_std:.4f} \n Nb density {sim_total_num:.4f} ", transform=ax.transAxes, fontsize=10,
                verticalalignment='top',   horizontalalignment='left', bbox=props)
            ax.set_title(f"Step {i}")

            ax.legend()
        elif i == 15:
            exp_bin_edge, exp_hist = get_hist(exp_data15, property)
            exp_hist_norm = exp_hist / exp_area15

            if len(exp_bin_edge) > len(exp_hist_norm):
                exp_bin_edge = exp_bin_edge[:-1]  
            elif len(exp_hist_norm) > len(exp_bin_edge):
                exp_hist_norm = exp_hist_norm[:-1] 
            if len(sim_bin_edge) > len(sim_hist_norm):
                sim_bin_edge = sim_bin_edge[:-1]  
            elif len(sim_hist_norm) > len(sim_bin_edge):
                sim_hist_norm = sim_hist_norm[:-1]  

            ax.set_xlim(0, x_lim)
            ax.set_ylim(0, y_lim)
            bin_width = sim_bin_edge[1] - sim_bin_edge[0]
            ax.bar(
                exp_bin_edge[:-1],
                exp_hist_norm[:-1],
                width=np.diff(exp_bin_edge),
                align="edge",
                edgecolor="black",
                label= "experimental"
            )
            ax.plot(sim_bin_edge + 0.5*bin_width, sim_hist_norm, color="red", linewidth=1, label="simulation")
            ax.set_xlabel(x_axis_text, fontsize=12)
            ax.set_ylabel("Particle density #/μm²  ", fontsize=12)
            exp_mean = exp_data15.sel(properties=property).mean()
            exp_std = exp_data15.sel(properties=property).std()
            # exp_total_num = exp_data.sel(properties=property).sum()
            exp_total_num = (np.shape(exp_data15)[0])/exp_area15

            # Calculate and print statistics for simulated data
            sim_mean = sim_data.sel(properties=property).mean()
            sim_std = sim_data.sel(properties=property).std()
            # sim_total_num = sim_data.sel(properties=property).sum()
            sim_total_num = np.shape(sim_data)[0]/sim_image_area
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            ax.text(0.58, 0.97, f"Experimental data: \n Mean {exp_mean:.4f} \n Std {exp_std:.4f} \n Nb density {exp_total_num:.4f} \n Simulation data: \n Mean {sim_mean:.4f} \n Std {sim_std:.4f} \n Nb density {sim_total_num:.4f} ", transform=ax.transAxes, fontsize=10,
                verticalalignment='top',   horizontalalignment='left', bbox=props)

            ax.legend()
            ax.set_title(f"Step {i}")
        else:
            ax.plot(sim_bin_edge + 0.5*bin_width, sim_hist_norm, color="red", linewidth=1)
            ax.set_xlim(0, x_lim)
            ax.set_ylim(0, y_lim)
            ax.set_xlabel(x_axis_text, fontsize=12)
            ax.set_ylabel("Particle density #/μm^2", fontsize=12)
            ax.set_title(title, fontsize=18)

        # Calculate and print statistics for simulated data
            sim_mean = sim_data.sel(properties=property).mean()
            sim_std = sim_data.sel(properties=property).std()
        # sim_total_num = sim_data.sel(properties=property).sum()
            sim_total_num = np.shape(sim_data)[0]/sim_image_area
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            ax.text(0.58, 0.97, f"Simulation data: \n Mean {sim_mean:.4f} \n Std {sim_std:.4f} \n Nb density {sim_total_num:.4f} ", transform=ax.transAxes, fontsize=10,
                verticalalignment='top',   horizontalalignment='left', bbox=props)
            ax.set_title(f"Step {i}")


    print(f"Simulated data (property: {property}):")
    print(f"  Average: {sim_mean:.4f}")
    print(f"  Standard Deviation: {sim_std:.4f}")
    print(f"  Total Number Density: {int(sim_total_num)}")
    fig.suptitle(f'Evolution of the {title}', fontsize=16)
    # Show the plot
    plt.tight_layout()
    plt.show()



# Import the edge data 
def multiplot_V2(full_data, sim_image_area, exp_data16, exp_area16, property, x_axis_text, title=" ", step_list=[1, 2, 5, 10, 15, 16]):
    """Function to plot experimental and simulated data together at multiple different steps

    Args:
        sim_data (np.array): simulated data
        sim_image_area (float): area of the simulation data
        property (str): name of the property in the array to plot
        x_axis_text (str): text to use as label of the x axis 
        title (str, optional): Title for the graph. Defaults to " ".
    """
    x_lim = 10
    y_lim = 10
    # for center
    if property == "orientation_deg":
        x_lim = 90
        y_lim = 0.002
    if property == "smallest_dimension":
        x_lim = 5.5
        y_lim = 0.0009
    if property == "smallest_dim_area":
        x_lim = 8
        y_lim = 0.0009
    if property == "area":
        x_lim = 40
        y_lim = 0.003
    if property == "largest_dimension":
        x_lim = 40
        y_lim = 0.0025
    if property == "ECD":
        x_lim = 13
        y_lim = 0.003
    if property == "Aspect Ratio":
        x_lim = 30
        y_lim = 0.005


    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, i in zip(axes, step_list):
        sim_data = full_data[i-1]
        sim_bin_edge, sim_hist = get_hist(sim_data, property)

        sim_hist_norm = sim_hist / sim_image_area

        if len(sim_bin_edge) > len(sim_hist_norm):
            sim_bin_edge = sim_bin_edge[:-1]  
        elif len(sim_hist_norm) > len(sim_bin_edge):
            sim_hist_norm = sim_hist_norm[:-1]  


        bin_width = sim_bin_edge[1] - sim_bin_edge[0]

        if i==16:
            exp_bin_edge, exp_hist = get_hist(exp_data16, property)
            exp_hist_norm = exp_hist / exp_area16

            if len(exp_bin_edge) > len(exp_hist_norm):
                exp_bin_edge = exp_bin_edge[:-1]  
            elif len(exp_hist_norm) > len(exp_bin_edge):
                exp_hist_norm = exp_hist_norm[:-1] 
            if len(sim_bin_edge) > len(sim_hist_norm):
                sim_bin_edge = sim_bin_edge[:-1]  
            elif len(sim_hist_norm) > len(sim_bin_edge):
                sim_hist_norm = sim_hist_norm[:-1]  

            ax.set_xlim(0, x_lim)
            ax.set_ylim(0, y_lim)
            bin_width = sim_bin_edge[1] - sim_bin_edge[0]
            ax.bar(
                exp_bin_edge[:-1],
                exp_hist_norm[:-1],
                width=np.diff(exp_bin_edge),
                align="edge",
                edgecolor="black",
                label= "experimental"
            )
            ax.plot(sim_bin_edge + 0.5*bin_width, sim_hist_norm, color="red", linewidth=1, label="simulation")
            ax.set_xlabel(x_axis_text, fontsize=12)
            ax.set_ylabel("Particle density #/μm²  ", fontsize=12)
            exp_mean = exp_data16.sel(properties=property).mean()
            exp_std = exp_data16.sel(properties=property).std()
            # exp_total_num = exp_data.sel(properties=property).sum()
            exp_total_num = (np.shape(exp_data16)[0])/exp_area16

            # Calculate and print statistics for simulated data
            sim_mean = sim_data.sel(properties=property).mean()
            sim_std = sim_data.sel(properties=property).std()
            # sim_total_num = sim_data.sel(properties=property).sum()
            sim_total_num = np.shape(sim_data)[0]/sim_image_area
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            ax.text(0.58, 0.97, f"Experimental data: \n Mean {exp_mean:.4f} \n Std {exp_std:.4f} \n Nb density {exp_total_num:.4f} \n Simulation data: \n Mean {sim_mean:.4f} \n Std {sim_std:.4f} \n Nb density {sim_total_num:.4f} ", transform=ax.transAxes, fontsize=10,
                verticalalignment='top',   horizontalalignment='left', bbox=props)
            ax.set_title(f"Step {i}")

            ax.legend()
        
        else:
            ax.plot(sim_bin_edge + 0.5*bin_width, sim_hist_norm, color="red", linewidth=1)
            ax.set_xlim(0, x_lim)
            ax.set_ylim(0, y_lim)
            ax.set_xlabel(x_axis_text, fontsize=12)
            ax.set_ylabel("Particle density #/μm²", fontsize=12)
            ax.set_title(title, fontsize=18)

        # Calculate and print statistics for simulated data
            sim_mean = sim_data.sel(properties=property).mean()
            sim_std = sim_data.sel(properties=property).std()
        # sim_total_num = sim_data.sel(properties=property).sum()
            sim_total_num = np.shape(sim_data)[0]/sim_image_area
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            ax.text(0.58, 0.97, f"Simulation data: \n Mean {sim_mean:.4f} \n Std {sim_std:.4f} \n Nb density {sim_total_num:.4f} ", transform=ax.transAxes, fontsize=10,
                verticalalignment='top',   horizontalalignment='left', bbox=props)
            ax.set_title(f"Step {i}")


    print(f"Simulated data (property: {property}):")
    print(f"  Average: {sim_mean:.4f}")
    print(f"  Standard Deviation: {sim_std:.4f}")
    print(f"  Total Number Density: {int(sim_total_num)}")
    # fig.suptitle(f'Evolution of the {title}', fontsize=16)
    # Show the plot
    plt.tight_layout()
    plt.show()

