# IMC_Fragmentation
Model to simulate the fragmentation of the intermetallics duing the hot-rolling process. 

## Structure of the code and repository
There are two main file in this code, the `IMCFunctions.py`file which contains the declaration of all the functions, and the `IMCFragmentation.ipynb`file which contains the execution of the code. 
The function declerations are commented with the definition of the input parameters, their expected format, the role of the function and the resulting output. 

The main steps in executing the code are:
- Identify the particles present in the segmented image and calculate their properties. 
- Import/ set the strain values 
- Run the fragmentation model
- Plot the results


## How to install
This code can be installed by cloning this repository with `git clone github.com/zoepevans/IMC_Fragmentation_Public`. 

The image segmentation files then need to be unziped. This can be done through the graphical interface of the file manager, or running `unzip` in the correct directory in a terminal (linux and mac).

The model runs with Python 3.12.12 and requires the following packages:
- `importlib`
- `numpy`
- `pandas`
- `xarray`
- `h5py`
- `pickle`
- `scipy`
- `skimage`
- `matplotlib`
- `math`
- `random`
- `re`


## How to run the model

1. Get the segmentation of the image from *Ilastik*. Run *Ilastik* and select the pixel classification function. 
Add the As-homogenized SEM image in the first tab on the right. 
In the second tab select all 37 of the features.
Then in the third tab set the three regions in order to the matrix, other particles and intermetallics. 
Using the pencil tool indicate which regions in the image correspond to each classification region. 
A few pencil marks should be sufficient to correctly classify the image. 
Use the live update view to check the classification of the image. 
When the results are good, select the fourth tab and set the export setting to h5df file and the data to export to simple segmentation. 
The export file and location can also be set at this step. Export the file.
2. Open the `IMCFragmentation.ipynb` file, and run the first cell to import the necessary modules
3. Import the strain values, either from FEM simulation files or a list of values
4. Import the segmentation from an image and identify the properties of the particles in the images
5. The array containing the particle properties can be saved into a pickle file (for easy reading into the code) or a csv file to view as a spreadsheet. The area of the image is also calculate here. 
This step requires a conversion factor that is the number of pixels per μm in the image (depends on the image resolution and the magnification, a software such as *ImageJ* can be used to measure this)
6. The fragmentation model is run on the imported data and the particle properties for each step are output. 
7. The data is plot relative to experimental results. 
8. The data can be output to a csv file for further manipulation
