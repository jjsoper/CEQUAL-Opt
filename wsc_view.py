import shapefile as shp
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd

data_folder = "C:\\Users\\Josh Soper\\Documents\\Master's Thesis\\Research\\GIS\\Data"
grid = gpd.read_file(data_folder + '\\' + 'cequal_grids.shp')
segs = [np.r_[1:10], np.r_[10:28], np.r_[46:51], np.r_[28:46, 51:63]]
grid.head()
