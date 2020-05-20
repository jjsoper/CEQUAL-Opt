import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
import pandas as pd
import os

gdf = gpd.read_file('..\\shp\\cequal_grids.shp')
gdf['Segment'] = pd.to_numeric(gdf['Segment'])
gdf['color'] = np.zeros(len(gdf))

# regions dict from calibration script
wsc = {
    'Default': np.r_[1:10],
    'Thomas': np.r_[10:28],
    'South': np.r_[46:51],
    'North': np.r_[28:46, 51:63]
}

regions = [key for key in wsc.keys()]
for index in range(len(regions)):
    gdf.loc[gdf.Segment.isin(wsc[regions[index]]),'color'] = regions[index]

gdf.plot(column = 'color', legend = True, legend_kwds={'loc': 'lower right'}, cmap = 'Dark2', )
plt.show()
