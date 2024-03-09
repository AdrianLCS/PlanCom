import ee

def get_vec_land_surfelv(loncoord, latcoord):

    ee.Initialize(project="plancom-409417")
    altura = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2').select('DSM')
    lcover10 = ee.ImageCollection("ESA/WorldCover/v200")
    dsm = []
    landcover = []
    for i in range(len(latcoord)):
        ponto = ee.Geometry.Point(loncoord[i], latcoord[i])

        dsm.append(altura.mean().sample(ponto, 30).first().get('DSM').getInfo())
        landcover.append(lcover10.first().sample(ponto, 10).first().get('Map').getInfo())
        if i < len(latcoord)-1:
            latpasso = (latcoord[i+1]-latcoord[i])/3
            lonpasso = (loncoord[i+1]-loncoord[i])/3
            ponto2 = ee.Geometry.Point(loncoord[i]+lonpasso, latcoord[i]+latpasso)
            ponto3 = ee.Geometry.Point(loncoord[i]+2*lonpasso, latcoord[i]+2*latpasso)
            landcover.append(lcover10.first().sample(ponto2, 10).first().get('Map').getInfo())
            landcover.append(lcover10.first().sample(ponto3, 10).first().get('Map').getInfo())

    return landcover, dsm


ee.Initialize(project="plancom-409417")
# Import the MODIS land cover collection.
lc2 = ee.ImageCollection("COPERNICUS/Landcover/100m/Proba-V-C3/Global")
lc = ee.ImageCollection('MODIS/006/MCD12Q1')
lc10m = ee.ImageCollection("ESA/WorldCover/v200")

# Import the MODIS land surface temperature collection.
lst = ee.ImageCollection('MODIS/006/MOD11A1')

# Import the USGS ground elevation image.
elv = ee.Image('USGS/SRTMGL1_003')
elvn = ee.Image("NASA/NASADEM_HGT/001")

# Import the DSM.
Dsm2 = ee.ImageCollection('JAXA/ALOS/AW3D30/V3_2')
alt = Dsm2.select('DSM')
Dsm1 = ee.ImageCollection("COPERNICUS/DEM/GLO30")


# construcoes
# https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_Research_open-buildings_v3_polygons
constru = ee.FeatureCollection("GOOGLE/Research/open-buildings/v3/polygons")

# Initial date of interest (inclusive).
i_date = '2017-01-01'

# Final date of interest (exclusive).
f_date = '2020-01-01'

# Selection of appropriate bands and dates for LST.
lst = lst.select('LST_Day_1km', 'QC_Day').filterDate(i_date, f_date)

# Define the urban location of interest as a point near Lyon, France.
u_lon = 4.8148
u_lat = 45.7758
u_poi = ee.Geometry.Point(u_lon, u_lat)

# Define the rural location of interest as a point away from the city.
r_lon = -43.166828
r_lat = -22.954277
r_poi = ee.Geometry.Point(r_lon, r_lat)

scale = 30  # scale in meters

# Print the elevation near Lyon, France.
#{'type': 'FeatureCollection', 'columns': {'DSM': 'Float<-32768.0, 32767.0>'}, 'properties': {'band_order': ['DSM']}, 'features': [{'type': 'Feature', 'geometry': None, 'id': '0', 'properties': {'DSM': 10}}]} m
dsm_value = float(alt.mean().sample(r_poi, 30).getInfo()['features'][0]['properties']['DSM'])
surface_elv = alt.mean().sample(r_poi, 30).first().get('DSM').getInfo()
elv_urban_point = elvn.sample(r_poi, scale).first().get('elevation').getInfo()
print('Surface elevation at urban point:', surface_elv, 'm')
print('Ground elevation at urban point:', elv_urban_point, 'm')

# Calculate and print the mean value of the LST collection at the point.
lst_urban_point = lst.mean().sample(r_poi, scale).first().get('LST_Day_1km').getInfo()
print('Average daytime LST at urban point:', round(lst_urban_point*0.02 -273.15, 2), '°C')

# Print the land cover type at the point.
lc_urban_point = lc10m.first().sample(r_poi, 10).first().get('Map').getInfo()
print('Land cover value at urban point is:', lc_urban_point)