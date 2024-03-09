import ee

# Inicializar a API Earth Engine
ee.Initialize()

# Função para calcular a orientação da rua
def calculate_orientation(feature):
    geometry = ee.Feature(feature).geometry()
    angle = ee.Number(geometry.rotation())
    return angle

# Área de interesse (substitua isso pela sua própria região de interesse)
roi = ee.Geometry.Rectangle([-74.5, 40, -73.5, 41])

# Filtrar a coleção Google Maps Roads
roads_collection = ee.FeatureCollection('google/roads').filterBounds(roi)

# Selecionar uma rua específica (substitua isso pelo seu próprio método de seleção)
street_feature = roads_collection.first()

# Calcular a orientação da rua
orientation = calculate_orientation(street_feature)

# Imprimir a orientação da rua
print("Orientação da rua:", orientation.getInfo(), "radianos")