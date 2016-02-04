CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(Prueba_ORB)
ADD_EXECUTABLE(myOrb Tiposgenericos.cpp)

# Defino las flags para el preprocesador
SET(CMAKE_CXX_FLAGS "-DTIME")

# Busco paquetes 
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Boost REQUIRED)

# Busco una librería 
FIND_LIBRARY(OPENCV NAMES opencv_core)
IF(NOT OPENCV)
    MESSAGE("No se pudo encontrar la librería: OpenCV")
ELSE()
	MESSAGE("Se encontró la ibrería: OpenCV")
ENDIF()

# Linkeo librerías
TARGET_LINK_LIBRARIES(myOrb ${OpenCV_LIBS})

# Imprimo librerías cargadas
MESSAGE(${OpenCV_LIBS}) 


# Defino tipo de construcción
IF(NOT CMAKE_BUILD_TYPE )
   SET( CMAKE_BUILD_TYPE "Release" )
   MESSAGE(STATUS " **************  Versión Estable")
ENDIF()

