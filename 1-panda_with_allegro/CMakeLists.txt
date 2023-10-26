# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ALLEGRO_HAND_GRASP_BINARY_DIR}/1-panda_with_allegro)
ADD_EXECUTABLE (task_controller task_controller.cpp ${ALLEGRO_HAND_GRASP_COMMON_SOURCE})
ADD_EXECUTABLE (simviz simviz.cpp ${ALLEGRO_HAND_GRASP_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (task_controller ${ALLEGRO_HAND_GRASP_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz ${ALLEGRO_HAND_GRASP_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY world.urdf DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY panda_arm.urdf DESTINATION ${APP_RESOURCE_DIR})
