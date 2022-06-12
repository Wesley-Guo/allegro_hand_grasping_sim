# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ALLEGRO_HAND_GRASP_BINARY_DIR}/0-allegro_hand_only)
ADD_EXECUTABLE (control_0_allegro controller.cpp ${ALLEGRO_HAND_GRASP_COMMON_SOURCE})
ADD_EXECUTABLE (simviz_0_allegro simviz.cpp ${ALLEGRO_HAND_GRASP_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (control_0_allegro  ${ALLEGRO_HAND_GRASP_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz_0_allegro  ${ALLEGRO_HAND_GRASP_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY world.urdf DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY allegro_hand.urdf DESTINATION ${APP_RESOURCE_DIR})

# copy launch scripts
# FILE(COPY launch_sim+con.sh DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})