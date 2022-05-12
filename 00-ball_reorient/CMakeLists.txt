# create an executable
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${ALLEGRO_HAND_GRASP_BINARY_DIR}/00-ball_reorient)
ADD_EXECUTABLE (control_00_ball_reorient controller.cpp ${ALLEGRO_HAND_GRASP_COMMON_SOURCE})
ADD_EXECUTABLE (simviz_00_ball_reorient simviz.cpp ${ALLEGRO_HAND_GRASP_COMMON_SOURCE})

# and link the library against the executable
TARGET_LINK_LIBRARIES (control_00_ball_reorient  ${ALLEGRO_HAND_GRASP_COMMON_LIBRARIES})
TARGET_LINK_LIBRARIES (simviz_00_ball_reorient  ${ALLEGRO_HAND_GRASP_COMMON_LIBRARIES})

# export resources such as model files.
# NOTE: this requires an install build
SET(APP_RESOURCE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/resources)
FILE(MAKE_DIRECTORY ${APP_RESOURCE_DIR})
FILE(COPY world.urdf DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY panda_arm.urdf DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY panda_arm_allegro.urdf DESTINATION ${APP_RESOURCE_DIR})
FILE(COPY panda_arm_allegro_left.urdf DESTINATION ${APP_RESOURCE_DIR})
