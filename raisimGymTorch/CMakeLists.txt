cmake_minimum_required(VERSION 3.10)
project(_raisim_gym_torch)
set(CMAKE_CXX_STANDARD 17)
set(folder "raisimGymTorch")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env/bin)
SET(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env/bin)
SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env/bin)
## Add an option to build specify that an external raisim library is used
option(USE_EXTERNAL_RAISIM "Use external raisim library (i.e. The raisim build library is in other directory)" OFF)


# Set the Eigen3 directory to: Eigen3_DIR=$RAISIM_WORKSPACE/thirdParty/Eigen3/share/eigen3/cmake
set(Eigen3_DIR "$ENV{RAISIM_WORKSPACE}/thirdParty/Eigen3/share/eigen3/cmake")

######################
### help functions ###
######################
MACRO(SUBDIRLIST result curdir)
    FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
    SET(dirlist "")
    FOREACH(child ${children})
        IF(IS_DIRECTORY ${curdir}/${child})
            LIST(APPEND dirlist ${child})
        ENDIF()
    ENDFOREACH()
    SET(${result} ${dirlist})
ENDMACRO()

####################
### dependencies ###
####################
set(Dependencies)

add_subdirectory(thirdParty/pybind11 pybind11)
#list(APPEND CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../thirdParty/Eigen3)
find_package(Eigen3 CONFIG REQUIRED)

if (APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
elseif(WIN32)
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /openmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
else ()
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

if (UNIX AND NOT APPLE AND NOT ANDROID AND NOT WEBGL AND NOT WIN32)
    set(RAISIM_OS linux)
    if (USE_EXTERNAL_RAISIM)
        message("A path to the raisim library should have been added to the CMAKE_PREFIX_PATH variable")
    else()
        list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS})
    endif()
elseif(APPLE)
    # on macOS "uname -m" returns the architecture (x86_64 or arm64)
    execute_process(
            COMMAND uname -m
            RESULT_VARIABLE result
            OUTPUT_VARIABLE OSX_NATIVE_ARCHITECTURE
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(RAISIM_ARCH ${OSX_NATIVE_ARCHITECTURE} CACHE STRING "ARCH")

    if(${RAISIM_ARCH} STREQUAL "arm64")
        set(RAISIM_OS m1)
    else()
        set(RAISIM_OS mac)
    endif()
    if (USE_EXTERNAL_RAISIM)
        message("A path to the raisim library should have been added to the CMAKE_PREFIX_PATH variable")
    else()
        list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS})
    endif()
elseif(WIN32)
    set(RAISIM_OS win32)
    if (USE_EXTERNAL_RAISIM)
        message("Using external raisim library")
        message("A path to the raisim library should have been added to the CMAKE_PREFIX_PATH variable")
        message("A path to the thirdParty library (EIGEN) should have been added to the CMAKE_PREFIX_PATH variable")
    else()
        list(APPEND CMAKE_PREFIX_PATH ../raisim/${RAISIM_OS})
        list(APPEND CMAKE_PREFIX_PATH ../thirdParty)
    endif()
    list(APPEND Dependencies Ws2_32)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env/bin)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env/bin)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env/bin)
endif()

find_package(raisim CONFIG REQUIRED)

#######################
### src and linking ###
#######################
SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env/envs)
set(RAISIMGYM_ENV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env/envs)

message("raisim os ${RAISIM_OS}")
message("raisim arch ${RAISIM_ARCH}")
FOREACH(subdir ${SUBDIRS})
    message("building ${subdir}")
endforeach()

######## Env ##########
FOREACH(subdir ${SUBDIRS})
    message("building ${subdir}")
    if (${subdir} STREQUAL "giadog")
        pybind11_add_module(${subdir}
                            ${folder}/env/raisim_gym.cpp 
                            ${RAISIMGYM_ENV_DIR}/${subdir}/Environment.cpp
                            ${RAISIMGYM_ENV_DIR}/${subdir}/ContactSolver.cpp
                            ${RAISIMGYM_ENV_DIR}/${subdir}/ExternalForce.cpp
                            ${RAISIMGYM_ENV_DIR}/${subdir}/HeightScanner.cpp
                            ${RAISIMGYM_ENV_DIR}/${subdir}/WorldGenerator.cpp
                            ${RAISIMGYM_ENV_DIR}/${subdir}/utils.cpp
                            ${folder}/env/Yaml.cpp
                            )
    else()
        pybind11_add_module(${subdir}
                            ${folder}/env/raisim_gym.cpp 
                            ${folder}/env/Yaml.cpp
        )
    endif()
    target_link_libraries(${subdir} PRIVATE raisim::raisim ${Dependencies})
    target_include_directories(${subdir} PUBLIC ${EIGEN3_INCLUDE_DIRS} ${RAISIMGYM_ENV_DIR}/${subdir} ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env)
    if (WIN32)
        target_compile_options(${subdir} PRIVATE /D_USRDLL /MD /O2 /GL /arch:AVX)
    else()
        target_compile_options(${subdir} PRIVATE -fPIC -O3 -flto -mno-avx2)
    endif()
    target_compile_definitions(${subdir} PRIVATE "-DRAISIMGYM_TORCH_ENV_NAME=${subdir}")
    target_compile_definitions(${subdir} PRIVATE "$<$<CONFIG:Release>:EIGEN_NO_DEBUG>")
    target_compile_definitions(${subdir} PRIVATE EIGEN_DONT_PARALLELIZE)

    message("raisim os ${RAISIM_OS}")
    message("raisim arch ${RAISIM_ARCH}")


    if(${RAISIM_OS} STREQUAL "m1")
        if(${RAISIM_ARCH} STREQUAL "arm64")
            message("building with correct lib")
            target_compile_options(${subdir} PRIVATE -mcpu=apple-m1)
        endif()
    endif()

    if (CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
        message("[RAISIM_GYM] BUILDING THE DEBUG APP for ${subdir}")
        add_executable(${subdir}_debug_app ${folder}/env/debug_app.cpp ${folder}/env/Yaml.cpp)
        target_link_libraries(${subdir}_debug_app PRIVATE raisim::raisim)
        target_include_directories(${subdir}_debug_app PUBLIC ${folder}/env/envs/${subdir} ${EIGEN3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/${folder}/env)
        if(WIN32)
            target_link_libraries(${subdir}_debug_app PRIVATE Ws2_32)
        else()
            target_compile_options(${subdir}_debug_app PRIVATE -fPIC -g -O0 -mno-avx2)
        endif()
    endif()
ENDFOREACH()
