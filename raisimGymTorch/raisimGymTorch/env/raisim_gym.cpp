//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisim;
int THREAD_COUNT = 1;

#ifndef ENVIRONMENT_NAME
#define ENVIRONMENT_NAME RaisimGymEnv
#endif

PYBIND11_MODULE(RAISIMGYM_TORCH_ENV_NAME, m)
{
    // Class that represents the output of a step in the simulation
    py::class_<step_array_info_t>(m, "StepArrayInfo")
        .def_readwrite("non_privileged_observations", &step_array_info_t::non_privileged_observations)
        .def_readwrite("privileged_observations", &step_array_info_t::privileged_observations)
        .def_readwrite("historic_observations", &step_array_info_t::historic_observations)
        .def_readwrite("dones", &step_array_info_t::dones)
        .def_readwrite("rewards", &step_array_info_t::rewards)
        .def_readwrite("traversability", &step_array_info_t::traversability)
        .def_readwrite("froude", &step_array_info_t::froude)
        .def_readwrite("projected_speed", &step_array_info_t::projected_speed)
        .def_readwrite("max_torque", &step_array_info_t::max_torque)
        .def_readwrite("power", &step_array_info_t::power);

    // Class that represents the scaling stats of the environment
    py::class_<statistics_t>(m, "Statistics")
        .def_readwrite("mean", &statistics_t::mean)
        .def_readwrite("var", &statistics_t::var)
        .def_readwrite("count", &statistics_t::count);

    py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
        .def(
            py::init<std::string, std::string, int, bool, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>(),
            py::arg("resource_dir"), 
            py::arg("cfg"), 
            py::arg("port"), 
            py::arg("normalize"),
            py::arg("non_privileged_obs"),
            py::arg("privileged_obs"),
            py::arg("historic_obs")
            )
        .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
        .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
        .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
        .def("get_statistics", &VectorizedEnvironment<ENVIRONMENT>::get_statistics)
        .def("set_statistics", &VectorizedEnvironment<ENVIRONMENT>::set_statistics)
        .def("update_curriculum_coefficient", &VectorizedEnvironment<ENVIRONMENT>::update_curriculum_coefficient)
        .def("set_curriculum_coefficient", &VectorizedEnvironment<ENVIRONMENT>::set_curriculum_coefficient)
        .def("hills", &VectorizedEnvironment<ENVIRONMENT>::hills)
        .def("stairs", &VectorizedEnvironment<ENVIRONMENT>::stairs)
        .def("cellular_steps", &VectorizedEnvironment<ENVIRONMENT>::cellular_steps)
        .def("steps", &VectorizedEnvironment<ENVIRONMENT>::steps)
        .def("slope", &VectorizedEnvironment<ENVIRONMENT>::slope)
        .def("set_command", &VectorizedEnvironment<ENVIRONMENT>::set_command)
        .def("get_num_envs", &VectorizedEnvironment<ENVIRONMENT>::get_num_envs)
        .def("set_absolute_position", &VectorizedEnvironment<ENVIRONMENT>::set_absolute_position)
        .def("set_absolute_velocity", &VectorizedEnvironment<ENVIRONMENT>::set_absolute_velocity)
        .def("set_foot_positions_and_base_pose",  &VectorizedEnvironment<ENVIRONMENT>::set_foot_positions_and_base_pose)
        .def("set_gait_config",  &VectorizedEnvironment<ENVIRONMENT>::set_gait_config)
        .def("get_obs_sizes", &VectorizedEnvironment<ENVIRONMENT>::get_obs_sizes)
        .def("get_observations_indexes", &VectorizedEnvironment<ENVIRONMENT>::get_observations_indexes)
        .def(py::pickle(
            [](const VectorizedEnvironment<ENVIRONMENT> &p) { // __getstate__ --> Pickling to Python
                /* Return a tuple that fully encodes the state of the object */
                return py::make_tuple(p.get_resource_dir(), p.get_cfg_string());
            },
            [](py::tuple t) { // __setstate__ - Pickling from Python
                if (t.size() != 2)
                {
                    throw std::runtime_error("Invalid state!");
                }

                /* Create a new C++ instance */
                VectorizedEnvironment<ENVIRONMENT> p(
                    t[0].cast<std::string>(),
                    t[1].cast<std::string>(),
                    t[2].cast<int>(),
                    t[3].cast<int>(),
                    t[4].cast<std::vector<std::string>>(),
                    t[5].cast<std::vector<std::string>>(),
                    t[6].cast<std::vector<std::string>>());

                return p;
            }));
}
