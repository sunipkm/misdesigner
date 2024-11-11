# SPDX-FileCopyrightText: 2024 Sunip K. Mukherjee
#
# SPDX-License-Identifier: Apache-2.0

project('misdesigner', 'c',
  version : '1.0',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2'],
)

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

compiler = meson.get_compiler('c')

add_global_arguments('-O3', language : 'c')
add_global_arguments('-c', language : 'c')

omp_dep = dependency('openmp', required: false)
m_dep = compiler.find_library('m', required: true)

if omp_dep.found()
  # OK
else
  add_global_arguments('-DNO_OMP', language : 'c')
endif

c_cpp_srcs = [
  'src/c_cpp/integrate.c',
  'src/c_cpp/integrate.h', 
]

lib = shared_library('integrate_unsort_lib',
  name_prefix: '',
  sources: c_cpp_srcs,
  include_directories: 'src/c_cpp',
  install : true,
  install_dir : py.get_install_dir() / 'misdesigner',
  dependencies: [omp_dep, m_dep],
)

py.install_sources(
    'src/misdesigner/__init__.py',
    'src/misdesigner/instrument_params.py',
    'src/misdesigner/line_predictor.py',
    'src/misdesigner/multi_integrate.py',
    'src/misdesigner/utils.py',
    subdir: 'misdesigner',
    pure: false,
)


# py.extension_module('integrate_unsort_lib',
#   c_cpp_srcs,
#   include_directories: 'src/C',
#   install : true,
#   subdir: 'integrate_unsort'
# )