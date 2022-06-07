from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, _get_cuda_arch_flags, _join_cuda_home
import os
import glob
import itertools
import re

_arch_flags = _get_cuda_arch_flags()
print('arch flags:', _arch_flags)

_root = os.path.split(os.path.abspath(__file__))[0]
print("root path:", _root)
if not os.path.exists(os.path.join(_root, 'build/__cmrc_Renderer-kernels')):
    print("No resources found, did you run CMake beforehand?")
    exit(-1)
def get_files(base, filter=".*"):
    fx1 = glob.iglob(base + "/**/*.cu", recursive=True)
    fx2 = glob.iglob(base + "/**/*.cpp", recursive=True)
    fx3 = glob.iglob(base + "/**/*.c", recursive=True)
    fx = itertools.chain(fx1, fx2, fx3)
    prog = re.compile(filter)
    return [os.path.relpath(f, _root) for f in fx if prog.fullmatch(f)]

_renderer_files = get_files(os.path.join(_root, 'renderer'))
_compression_files = get_files(os.path.join(_root, 'compression/include')) + get_files(os.path.join(_root, 'compression/src'))
_binding_files = get_files(os.path.join(_root, 'bindings'))
_resource_files = get_files(os.path.join(_root, 'build/__cmrc_Renderer-kernels/intermediate')) +\
    ['build/__cmrc_Renderer-kernels/lib.cpp']
_thirdparty_files = get_files(os.path.join(_root, 'third-party/lz4/lib'))
_imgui_files = get_files(os.path.join(_root, 'imgui'), '^(?!.*impl).*$')
print("renderer files:", _renderer_files)
print("binding files:", _binding_files)
print("resource files:", _resource_files)
print("third party files:", _thirdparty_files)
print("ImGui files:", _imgui_files)

_include_dirs = [
    '%s/renderer'%_root,
    '%s/third-party/cuMat'%_root,
    '%s/third-party/cuMat/third-party'%_root,
    '%s/third-party/magic_enum/include'%_root,
    '%s/third-party/cudad/include/cudAD'%_root,
    '%s/third-party/tinyformat'%_root,
    '%s/third-party/nlohmann'%_root,
    '%s/third-party/lz4/lib'%_root,
    '%s/third-party/portable-file-dialogs'%_root,
    '%s/third-party/thread-pool/include'%_root,
    '%s/build/_cmrc/include'%_root,
    '%s/compression/include'%_root,
    '%s/compression/src'%_root,
    '%s/imgui'%_root,
    '/usr/include',
]

_libraries = [
    'cuda',
    'nvrtc',
    'curand',
    'GL', 'GLU',
]

_common_args = [
    '-std=c++17',
    '-DNVCC_ARGS="%s"'%_arch_flags[0],
    '-DNVCC_INCLUDE_DIR=%s'%_join_cuda_home('include'),
    '-DUSE_DOUBLE_PRECISION=1',
    '-DRENDERER_OPENGL_SUPPORT=0',
    '-DCDUMAT_SINGLE_THREAD_CONTEXT=1',
    '-DTHRUST_IGNORE_CUB_VERSION_CHECK=1',
]

setup(
    name='pyrenderer',
    ext_modules=[
        CUDAExtension('pyrenderer',
            _renderer_files+_binding_files+_resource_files+_thirdparty_files+_imgui_files+_compression_files,
            extra_compile_args = {
                'cxx': _common_args,
                'nvcc': _common_args+["--extended-lambda"]
            },
            include_dirs = _include_dirs,
            libraries = _libraries),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
