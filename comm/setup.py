import torch
from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import subprocess
import pathlib
import os
import re
import sys
from pathlib import Path

def get_version():
    version = '0.0.5.post1'
    # with open('stepkv/version.py', 'r') as fd:
    #     version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]',
    #                         fd.read(), re.MULTILINE).group(1)
    if len(sys.argv) >= 2:
        if sys.argv[1] == 'bdist_wheel':
            import torch
            torch_version = torch.__version__.replace("+", "")
            version = f"{version}+torch{torch_version}"
    assert version, 'Cannot find version information'
    print(version)
    return version

def _get_cuda_bare_metal_version(cuda_dir):
    assert cuda_dir is not None, "Please ensure cuda is installed"
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"],
                                         universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]

    return bare_metal_major, bare_metal_minor


__SRC_PATH__ = 'fserver/csrc/'
__PS_PATH__ = f'{Path.cwd()}'

def _find_ucx_libraries():
    """Find UCX libraries from NIXL package or system paths"""
    import site
    import glob
    
    # Try to find UCX libraries in NIXL package directory
    ucx_lib_dir = None
    ucx_libs = []
    
    # Check Python site-packages for NIXL UCX libraries
    site_packages = site.getsitepackages() + [site.getusersitepackages()]
    for site_dir in site_packages:
        nixl_lib_paths = [
            f"{site_dir}/nixl_cu12.libs",
            f"{site_dir}/.nixl_cu12.mesonpy.libs",
        ]
        for lib_path in nixl_lib_paths:
            if os.path.exists(lib_path):
                # Check for UCX core libraries
                ucp_lib = glob.glob(f"{lib_path}/libucp*.so*")
                ucs_lib = glob.glob(f"{lib_path}/libucs*.so*")
                uct_lib = glob.glob(f"{lib_path}/libuct*.so*")
                ucm_lib = glob.glob(f"{lib_path}/libucm*.so*")
                
                if ucp_lib and ucs_lib and uct_lib and ucm_lib:
                    ucx_lib_dir = lib_path
                    # Use the .so.0.0.0 versions (full versioned names)
                    ucp_lib = sorted(ucp_lib, key=len, reverse=True)[0]
                    ucs_lib = sorted(ucs_lib, key=len, reverse=True)[0]
                    uct_lib = sorted(uct_lib, key=len, reverse=True)[0]
                    ucm_lib = sorted(ucm_lib, key=len, reverse=True)[0]
                    print(f"Found UCX libraries in NIXL package: {lib_path}")
                    print(f"  UCP: {os.path.basename(ucp_lib)}")
                    print(f"  UCS: {os.path.basename(ucs_lib)}")
                    print(f"  UCT: {os.path.basename(uct_lib)}")
                    print(f"  UCM: {os.path.basename(ucm_lib)}")
                    return lib_path, [ucp_lib, uct_lib, ucs_lib, ucm_lib]
    
    # Fallback to system UCX libraries (if installed)
    print("UCX libraries not found in NIXL package, trying system paths...")
    return None, []

if __name__ == "__main__":
    cc_flag = []

    torch_cxx11_abi = torch.compiled_with_cxx11_abi()
    use_cuda = os.environ.get("USE_CUDA",'1')=='1'
    extra_link = ['-lrdmacm', '-libverbs']
    extra_compile_args={
            'cxx': [
                '-O3', '-fPIC', 
                f'-I{__PS_PATH__}/include', 
                f'-D_GLIBCXX_USE_CXX11_ABI={str(int(torch_cxx11_abi))}',
                '-DDMLC_USE_ZMQ',
                '-DSTEPMESH_USE_GDR',
                '-DDMLC_USE_RDMA', 
                '-DSTEPMESH_USE_TORCH',
                '-DSTEPMESH_ENABLE_TRACE',
                '-fvisibility=hidden',
                ],
                'nvcc': [],
                }
    
    # Check for UCX support (needed when using UCX backend)
    ucx_lib_dir, ucx_lib_files = _find_ucx_libraries()
    if ucx_lib_dir and ucx_lib_files:
        print("UCX support enabled: Adding UCX libraries to link arguments")
        extra_compile_args['cxx'] += ['-DDMLC_USE_UCX',]
        # Add runtime library path (rpath) so the .so can find UCX libraries at runtime
        extra_link.insert(0, f'-Wl,-rpath,{ucx_lib_dir}')
        # Link UCX libraries using full paths (order matters: ucp, uct, ucs, ucm)
        # For versioned library names like libucp-9d14a46b.so.0.0.0,
        # we need to link them directly using full paths
        extra_link.extend(ucx_lib_files)
    else:
        print("UCX libraries not found, UCX backend will not be available")
        print("  (This is OK if using RDMA backend instead of UCX backend)")
    
    if use_cuda:
        extra_link += ['-lcuda', '-lcudart']
        extra_compile_args['cxx'] += ['-DDMLC_USE_CUDA',]
        extra_compile_args['nvcc'] = ['-O3', '-gencode', 'arch=compute_90,code=sm_90', '-gencode', 'arch=compute_80,code=sm_80', '-gencode', 'arch=compute_89,code=sm_89','-gencode', 'arch=compute_90a,code=sm_90a',  
                '--use_fast_math', f'-D_GLIBCXX_USE_CXX11_ABI={str(int(torch_cxx11_abi))}'] + cc_flag
        bare_metal_major, bare_metal_minor = \
            _get_cuda_bare_metal_version(cpp_extension.CUDA_HOME)

    setup(
        name='FServer',
        description='A Remote FFN Server Implementation for AF Disaggregation',
        author='StepFun',
        version=get_version(),
        packages=['fserver'],
        url='',
        ext_modules=[
            CUDAExtension(
                'fserver_lib',
                [
                    __SRC_PATH__ + 'ops.cc',
                    __SRC_PATH__ + 'wait_kernel.cu',
                ],
                extra_compile_args=extra_compile_args,
                extra_link_args=extra_link,
                extra_objects=[f"{__PS_PATH__}/cmake_build/libaf.a", f"{__PS_PATH__}/deps/lib/libzmq.a"],
            )
        ],
        cmdclass={
            'build_ext': BuildExtension
        }
    )
