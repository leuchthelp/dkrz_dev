# Prepare your system
#sudo apt update
#sudo apt upgrade

# Install packages required by spack
#sudo apt update
#sudo apt install bzip2 ca-certificates g++ gcc gfortran git gzip lsb-release patch python3 tar unzip xz-utils zstd

compiler=gcc@11.2.0

# Install spack and add to local shell
git clone --depth=2 --branch=releases/v0.23 https://github.com/spack/spack.git ~/spack

export SPACK_ROOT=~/spack
. spack/share/spack/setup-env.sh
. $SPACK_ROOT/share/spack/setup-env.sh

spack compiler find

# Activate creation of module files via spack
spack config add "modules:default:enable:[tcl]"

# change this for you compiler of choice, has to be done for compatibility with levant.dkrz.de
#spack compiler add "$(spack location -i $compiler%$compiler)"

spack compilers

# Install and setup "module" and add to local shell
spack install --fresh lmod %$compiler
. $(spack location -i lmod)/lmod/lmod/init/profile
. $SPACK_ROOT/share/spack/setup-env.sh

# additionals for my own comfort
spack install --fresh git %$compiler
module load git
spack install --fresh nano %$compiler 

spack install --fresh python@3.11.9 %$compiler
module load python
spack install --fresh openmpi@5.0.5 %$compiler
module load openmpi
spack install --fresh hdf5@1.14.5~cxx~fortran+hl~ipo~java~map+mpi+shared+subfiling~szip+threadsafe+tools %$compiler
module load hdf5
spack install --fresh argobots@main %$compiler
module load argobots
spack install hdf5-vol-async@develop %$compiler 
module load hdf5-vol-async
spack install netcdf-c@4.9.2 build_system=cmake %$compiler
module load netcdf-c
spack install py-pip@23.1.2 %$compiler
module load py-pip
spack install py-mpi4py@4.0.1 %$compiler
module load py-mpi4py
spack install py-netcdf4@1.7.1 %$compiler
module load py-netcdf4
spack install py-h5py@3.12.1 %$compiler
module load py-h5py
spack install py-rich %$compiler
module load py-rich

# Add to end of home/user/.bashrc via nano .bashrc from within your home/user directory
#. ~/spack/share/spack/setup-env.sh
#. $SPACK_ROOT/share/spack/setup-env.sh
#. $(spack location -i lmod)/lmod/lmod/init/profile
# module load hdf5 netcdf-c py-mpi4py openmpi python py-pip py-netcdf4 py-h5py py-rich py-numpy git nano gcc
# module load hdf5-vol-async/develop

# Create and enter python virtualenv and install zarr v3.0.5 as this doesnt exist in spack yet
# cd /dkrz_dev
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt

# All other packages should already be available to pip via spack / modules
