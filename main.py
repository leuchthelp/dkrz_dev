import zarr, h5py
from mpi4py import MPI
from netCDF4 import Dataset

if __name__ == '__main__':

    #Todo read up on hdf5, zarr, netcdf, mpi, lustre, ceph
    # and check back with dkrz their structure, code conventions, datasets

    #Todo use netcdf dataset for testing (for now) should already bet hdf5
    dest_hdf = Dataset("data/source.nc", "w", format="NETCDF4")
    print(dest_hdf.data_model)
    dest_hdf.close()

    #Todo import netcdf data through hdf5 to zarr (for now)
    dest_zarr = zarr.open_group('data/example2.zarr', mode='w')
    zarr.copy_all(dest_hdf, dest_zarr)
    dest_zarr.tree()

    #Todo setup test cases proactively to ensure proper comparability and best practice

    #Todo setup Lustre and Ceph (need info on that)

    #Todo setup benchmark (prob compression, filesize, access-time, r/w-time) for sequential access

    #Todo setup benchmark for parallel access

    #Todo setup benchmark for random access

    #Todo setup benchmark for parallel with subfileing and async / I/O



