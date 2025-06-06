from func.datastruct import bcolors as color
import pandas as pd
import os, shutil, subprocess, json
    

def delete_json(path):
     if os.path.exists(path):
            os.remove(path)
        
    
def show_header(ds_tmp):
    print(f"{color.OKGREEN}check header with format: {ds_tmp.engine}{color.ENDC}")
    ds_tmp.read(pattern="header")
    
    print(f"{color.OKCYAN}check variable header with format: {ds_tmp.engine}{color.ENDC}")
    ds_tmp.read(pattern="header", variable="X")


def calc_chunksize(chunks, stop_depth="GB"):
    
    res = 1
    size = "Byte"
    
    for chunksize in chunks:
        res *= chunksize
    
    res *= 8
    if stop_depth == "KB":
        res /= 1024
        size="KB"
       
    if stop_depth == "MB":
        res = res / 1024 / 1024
        size = "MB"
       
    if stop_depth == "GB":
        res = res / 1024 / 1024 / 1024
        size = "GB"

    return (res, size)


def runner_zarr(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index):
    
    for i in range(iterations):
        ## run benchmark on zarr python file
        print(f"{color.OKBLUE}bench zarr with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"python benchmarks.py -b 1 -v {variable} -i {1}"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()

        times = pd.read_json("data/results/test-zarr-python.json")[0].tolist()
        tmp = pd.DataFrame(data={"time taken": times, "format": f"zarr-python-{shape}-{chunks}", "run":index, "engine": "zarr-python", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True)

    ## delete zarr python file
        
    return df


def runner_zarr_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks):
    
    for i in range(iterations):
        ## run benchmark on zarr python file
        print(f"{color.OKBLUE}bench zarr parallel with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk:  {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"mpiexec -n {mpi_ranks} python benchmarks.py -b 4 -v {variable} -i {1}"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()

        times = pd.read_json("data/results/test-zarr-python-parallel.json")[0].tolist()
        tmp = pd.DataFrame(data={"time taken": times, "format": f"zarr-python-parallel-{shape}-{chunks}", "run":index, "engine": "zarr-python-parallel", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df= pd.concat([df, tmp], ignore_index=True)

    ## delete zarr python file
    
    return df


def runner_netcdf4(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index):
    
    for i in range(iterations):
        ## run benchmark on netcdf4 python file
        print(f"{color.OKBLUE}bench netcdf4 with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk:  {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"python benchmarks.py -b 2 -v {variable} -i {1}"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()

        times = pd.read_json("data/results/test-netcdf4-python.json")[0].tolist()
        tmp = pd.DataFrame(data={"time taken": times, "format": f"netcdf4-python-{shape}-{chunks}", "run":index, "engine": "netcdf4-python", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df= pd.concat([df, tmp], ignore_index=True)

    ## delete netcdf4 python file - current still need for a later c run of netcdf4

    return df


def runner_netcdf4_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks):
    
    for i in range(iterations):
        ## run benchmark on netcdf4 python file
        print(f"{color.OKBLUE}bench netcdf4 parallel with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk:  {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"mpiexec -n {mpi_ranks} python benchmarks.py -b 5 -v {variable} -i {1}"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()

        times = pd.read_json("data/results/test-netcdf4-python-parallel.json")[0].tolist()
        tmp = pd.DataFrame(data={"time taken": times, "format": f"netcdf4-python-parallel-{shape}-{chunks}", "run":index, "engine": "netcdf4-python-parallel", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df= pd.concat([df, tmp], ignore_index=True)

    ## delete netcdf4 python file - current still need for a later c run of netcdf4
    
    return df


def runner_hdf5(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index):

    
    for i in range(iterations):
        ## run benchmark on hdf5 python file
        print(f"{color.OKBLUE}bench hdf5 with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"python benchmarks.py -b 3 -v {variable} -i {1}"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()

        times = pd.read_json("data/results/test-hdf5-python.json")[0].tolist()
        tmp = pd.DataFrame(data={"time taken": times, "format": f"hdf5-python-{shape}-{chunks}", "run":index, "engine": "hdf5-python", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True)

    ## delete hdf5 python file
        
    return df


def runner_hdf5_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks):

    for i in range(iterations):
        ## run benchmark on hdf5 python file
        print(f"{color.OKBLUE}bench hdf5 parallel with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"mpiexec -n {mpi_ranks} python benchmarks.py -b 6 -v {variable} -i {1}"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()

        times = pd.read_json("data/results/test-hdf5-python-parallel.json")[0].tolist()
        tmp = pd.DataFrame(data={"time taken": times, "format": f"hdf5-python-parallel-{shape}-{chunks}", "run":index, "engine": "hdf5-python-parallel", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True)

    ## delete hdf5 python file
        
    return df


def runner_netcdf4_c(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index):
        
    for i in range(iterations):    
        # netcd-c
        filesize = shape[0]

        print(f"{color.WARNING}create netcdf4-c Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        ## create netcdf4 C file - current not needed as file created by python benchmark is used

        ## run benchmark on netcdf4 C file
        print(f"{color.OKBLUE}bench netcdf-c with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"./a.out -b 2 -i {1} -s {filesize}"
        p = subprocess.Popen(["sbatch", "../slurm-scripts/run-anything.sh" , call, "False"],  cwd="./c-stuff")
        p.wait()

        tmp = pd.read_json("./c-stuff/data/results/test_netcdf4.json")
        df_netcdf4_c = tmp["netcdf4-read"].tolist()

        tmp = pd.DataFrame(data={"time taken": df_netcdf4_c, "format": f"netcdf4-c-[{filesize}]-{chunks}", "run":index, "engine": "netcdf4-c", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True)

    ## delete netcdf4 file
        
    return df
 
 
def runner_netcdf4_c_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks):
    
    for i in range(iterations):
        # netcdf4-c-parallel
        filesize = shape[0]

        print(f"{color.WARNING}create netcdf4-c Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        ## create netcdf4 C file parallel 

        ## run benchmark on netcdf4 C file parallel
        print(f"{color.OKBLUE}bench netcdf4-c parallel with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"mpiexec -n {mpi_ranks} ./a.out -b 7 -i {1} -s {filesize}"
        p = subprocess.Popen(["sbatch", "../slurm-scripts/run-anything.sh" , call, "False"],  cwd="./c-stuff")
        p.wait()

        tmp= pd.read_json("./c-stuff/data/results/test_netcdf4-c_parallel.json")
        df_netcdf4_c_parallel = tmp["netcdf4-c-read-parallel"].tolist()

        tmp = pd.DataFrame(data={"time taken": df_netcdf4_c_parallel, "format": f"netcdf4-c-parallel-[{filesize}]-{chunks}", "run":index, "engine": "netcdf4-c-parallel", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True)

    ## delete netcdf4 C file parallel
    if os.path.exists("data/datasets/test_dataset.nc"):
        os.remove("data/datasets/test_dataset.nc")
        
    return df 
 

def runner_hdf5_c(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index):
    
    for i in range(iterations):
        # hdf-c
        filesize = shape[0]

        print(f"{color.WARNING}create hdf5-c Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        ## create hdf5 C file

        ## run benchmark on hdf5 C file
        print(f"{color.OKBLUE}bench hdf5-c with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"./a.out -b 1 -i {1} -s {filesize}"
        p = subprocess.Popen(["sbatch", "../slurm-scripts/run-anything.sh" , call, "False"],  cwd="./c-stuff")
        p.wait()

        tmp= pd.read_json("./c-stuff/data/results/test_hdf5-c.json")
        df_hdf5_c = tmp["hdf5-c-read"].tolist()

        tmp = pd.DataFrame(data={"time taken": df_hdf5_c, "format": f"hdf5-c-[{filesize}]-{chunks}", "run":index, "engine": "hdf5-c", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True) 

    ## delete hdf5 C file
    if os.path.exists("c-stuff/data/datasets/test_dataset_hdf5-c.h5"):
        os.remove("c-stuff/data/datasets/test_dataset_hdf5-c.h5")
        
    return df
 
    
def runner_hdf5_c_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks):
    
    for i in range(iterations):
        # hdf-c-parallel
        filesize = shape[0]

        print(f"{color.WARNING}create hdf5-c parallel Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        ## create hdf5 C file parallel 


        ## run benchmark on hdf5 C file parallel
        print(f"{color.OKBLUE}bench hdf5-c parallel with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"mpiexec -n {mpi_ranks} ./a.out -b 4 -i {1} -s {filesize} -k {chunks}"
        p = subprocess.Popen(["sbatch", "../slurm-scripts/run-anything.sh" , call, "False"],  cwd="./c-stuff")
        p.wait()
        print(p.returncode)

        tmp= pd.read_json("./c-stuff/data/results/test_hdf5-c_parallel.json")
        df_hdf5_c_parallel = tmp["hdf5-c-read-parallel"].tolist()

        tmp = pd.DataFrame(data={"time taken": df_hdf5_c_parallel, "format": f"hdf5-c-parallel-[{filesize}]-{chunks}", "run":index, "engine": "hdf5-c-parallel", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True)

    ## delete hdf5 C file parallel
    if os.path.exists("c-stuff/data/datasets/test_dataset_hdf5-c.h5"):
        os.remove("c-stuff/data/datasets/test_dataset_hdf5-c.h5") 
        
    return df
 
    
def runner_hdf5_c_async(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks):
        
    for i in range(iterations):    
        # hdf5-c-async
        filesize = shape[0]

        print(f"{color.WARNING}create hdf5-c async Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        ## create hdf5-vol-async C file 

        ## run benchmark on hdf5-vol-async C file
        print(f"{color.OKBLUE}bench hdf5-c async with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"mpiexec -n {mpi_ranks} ./a.out -b 5 -i {1} -s {filesize} -k {chunks}"
        p = subprocess.Popen(["sbatch", "../slurm-scripts/run-anything.sh" , call, "False"],  cwd="./c-stuff")
        p.wait()
        print(p.returncode)

        tmp = pd.read_json("./c-stuff/data/results/test_hdf5-c_async.json")
        df_hdf5_async = tmp["hdf5-c-async-read"].tolist()

        tmp = pd.DataFrame(data={"time taken": df_hdf5_async, "format": f"hdf5-async-[{filesize}]-{chunks}", "run":index, "engine": "hdf5-async", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True)

    ## delete hdf5-vol-async file C parallel
    if os.path.exists("c-stuff/data/datasets/test_dataset_hdf5-c_async.h5"):
        os.remove("c-stuff/data/datasets/test_dataset_hdf5-c_async.h5")
        
    return df
   
    
def runner_hdf5_c_subfiling(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks):
    
    for i in range(iterations):
        
        # hdf5-c-subfiling
        filesize = shape[0]

        ## create hdf5-subfiling C file 
        print(f"{color.WARNING}create hdf5-c subfiling Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        ## run benchmark on hdf5-subfiling C file
        print(f"{color.OKBLUE}bench hdf5-c parallel with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")

        call = f"mpiexec -n {mpi_ranks} ./a.out -b 6 -i {1} -s {filesize} -k {chunks}"
        p = subprocess.Popen(["sbatch", "../slurm-scripts/run-anything.sh" , call, "False"],  cwd="./c-stuff")
        p.wait()
        print(p.returncode)

        tmp = pd.read_json("./c-stuff/data/results/test_hdf5_subfiling.json")
        df_hdf5_subfiling = tmp["hdf5-subfiling-read"].tolist()

        tmp = pd.DataFrame(data={"time taken": df_hdf5_subfiling, "format": f"hdf5-subfiling-[{filesize}]-{chunks}", "run":index, "engine": "hdf5-subfiling", "total filesize": f"{total_filesize[0]} {total_filesize[1]}", "filesize per chunk": f"{size_chunks[0]} {size_chunks[1]}"})
        df = pd.concat([df, tmp], ignore_index=True)

    ## delete hdf5-subfiling C file - current not possible due to the unique structure of subfiling 
    
    return df


def bench_variable(setup, df, variable, iterations, mpi_ranks, path):
    index = 0
    
    if os.path.exists("log/"):
        shutil.rmtree("log/")
        
    if os.path.exists("c-stuff/log/"):
        shutil.rmtree("c-stuff/log/")
    
    for run in setup.values():
        
        shape = run[variable][0]
        chunks = run[variable][1]
        
        total_filesize = calc_chunksize(chunks=shape)
        
        size_chunks = [None, None]
        if len(chunks) != 0:
            size_chunks = calc_chunksize(chunks=chunks, stop_depth="MB")
        
        print(f"{color.WARNING}create python Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        
        with open("data/tmp/run_config.json", "w") as f:
            json.dump(run, f)
        
        ## create Zarr, NetCDF4 and HDF5 file sequentially 
        call = "python benchmarks.py -c 1"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()
        
        call = "python benchmarks.py -c 2"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()
        
        call = "python benchmarks.py -c 3"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()
        
        df = runner_zarr(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_netcdf4(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_hdf5(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_zarr_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_netcdf4_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_hdf5_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        
        ## delete zarr python file
        if os.path.exists("data/datasets/test_dataset.zarr"):
            shutil.rmtree("data/datasets/test_dataset.zarr")
        
        ## delete hdf5 python file
        if os.path.exists("data/datasets/test_dataset.h5"):
            os.remove("data/datasets/test_dataset.h5")
        
        # setup metadata for c-runs
        filesize = shape[0]
        chunksize = 0
        
        if len(chunks) != 0:
            chunksize = chunks[0]
        
        print(f"{color.WARNING}create c Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        
        call_create_hdf5 = f"./a.out -c 1 -s {filesize} -k {chunksize}"
        call_create_hdf5_parallel = f"mpiexec -n  {mpi_ranks} ./a.out -c 4 -s {filesize} -k {chunksize}"
        call_create_hdf5_async = f"mpiexec -n  {mpi_ranks} ./a.out -c 5 -s {filesize} -k {chunksize}"
        call_create_hdf5_subfiling = f"mpiexec -n  {mpi_ranks} ./a.out -c 6 -s {filesize} -k {chunksize}"
        p = subprocess.Popen(["sbatch", "../slurm-scripts/run-anything.sh" , call_create_hdf5, "False", call_create_hdf5_parallel, call_create_hdf5_async, call_create_hdf5_subfiling], cwd="./c-stuff")
        p.wait()
        
        df = runner_netcdf4_c(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_hdf5_c(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_netcdf4_c_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_hdf5_c_parallel(df, shape, chunksize, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_hdf5_c_async(df, shape, chunksize, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_hdf5_c_subfiling(df, shape, chunksize, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        
        index +=1
        df.to_json(path)
        
        
def bench_python(setup, df, variable, iterations, mpi_ranks, path):
    index = 0
    
    if os.path.exists("log/"):
        shutil.rmtree("log/")
    
    for run in setup.values():
        
        shape = run[variable][0]
        chunks = run[variable][1]
        
        total_filesize = calc_chunksize(chunks=shape)
        
        size_chunks = [None, None]
        if len(chunks) != 0:
            size_chunks = calc_chunksize(chunks=chunks, stop_depth="MB")
        
        print(f"{color.WARNING}create python Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        
        with open("data/tmp/run_config.json", "w") as f:
            json.dump(run, f)
        
        ## create Zarr, NetCDF4 and HDF5 file sequentially 
        
        call = "python benchmarks.py -c 1"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()
        
        call = "python benchmarks.py -c 2"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()
        
        call = "python benchmarks.py -c 3"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()
        
        df = runner_zarr(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_netcdf4(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_hdf5(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_zarr_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_netcdf4_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_hdf5_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        
        ## delete zarr python file
        if os.path.exists("data/datasets/test_dataset.zarr"):
            shutil.rmtree("data/datasets/test_dataset.zarr")
        
        ## delete hdf5 python file
        if os.path.exists("data/datasets/test_dataset.h5"):
            os.remove("data/datasets/test_dataset.h5")
        
        index +=1
        df.to_json(path)
        
        
def bench_c(setup, df, variable, iterations, mpi_ranks, path):
    index = 0
    
    if os.path.exists("c-stuff/log/"):
        shutil.rmtree("c-stuff/log/")
    
    for run in setup.values():
        
        shape = run[variable][0]
        chunks = run[variable][1]
        
        total_filesize = calc_chunksize(chunks=shape)
        
        size_chunks = [None, None]
        if len(chunks) != 0:
            size_chunks = calc_chunksize(chunks=chunks, stop_depth="MB")
        
        print(f"{color.WARNING}create python Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        
        with open("data/tmp/run_config.json", "w") as f:
            json.dump(run, f)
        
        ## create Zarr, NetCDF4 and HDF5 file sequentially 
        call = "python benchmarks.py -c 2"
        p = subprocess.Popen(["sbatch", "slurm-scripts/run-anything.sh" , call, "False"])
        p.wait()
        
        # setup metadata for c-runs
        filesize = shape[0]
        chunksize = 0
        
        if len(chunks) != 0:
            chunksize = chunks[0]
        
        print(f"{color.WARNING}create c Datasets for variable: {variable} with shape: {shape} and chunks: {chunks}, total filesize: {total_filesize[0]} {total_filesize[1]}, filesize per chunk: {size_chunks[0]} {size_chunks[1]}{color.ENDC}")
        
        call_create_hdf5 = f"./a.out -c 1 -s {filesize} -k {chunksize}"
        call_create_hdf5_parallel = f"mpiexec -n  {mpi_ranks} ./a.out -c 4 -s {filesize} -k {chunksize}"
        call_create_hdf5_async = f"mpiexec -n  {mpi_ranks} ./a.out -c 5 -s {filesize} -k {chunksize}"
        call_create_hdf5_subfiling = f"mpiexec -n  {mpi_ranks} ./a.out -c 6 -s {filesize} -k {chunksize}"
        p = subprocess.Popen(["sbatch", "../slurm-scripts/run-anything.sh" , call_create_hdf5, "False", call_create_hdf5_parallel, call_create_hdf5_async, call_create_hdf5_subfiling], cwd="./c-stuff")
        p.wait()
        
        df = runner_netcdf4_c(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_hdf5_c(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index)
        
        df = runner_netcdf4_c_parallel(df, shape, chunks, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_hdf5_c_parallel(df, shape, chunksize, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_hdf5_c_async(df, shape, chunksize, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        df = runner_hdf5_c_subfiling(df, shape, chunksize, variable, iterations, total_filesize, size_chunks, index, mpi_ranks)
        
        
        index +=1
        df.to_json(path)

        
def main():
    
    iterations = 5
    variable = "X"
    mpi_ranks = 32
    
    setup = {
            #"run01": {"X": ([1 * 134217728], [])},
            #"run02": {"X": ([2 * 134217728], [])},
            #"run03": {"X": ([3 * 134217728], [])},
            #"run04": {"X": ([4 * 134217728], [])},
            #"run05": {"X": ([5 * 134217728], [])},
            #"run06": {"X": ([6 * 134217728], [])},
            #"run07": {"X": ([7 * 134217728], [])},
            #"run08": {"X": ([8 * 134217728], [])},
            #"run09": {"X": ([9 * 134217728], [])},
            #"run10": {"X": ([10 * 134217728], [])},
            
            "run10": {"X": ([10 * 134217728], [])},
            "run11": {"X": ([20 * 134217728], [])},
            #"run11": {"X": ([25 * 134217728], [])},
            "run12": {"X": ([30 * 134217728], [])},
            "run13": {"X": ([40 * 134217728], [])},
            "run14": {"X": ([50 * 134217728], [])},
            "run15": {"X": ([60 * 134217728], [])},
            "run16": {"X": ([70 * 134217728], [])},
            #"run16": {"X": ([75 * 134217728], [])},
            #"run17": {"X": ([80 * 134217728], [])},
            #"run18": {"X": ([90 * 134217728], [])},
            #"run19": {"X": ([100 * 134217728], [])},
            
            #"run14": {"X": ([10 * 134217728], [int(10 * 134217728 / 16)])},
            #"run15": {"X": ([10 * 134217728], [int(10 * 134217728 / 32)])},
            #"run16": {"X": ([10 * 134217728], [int(10 * 134217728 / 64)])},
            #"run17": {"X": ([10 * 134217728], [int(10 * 134217728 / 128)])},
            #"run18": {"X": ([10 * 134217728], [int(10 * 134217728 / 256)])},
            
            
            #"run06": {"X": ([faktor * 512, 512, 512], [faktor * 512, 512, 128])},
            #"run07": {"X": ([faktor * 512, 512, 512], [faktor * 512, 512, 256])},
            #"run08": {"X": ([faktor * 512, 512, 512], [faktor * 512, 512, 512])},
            
            #"run11": {"X": ([512, 512, 512], [512, 1, 1])},
            #"run12": {"X": ([512, 512, 512], [512, 8, 8])},
            #"run13": {"X": ([512, 512, 512], [512, 16, 16]), "Y": ([10, 10], [2, 2])},
            #"run14": {"X": ([512, 512, 512], [512, 32, 32]), "Y": ([10, 10], [2, 2])},
            #"run15": {"X": ([512, 512, 512], [512, 64, 64]), "Y": ([10, 10], [2, 2])},
            #"run16": {"X": ([512, 512, 512], [512, 128, 128]), "Y": ([10, 10], [2, 2])},
            #"run17": {"X": ([512, 512, 512], [512, 256, 256]), "Y": ([10, 10], [2, 2])},
            #"run18": {"X": ([512, 512, 512], [512, 512, 512]), "Y": ([10, 10], [2, 2])}, 
            }
    
    df = pd.DataFrame()
    
    #bench_variable(setup, df=df, variable=variable, iterations=iterations, mpi_ranks=mpi_ranks, path="data/plotting/plotting_bench_variable.json")
    bench_python(setup, df=df, variable=variable, iterations=iterations, mpi_ranks=mpi_ranks, path="data/plotting/plotting_bench_python.json")
    #bench_c(setup, df=df, variable=variable, iterations=iterations, mpi_ranks=mpi_ranks, path="data/plotting/plotting_bench_c.json")
    
    

if __name__=="__main__":
    main()