from dataclasses import dataclass
import netCDF4, zarr, h5py, time
import numpy as np


@dataclass
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


@dataclass
class Datastruct:
    
    
    def __init__(self, path=None, shape=[], chunks=[], mode=None, engine=None, compression=None, dataset=any, log=False, parallel=False):
        self.path = path
        self.shape = shape
        self.chunks = chunks
        self.mode = mode
        self.engine = engine
        self.compression = compression
        self.dataset = dataset
        self.log = log
        self.parallel = parallel
        
    
    def create(self, path: str, form: dict, engine: str, parallel=False, dtype="f8"):

        if type(engine) == str:
            self.engine = engine
            
        self.parallel = parallel
                
            
        match self.engine:
            case "zarr":
                if self.parallel == False:
                    self.create_zarr(form=form, path=path, dtype=dtype)
                else:
                    self.create_zarr_parallel(form=form, path=path, dtype=dtype)
            
        
            case "hdf5":
                if self.parallel == False:
                    self.create_hdf5(form=form, path=path, dtype=dtype)
                else:
                    self.create_hdf5_parallel(form=form, path=path, dtype=dtype)
            
        
            case "netcdf4":
                if self.parallel == False:
                    self.create_netcdf4(form=form, path=path, dtype=dtype)
                else:
                    self.create_netcdf4_parallel(form=form, path=path, dtype=dtype)
                
                
        return self
 
    
    def create_zarr(self, form: dict, path: str, dtype: str):
        root = zarr.create_group(store=path, zarr_format=3, overwrite=True)
        
        for variable, element in form.items():
            shape = element[0]
            chunks = element[1]
            
            if len(chunks) != 0:
                x = root.create_array(name=variable, shape=shape, chunks=chunks, dtype=dtype)
            else: 
                x = root.create_array(name=variable, shape=shape, dtype=dtype)
            
            x[:] = np.random.random_sample(shape)
        self.dataset = root
        print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
        
        return self
    
    
    def create_zarr_parallel(self, form: dict, path: str, dtype: str):
        from mpi4py import MPI
        if MPI.COMM_WORLD.rank == 0: # type: ignore
            
            root = zarr.create_group(store=path, zarr_format=3, overwrite=True)
            
            for variable, element in form.items():
                shape = element[0]
                chunks = element[1]
                
                if len(chunks) != 0:
                    x = root.create_array(name=variable, shape=shape, chunks=chunks, dtype=dtype)
                else: 
                    x = root.create_array(name=variable, shape=shape, dtype=dtype)
                
                x[:] = np.random.random_sample(shape)
            self.dataset = root
            print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
    
        return self
    
    
    def create_hdf5(self, form: dict, path: str, dtype: str):
        if type(path) == str:
            self.path = path
        
        # Create file either through mpio or serial
        print(f"{bcolors.WARNING}Creating hdf5 file{bcolors.ENDC}")
        root = h5py.File(path, "w-")
            
            
        # Create dataset corresponding to the provide number of variables
        for variable, element in form.items():      
            shape = element[0]
            chunks = element[1]
            
            if len(chunks) != 0: 
                x = root.create_dataset(variable, shape=shape, chunks=tuple(chunks), dtype=dtype)
            else:
                x = root.create_dataset(variable, shape=shape, dtype=dtype)
            
            
            # File created dataset with values
            x[:] = np.random.random_sample(shape)   

        self.dataset = root
        root.close()
        print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
        
        return self
    
    
    def create_hdf5_parallel(self, form: dict, path: str, dtype: str):
        from mpi4py import MPI
        
        if type(path) == str:
            self.path = path
        
        # Create file either through mpio or serial
        print(f"{bcolors.WARNING}Creating hdf5 file{bcolors.ENDC}")
        root = h5py.File(path, "w-", driver="mpio", comm=MPI.COMM_WORLD) # type: ignore
            
            
        # Create dataset corresponding to the provide number of variables
        for variable, element in form.items():      
            shape = element[0]
            chunks = element[1]
            
            if len(chunks) != 0: 
                x = root.create_dataset(variable, shape=shape, chunks=tuple(chunks), dtype=dtype)
            else:
                x = root.create_dataset(variable, shape=shape, dtype=dtype)
            
            
            # File created dataset with values
            rank = MPI.COMM_WORLD.rank # type: ignore
            rsize = MPI.COMM_WORLD.size # type: ignore
            total_size = shape[0]
            size = int(total_size / rsize)
            
            rstart = rank * size
            rend = rstart + size
            
            x[rstart:rend:] = np.random.random_sample(size)
            MPI.COMM_WORLD.Barrier() # type: ignore

        self.dataset = root
        root.close()
        print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
        
        return self
    
    
    def create_netcdf4(self, form: dict, path: str, dtype: str):
        if type(path) == str:
            self.path = path
                    
                
        root = netCDF4.Dataset(path, "w", format="NETCDF4")  # type: ignore
        root.createGroup("/")
        used = 0
        
        for variable, element in form.items():
            shape = element[0]
            chunks = element[1]
            dimensions = []
            
            for size in shape:
                root.createDimension(f"{used}", size)
                dimensions.append(f"{used}")
                used += 1
            
            if len(chunks) != 0: 
                x = root.createVariable(variable, dtype, dimensions, chunksizes=chunks)
            else: 
                x = root.createVariable(variable, dtype, dimensions)
            
            
            x[:] = np.random.random_sample(shape)
         
                  
        self.dataset = root
        root.close()
        print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
        
        return self
    
    
    def create_netcdf4_parallel(self, form: dict, path: str, dtype: str):
        if type(path) == str:
            self.path = path
                    
                
        root = netCDF4.Dataset(path, "w", format="NETCDF4", parallel=True)  # type: ignore
        root.createGroup("/")
        used = 0
        
        for variable, element in form.items():
            shape = element[0]
            chunks = element[1]
            dimensions = []
            
            for size in shape:
                root.createDimension(f"{used}", size)
                dimensions.append(f"{used}")
                used += 1
            
            if len(chunks) != 0: 
                x = root.createVariable(variable, dtype, dimensions, chunksizes=chunks)
            else: 
                x = root.createVariable(variable, dtype, dimensions)
            

            rank = MPI.COMM_WORLD.rank  # type: ignore
            rsize = MPI.COMM_WORLD.size  # type: ignore
            total_size = shape[0]
            size = int(total_size / rsize)
            
            rstart = rank * size
            rend = rstart + size
            
            x.set_collective(True)
            x[rstart:rend:] = np.random.random_sample(size)
            MPI.COMM_WORLD.Barrier()  # type: ignore
                  
                  
        self.dataset = root
        root.close()
        print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
        
        return self
        
        
    def open(self, mode: str, engine: None | str, path: None | str, parallel=False):
        
        self.parallel = parallel
        
        if self.parallel == True:
            from mpi4py import MPI
        
        if type(path) == str:
            self.path = path
            
        if type(engine) == str:
            self.engine = engine
        
        if type(mode) == str:
            self.mode = mode
            
        match self.engine:
            case "zarr":
                
                self.dataset = zarr.open(self.path, mode=self.mode ,zarr_version=3)                    
                                        
            case "hdf5":
                
                if self.parallel: 
                    self.dataset = h5py.File(self.path, mode=self.mode, driver="mpio", comm=MPI.COMM_WORLD)  # type: ignore
                else:
                    self.dataset = h5py.File(self.path, mode=self.mode)  # type: ignore                  
                    
            case "netcdf4":  
                self.dataset = netCDF4.Dataset(self.path, mode=self.mode, format="NETCDF4", parallel=self.parallel)  # type: ignore

                               
        return self

                
    def __bench_variable(self, variable: list, iterations: int):
        bench = []
        
        match self.engine:
            case "zarr":
                
                size = {self.dataset[variable].shape[0]}  # type: ignore
                
                for i in range(iterations):
                    print(f"i: {i} for variable: {variable} for engine: {self.engine}, size: {size}")
                    start = time.monotonic()
                    self.dataset[variable][:]  # type: ignore
                    bench.append(time.monotonic() - start)
                
                self.log = bench
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
                
            case "hdf5":
                
                size = {self.dataset[variable].shape[0]}  # type: ignore
                
                for i in range(iterations):
                    print(f"i: {i} for variable: {variable} for engine: {self.engine}, size: {size}")
                    start = time.monotonic()
                    self.dataset[variable][:]  # type: ignore
                    bench.append(time.monotonic() - start)
                
                self.dataset.close()  # type: ignore
                self.log = bench
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
                
            case "netcdf4":
                
                size = {self.dataset[variable].shape[0]}  # type: ignore
                
                for i in range(iterations):
                    print(f"i: {i} for variable: {variable} for engine: {self.engine}, size: {size}")
                    start = time.monotonic()
                    self.dataset[variable][:]  # type: ignore
                    bench.append(time.monotonic() - start)
                
                self.dataset.close()  # type: ignore
                self.log = bench
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
 
 
    def __bench_variable_parallel(self, variable: list, iterations: int):
        from mpi4py import MPI
        
        bench = []
        
        rank = MPI.COMM_WORLD.rank
        rsize = MPI.COMM_WORLD.size
        
        match self.engine:
            case "zarr":
                
                for i in range(iterations):
                    size = {self.dataset[variable].shape[0]}  # type: ignore
                    print(f"i: {i} for variable: {variable} for engine: {self.engine}, rank: {rank}, size: {size}")
                    
                    if rank == 0:
                        start = time.monotonic()
                    
                    total_size = self.dataset[variable].shape[0]  # type: ignore
                    size = int(total_size / rsize)
                    
                    rstart = rank * size
                    rend = rstart + size
                    self.dataset[variable][rstart:rend:]  # type: ignore
                    
                    if rank == 0: 
                        bench.append(time.monotonic() - start)  # type: ignore
                    MPI.COMM_WORLD.Barrier()
                
                self.log = bench
                
                MPI.COMM_WORLD.Barrier()
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
                
            case "hdf5":
                
                for i in range(iterations):
                    size = {self.dataset[variable].shape[0]}  # type: ignore
                    print(f"i: {i} for variable: {variable} for engine: {self.engine}, rank: {rank}, size: {size}")
                    
                    if rank == 0:
                        start = time.monotonic()
                    
                           
                    total_size = self.dataset[variable].shape[0]  # type: ignore
                    size = int(total_size / rsize)
                    
                    rstart = rank * size
                    rend = rstart + size
                    
                    self.dataset[variable][rstart:rend:]  # type: ignore
                    
                    if rank == 0:
                        bench.append(time.monotonic() - start)  # type: ignore
                    MPI.COMM_WORLD.Barrier()
                
                if rank == 0:
                    self.log = bench
                
                self.dataset.close()  # type: ignore
                MPI.COMM_WORLD.Barrier()
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
                
            case "netcdf4":
                
                for i in range(iterations):
                    size = {self.dataset[variable].shape[0]}  # type: ignore
                    print(f"i: {i} for variable: {variable} for engine: {self.engine}, rank: {rank}, size: {size}")
                    
                    if rank == 0:
                        start = time.monotonic()
                    
                    self.dataset[variable].set_collective(True)  # type: ignore
                    
                    total_size = self.dataset[variable].shape[0]  # type: ignore
                    size = int(total_size / rsize)
                    
                    rstart = rank * size
                    rend = rstart + size
                    
                    self.dataset[variable][rstart:rend:]  # type: ignore
                    
                    if rank == 0:
                        bench.append(time.monotonic() - start)  # type: ignore
                        
                    MPI.COMM_WORLD.Barrier()
                
                if rank == 0:
                    self.log = bench
                
                self.dataset.close()  # type: ignore  
                MPI.COMM_WORLD.Barrier()
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
  
  
    def __bench_complete(self, variable: list, iterations: int):
        bench = []
        size = []
        var_tmp = []
         
        match self.engine:
            case "zarr":
                
                for var in variable:
                    try:
                        size.append({self.dataset[var].shape[0]})  # type: ignore
                        var_tmp.append(var)
                    except KeyError:
                        print(f"Variable: {var} does not exist.")
                
                for i in range(iterations):
                    print(f"i: {i} for variable: {var_tmp} for engine: {self.engine}, size: {size}")
                    start = time.monotonic()
                    
                    for var in variable:
                        try:
                            self.dataset[var][:]  # type: ignore
                        except KeyError:
                            print(f"Variable: {var} does not exist.")
                    
                    bench.append(time.monotonic() - start)
                
                self.log = bench
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
                
            case "hdf5":
                for var in variable:
                    try:
                        size.append({self.dataset[var].shape[0]})  # type: ignore
                        var_tmp.append(var)
                    except KeyError:
                        print(f"Variable: {var} does not exist.")
                
                for i in range(iterations):
                    print(f"i: {i} for variable: {var_tmp} for engine: {self.engine}, size: {size}")
                    start = time.monotonic()
                    
                    for var in variable:
                        try:
                            #self.dataset[variable].read_direct(arr)
                            self.dataset[var][:]  # type: ignore
                        except KeyError:
                            print(f"Variable: {var_tmp} does not exist.")
                        
                    bench.append(time.monotonic() - start)
                
                self.dataset.close()  # type: ignore
                self.log = bench
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
                
            case "netcdf4":
                for var in variable:
                    try:
                        size.append({self.dataset[var].shape[0]})  # type: ignore
                        var_tmp.append(var)
                    except IndexError:
                            print(f"Variable: {var} does not exist.")    
                
                for i in range(iterations):
                    print(f"i: {i} for variable: {variable} for engine: {self.engine}, size: {size}")
                    start = time.monotonic()
                    
                    for var in variable:
                        try:
                            self.dataset[var][:]  # type: ignore
                        except IndexError:
                            print(f"Variable: {var} does not exist.")
                    
                    bench.append(time.monotonic() - start)
                
                self.dataset.close()  # type: ignore
                self.log = bench
                print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
 
 
    def __bench_complete_parallel(self, variable: list, iterations: int):  
        match self.engine:
            case "zarr":
                self.__bench_complete_parallel_zarr(variable=variable, iterations=iterations)
                
            case "hdf5":
                self.__bench_complete_parallel_hdf5(variable=variable, iterations=iterations)
                
            case "netcdf4":
                self.__bench_complete_parallel_netcdf4(variable=variable, iterations=iterations)
 
    
    def __bench_complete_parallel_zarr(self, variable: list, iterations: int):
        bench = []
        size = []
        var_tmp = []
        
        from mpi4py import MPI  
        rank = MPI.COMM_WORLD.rank
        rsize = MPI.COMM_WORLD.size
        
        for i in range(iterations):
            if rank == rank:
                for var in variable:
                    
                    try:
                        size.append({self.dataset[var].shape[0]})  # type: ignore
                        var_tmp.append(var)
                    except KeyError:
                        print(f"Variable: {var} does not exist.")
            print(f"i: {i} for variable: {var_tmp} for engine: {self.engine}, rank: {rank}, size: {size}")
            
            if rank == 0:
                start = time.monotonic()
            
            for var in variable:
                
                try:
                    total_size = self.dataset[var].shape[0]  # type: ignore
                    size = int(total_size / rsize)

                    rstart = rank * size
                    rend = rstart + size
                    self.dataset[var][rstart:rend:]  # type: ignore
                except KeyError:
                        print(f"Variable: {var} does not exist.")
                
            if rank == 0: 
                bench.append(time.monotonic() - start)  # type: ignore
                
            MPI.COMM_WORLD.Barrier()
        
        if rank == 0:        
            self.log = bench
        
        MPI.COMM_WORLD.Barrier()
        print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
     
        
    def __bench_complete_parallel_hdf5(self, variable: list, iterations: int): 
        bench = []
        size = []
        var_tmp = []
        
        from mpi4py import MPI  
        rank = MPI.COMM_WORLD.rank
        rsize = MPI.COMM_WORLD.size
        
        for i in range(iterations):
            for var in variable:
                try:
                    size.append({self.dataset[var].shape[0]})  # type: ignore
                    var_tmp.append(var)
                except KeyError:
                    print(f"Variable: {var} does not exist.")
            print(f"i: {i} for variable: {var_tmp} for engine: {self.engine}, rank: {rank}, size: {size}")
            
            if rank == 0:
                start = time.monotonic()
            
            for var in variable:
                
                try:
                    total_size = self.dataset[var].shape[0]  # type: ignore
                    size = int(total_size / rsize)

                    rstart = rank * size
                    rend = rstart + size

                    self.dataset[var][rstart:rend:]  # type: ignore
                except KeyError:
                    print(f"Variable: {var} does not exist.")
            
            if rank == 0: 
                bench.append(time.monotonic() - start)  # type: ignore
                
            MPI.COMM_WORLD.Barrier()
                
        if rank == 0:
            self.log = bench
         
        self.dataset.close()  # type: ignore 
        MPI.COMM_WORLD.Barrier()
        print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
        
        
    def __bench_complete_parallel_netcdf4(self, variable: list, iterations: int): 
        bench = []
        size = []
        var_tmp = []
        
        from mpi4py import MPI  
        rank = MPI.COMM_WORLD.rank
        rsize = MPI.COMM_WORLD.size
        
        for i in range(iterations):
            for var in variable:
                try:
                    size.append({self.dataset[var].shape[0]})  # type: ignore
                    var_tmp.append(var)
                except IndexError:
                    print(f"Variable: {var} does not exist.")
            print(f"i: {i} for variable: {var_tmp} for engine: {self.engine}, rank: {rank}, size: {size}")
            
            
            if rank == 0:
                start = time.monotonic()
            
            for var in variable:
                
                try:
                    self.dataset[var].set_collective(True)  # type: ignore
                    total_size = self.dataset[var].shape[0]  # type: ignore
                    size = int(total_size / rsize)

                    rstart = rank * size
                    rend = rstart + size

                    self.dataset[var][rstart:rend:]  # type: ignore
                except IndexError:
                    print(f"Variable: {var} does not exist.")
            
            if rank == 0:
                bench.append(time.monotonic() - start)  # type: ignore
                
            MPI.COMM_WORLD.Barrier()
                
        if rank == 0:
            self.log = bench
        
        self.dataset.close()  # type: ignore
        MPI.COMM_WORLD.Barrier()
        print(f"{bcolors.OKGREEN}FINISHED{bcolors.ENDC}")
    
            
    def read(self, pattern: str, variable: str, iterations: int, logging=False):
        
        patterns = {
            "bench_variable": self.__bench_variable,
            "bench_complete": self.__bench_complete,
            "bench_variable_parallel": self.__bench_variable_parallel,
            "bench_complete_parallel": self.__bench_complete_parallel,
        }

        return patterns[pattern](variable=variable.split(","), iterations=iterations)