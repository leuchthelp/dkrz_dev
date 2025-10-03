from func.dev_utils import calc_size_unit
from func.datastruct import bcolors
from spackmanager import SpackManager
from dataclasses import dataclass, asdict
from pathlib import Path
import shutil
import yaml
import hashlib
import subprocess
import os

@dataclass
class BenchmarkManager:
    
    """Defines handler to read configuration from yaml file and create matching benchmarks. Also configures the benchmark environments and gathers system information.
    
    Attributes
    ----------
    handler_id: str
        Unique handler ID to identify the handler assigned to this specific benchmark.
        
    id: str
        Unique ID to identify this specific benchmark with.
        
    run_config: dict
        The run configuration that was requested, contains the basic structure of a file that will be created.
        
    bm_config: dict
        The configuration of the benchmark file with all it's adjacent information like source code, commands and such.
        
    nodes: int
        Number of nodes used with the context of a slurm environment, otherwise always 1.
        
    parallel: bool
        If parallelism is enabled.
        
    par_backend: None | str
        What kind of backend is being used to facilitate parallelism. Will be "None" if parallelism is not requested.
        
    ranks: int
        Represents how many cores are used to execute the benchmark following the MPI terminology. Will always be 1 for serial benchmarks.
        
    collective: None | bool
        If collective MPI I/O is requested, default is "False" for independent I/O. Will be none for serial benchmarks.
        
    langauge: str

    format: str
        Simplified representation of the kind of benchmark that was requested.
        
    engine: str
        Represent the file format following the xarray terminology.
        
    extension: str
        File extension used for the created file.
        
    datatype: list
        Datatypes of each variable / dataset within the file.
        
    var_to_bm: str | list
        Which variable / dataset will be benchmarked from the file. Can either be a single value string or a list of strings.
        
    total_filesize: int
        The total filesize benchmarked calculated from all variables / datasets were requested for benchmarking.
        
    unit: str
        Unit for the total filesize, i.e. MB, GB, etc.
        
    filesize_var: list
        filesize per variable / dataset.
        
    chunksize_var: list
        filesize per variable / dataset chunks.
        
    iterations: int
        total iterations performed. In the context of a slurm environment equal to the number of unique node allocations performed.
        
    internal_i: int
        The benchmark itself can have iterations to be performed as well, this would result in caching of files on the nodes used in the context of a slurm environment.
    """
    
    handler_id      : str
    id              : str
    run_config      : dict
    bm_config       : dict
    nodes           : int
    parallel        : bool
    par_backend     : None | str
    ranks           : int
    collective      : None | bool
    language        : str
    format          : str
    engine          : str
    extension       : str
    datatype        : list
    var_to_bm       : str | list
    total_filesize  : int
    unit            : str
    filesize_var    : list
    chunksize_var   : list
    iterations      : int
    internal_i      : int
    
    
    def __init__(self, 
                 handler_id     : str, 
                 run_config     : dict,
                 bm_config      : dict,
                 nodes          : int,
                 slurm_options  : None | str,
                 parallel       : bool,
                 collective     : None | bool, 
                 ranks          : int, 
                 var_to_bm      : str | list,
                 iterations     : int, 
                 use_path       : Path, 
                 results_path   : Path,
                 requested      : dict, 
                 ):
        
        
        # Object config
        self.handler_id     = handler_id
        self.bm_config      = bm_config
        self.sbatch_location= "slurm config"
        self.slurm_options  = slurm_options
        
        # Source code
        try:
            self.create     = bm_config["create"]
        except:
            self.create     = ""
        
        try:
            self.compile    = bm_config["compile"]
            try:
                self.compile_command = bm_config["compile_command"]
            except:
                raise ValueError("Missing compile command for benchmark requiring compilation")
        except KeyError:
            self.compile    = False
            
            
        self.src            = bm_config["source"]
        
        
        # Benchmark config
        self.run_config     = run_config
        self.nodes          = nodes
        self.parallel       = parallel
        self.par_backend    = requested["par_backend"]
        self.collective     = collective
        self.ranks          = ranks
        self.language       = requested["language"]
        self.format         = requested["format"]
        self.engine         = f"{self.format}-{self.language}-parallel" if self.parallel == True else f"{self.format}-{self.language}"
        self.extension      = bm_config["extension"]
        
        try:
            config_par_backend = self.bm_config["par_backend"]
        except:
            config_par_backend = None
        
        
        datatype    = []
        for _, item in run_config.items():
            if any(isinstance(x, str) for x in item):
                datatype.append(item[-1])
            else:
                datatype.append("f8")
        self.datatype       = datatype
        
        self.var_to_bm      = var_to_bm
        self.iterations     = iterations
        self.internal_i     = 1
        self.no_caching     = False
        self.local          = False
        
        
        # Assemble ID
        id_str = (str(self.run_config) 
                  + str(config_par_backend) 
                  + str(self.par_backend) 
                  + str(self.bm_config["parallel"]) 
                  + str(self.parallel) 
                  + str(self.bm_config["format"])
                  + str(self.format)
                  + str(self.ranks) 
                  + str(self.var_to_bm)
                  + str(self.collective)
                  + str(self.nodes)
                  
                  # Reasoning: If source code changes, do not consider the same benchmark even if it might be functionally the same, could still have an effect in performance
                  + self.src
                  )
        self.id             = hashlib.sha256(id_str.encode()).hexdigest()
        
        self.use_path       = use_path
        self.results_path   = results_path
        self.dir_path       = Path(f"{self.use_path}/{str(self.id)}")
        
        
        # Benchmark info 
        self.location       = f"{self.id}.{self.extension}"
        
        filesize_per_var    = [(key, calc_size_unit(item[0])) for key, item in run_config.items() if key in self.var_to_bm] 
        
        total_filesize = 0
        for filesize in filesize_per_var:
            total_filesize += filesize[1][0]
        
        self.total_filesize = total_filesize  
        self.unit           = filesize_per_var[0][1][1] 
        self.filesize_var   = filesize_per_var   
        self.chunksize_var  = [(key, calc_size_unit(item[1])) for key, item in run_config.items() if key in self.var_to_bm] 
        self.show_metdata   = True
        
        
        # Environment config
        self.__checkpoint   = yaml
        self.__node         = int
        self.__node_info    = yaml
        self.__profiler     = bool
        
        
        print(bcolors.OKBLUE +  f"Managing Benchmark with; file-structure: {run_config}, "
                                f"nodes: {self.nodes}, "
                                f"datatype: {self.datatype}, "
                                f"parallel: {self.parallel}, " 
                                f"collective: {self.collective}, "
                                f"ranks: {self.ranks}, "
                                f"par_backend: {self.par_backend}, "
                                f"language: {self.language}, "
                                f"format: {self.format}, " 
                                f"{self.iterations} iterations. " 
                                f"It will be stored in {self.use_path}" + bcolors.ENDC
            )   
    
    
    def run(self):
        self.dir_path.mkdir(parents=True)

        try:
            self_dict = asdict(self)  

            if self.show_metdata:
                with open(f"{self.dir_path}/metadata.yaml", "w") as f:
                    yaml.safe_dump(self_dict, f)


            if self.create != None:  
                self.__create_file()


            self.__execute_file()
        
        finally:
            shutil.rmtree(path=self.dir_path)
            pass
            
        return self.id, self_dict

 
    def __create_file(self):
        create = self.create.replace("#MAIN", self.__replace_main(self.language))
        
        path_to_create_file = Path(f"{self.dir_path}/create.{self.language}")
        with open(path_to_create_file, "w") as file:
            file.write(create)
        
        
        create_file = f"create.{self.language}"
        if self.compile == True:
            compiled_file = self.__compile_file(path=path_to_create_file)
            create_file = f"./{compiled_file}"
        
        
        create_commands = self.bm_config["create_command"]
        
        create_command = ""
        try:
            create_command  = create_commands["serial"]
        except KeyError as e:
            if self.bm_config["parallel"] == True:
                pass
            else: 
                raise e
        
        
        if self.par_backend in create_commands.keys():
            create_command = create_commands[str(self.par_backend)]
            create_command = create_command + "-p"
            create_command = create_command.replace("-n ", f"-n {self.ranks} ")
            
            if self.collective == True:
                create_command = create_command + f"-I {self.collective}"


        create_command = create_command.replace("{runnable}", f"{create_file} ")
        create_command = create_command.replace("-p", f"-p {self.parallel} ")
        
        
        if "-c" not in create_command:
            create_command = create_command + " -c 1"
        
            
        if "-l" not in create_command:
            create_command = create_command + " -l"
                
        create_command = create_command.replace("-l", f"-l {self.location}")
        
        
        # Transform run config into 4 lists; variables (list(string)), shape (list(list(int))), chunks (list(list(int))) & datatypes (list(string)) 
        
        flag_variable = "-V"
        if flag_variable not in create_command:
            create_command = create_command + f" {flag_variable}"
        
        variables = ",".join(list(self.run_config.keys()))
        create_command = create_command.replace(f"{flag_variable}", f"{flag_variable} {variables}")
        
        values = list(self.run_config.values())
        shapes = []
        chunks = []
        datatypes = self.datatype
        for value in values:
            shapes.append(value[0])
            chunks.append(value[1])
            
        
        create_command = self.__append_flag(flag="-S", command=create_command, data=shapes)
        create_command = self.__append_flag(flag="-C", command=create_command, data=chunks)
        create_command = self.__append_flag(flag="-D", command=create_command, data=datatypes)
        
        
        if  "SLURM_JOB_ID" in os.environ and self.local == False:
            create_command = ["sbatch", self.__assemble_sbatch(self.sbatch_location, self.slurm_options), create_command] # type: ignore
        else: 
            create_command = create_command.split()

        print(create_command)
        p = subprocess.run(create_command, capture_output=True, text=True, cwd=self.dir_path)
        print(p.stderr)
        print(p.stdout)
    
    
    def __append_flag(self, flag: str, command: str, data: list):
        if flag not in command:
            command = command + f" {flag}"
            
        data_str = ",".join(str(x) for x in data)
        command = command.replace(f"{flag}", f"{flag} {data_str}")
        
        return command
    

    def __compile_file(self, path: Path):
        
        compile_command = self.compile_command.replace("{runnable}", f"{path.absolute()}")
        
        compiled_file = f"{path.name}.out"
        compile_command = compile_command + " -Wl,--unresolved-symbols=ignore-in-object-files" + f" -o {compiled_file}"
        
        print(compile_command)
        p = subprocess.run(compile_command.split(), capture_output=True, text=True, cwd=self.dir_path, check=True)
        #print(p.stderr)
        #print(p.stdout)
        
        return compiled_file
  

    def __execute_file(self):
        
        execute = self.src.replace("#MAIN", self.__replace_main(self.language))
        
        path_to_tmp_file = Path(f"{self.dir_path}/execute.{self.language}")
        with open(path_to_tmp_file, "w") as file:
            file.write(execute)
        
        
        tmp_file    = f"execute.{self.language}"
        if self.compile == True:
            compiled_file = self.__compile_file(path=path_to_tmp_file)
            tmp_file = f"./{compiled_file}"
        
        
        run_commands = self.bm_config["run_command"]
        
        run_command = ""
        try:
            run_command  = run_commands["serial"]
        except KeyError as e:
            if self.bm_config["parallel"] == True:
                pass
            else: 
                raise e
        
        if self.par_backend in run_commands.keys():
            run_command = run_commands[str(self.par_backend)]
            run_command = run_command +  "-p"
            run_command = run_command.replace("-n", f"-n {self.ranks} ")
            
            if self.collective == True:
                run_command = run_command + f"-I {self.collective}"

        
        run_command = run_command.replace("{runnable}", f"{tmp_file} ")
        run_command = run_command.replace("-i", f"-i {self.internal_i} ")
        run_command = run_command.replace("-p", f"-p {self.parallel} ")
        
        
        if "-b" not in run_command:
            run_command = run_command + "-b 1 "
            
        if "-l" not in run_command:
            run_command = run_command + f"-l {self.location} "
            
        vars_to_bm = ",".join(self.var_to_bm)
        if "-v" not in run_command:
            run_command = run_command + "-v"
            
        run_command = run_command.replace("-v", f"-v {vars_to_bm} ")
        

        if self.language == "c":
            
            size = []
            for var in self.var_to_bm:
                size.append(self.run_config[var][0])
            
            run_command = run_command + f"-s {sum([sum(x) for x in size])}"


        if  "SLURM_JOB_ID" in os.environ and self.local is False:
            run_command = ["sbatch", self.__assemble_sbatch(self.sbatch_location, self.slurm_options), run_command] # type: ignore


        print(run_command)
        for i in range(self.iterations):
            
            if  "SLURM_JOB_ID" in os.environ and self.local == False:
                p = subprocess.run(run_command, capture_output=True, text=True, cwd=self.dir_path)
                print(p.stderr)
                print(p.stdout)
            else: 
                p = subprocess.run(run_command.split(), capture_output=True, text=True, cwd=self.dir_path)   # type: ignore
                print(p.stderr)
                print(p.stdout)

                if self.no_caching == True:
                    new_path = Path(f"{self.dir_path}/{i}")
                    new_path.mkdir(parents=True)
                    
                    current_path = Path()
                    for path in self.dir_path.rglob(f"*.{self.extension}"):
                        current_path = path
                        
                    new_file_location = shutil.move(current_path.absolute(), f"{new_path.absolute()}/{i}.{self.extension}")  
                    
                    #print(f"current location: {self.location} -> new location: {new_file_location}")
                    
                    run_command = run_command.replace(f"-l {self.location}", f"-l {new_file_location}")   # type: ignore
                    #print(f"new run command: {run_command}")
                    self.location = new_file_location
                
        
    def __replace_main(self, language: str) -> str:  # type: ignore
        
        match language:
            
            ##################################################################################################
            #### Py Part to be injected for #MAIN
            ##################################################################################################
            
            case "py":
                return f"""
import ast
            
def main():

    parser = argparse.ArgumentParser(
        prog="Python Dataformat-Benchmark",
        description="run python based benchmark for Zarr, NetCDF4 and HDF5",
    )
    parser.add_argument("-c", "--create", type=int, default=-1, help="creates Zarr, NetCDF4 and HDF5 Files using a previously saved run format")
    parser.add_argument("-b", "--benchmark", type=int, default=-1, help="benchmark to run")
    parser.add_argument("-v", "--var_to_bm", type=str, default=None, help="var_to_bm to read, if none is provided all are read")
    parser.add_argument("-V", "--variable", type=str, default=None, help="variables to create")
    parser.add_argument("-S", "--shape", type=str, default=None, help="shapes per variable to create")
    parser.add_argument("-C", "--chunk", type=str, default=None, help="chunks per variable to create")
    parser.add_argument("-D", "--datatype", type=str, default=None, help="datatype per variable to create")
    parser.add_argument("-p", "--parallel", type=bool, default=False, help="If to run the benchmark using parallelism of any kind supported")
    parser.add_argument("-i", "--iterations", type=int, default=1, help="number of internal iterations to run the benchmark for. Will cause caching effects")
    parser.add_argument("-l", "--location", type=str, default="", help="Location where file will be create / saved")
    parser.add_argument("-I", "--input_output", type=bool, default=False, help="Set I/O to either use independent (default or False) or collective I/O (True)")
    args = parser.parse_args()
    
    match args.benchmark:
        case 1:
            result = bench(iterations=args.iterations, 
                            variable=args.var_to_bm, 
                            parallel=args.parallel, 
                            path=args.location, 
                            collective=args.input_output
                            )
                    
            from mpi4py import MPI
            if args.parallel is False or MPI.COMM_WORLD.rank == 0:
                from pathlib import Path
                if Path("{self.results_path.absolute()}/{self.id}.json").exists():
                    with open("{self.results_path.absolute()}/{self.id}.json", "r") as t:
                        result.extend(json.load(t))    

                with open("{self.results_path.absolute()}/{self.id}.json", "w") as f:
                    json.dump(result, f)
        case -1:
            variables   = args.variable.split(",")
            shapes      = [ast.literal_eval(e) for e in args.shape.split(",")]
            chunks      = [ast.literal_eval(e) for e in args.chunk.split(",")]
            datatypes   = args.datatype.split(",")
            
            if args.create != -1:
                create(variables=variables, 
                        shapes=shapes, 
                        chunks=chunks, 
                        datatypes=datatypes, 
                        parallel=args.parallel, 
                        path=args.location, 
                        collective=args.input_output
                        )

if __name__=="__main__":
    main()
        """
            
            ##################################################################################################
            #### C Part to be injected for #MAIN
            ##################################################################################################
        
            case "c":
                tmp = """
            
#include <unistd.h>
#include <argp.h>
#include <stdio.h>
#include <string.h>

void save_list_to_json(double *arr, char *file_name, size_t size)
{
    FILE *fptr;

    fptr = fopen(file_name, "r+");
    
    if (fptr == NULL)
    {
        fptr = fopen(file_name, "w");
        
        if (fptr == NULL)
        {
            fprintf(stderr, "cannot open target file %s\\n", file_name);
            exit(1);
        }
    }
    
    if(fgetc(fptr) != 91){     
        fprintf(fptr, "[");
    } else {
        (void)0;
    }
    
    int ch;
    while ((ch = fgetc(fptr)) != EOF)
    {   
        if (ch == ']')
        {
            fseek(fptr, -1, SEEK_CUR);
            fputc(',',fptr);
            fseek(fptr, 0, SEEK_CUR);
        }
    }

    for (size_t i = 0; i < size; i++)
    {
        fprintf(fptr, "%f", arr[i]);

        if (i != size - 1)
        {
            fprintf(fptr, ",");
        }
    }

    fprintf(fptr, "]");

    fclose(fptr);
}


typedef struct args_t
{
    int create;
    int benchmark;
    char *var_to_bm;
    char *variable;
    hsize_t size;
    char *shape;
    char *chunk;
    char *datatype;
    int parallel;
    int input_output;
    int iterations;
    char *location;
} args_t;

static int parse_opt(int key, char *arg, struct argp_state *state)
{
    args_t *arguments = state->input;

    switch (key)
    {
    case 'c':
        arguments->create = atoi(arg);
        break;
    case 'b':
        arguments->benchmark = atoi(arg);
        break;
    case 'v':
        arguments->var_to_bm = arg;
        break;
    case 'V':
        arguments->variable = arg;
        break;
    case 's':
        arguments->size = strtoull(arg, NULL, 10);
        break;
    case 'S':
        arguments->shape = arg;
        break;
    case 'C':
        arguments->chunk = arg;
        break;
    case 'D':
        arguments->datatype = arg;
        break;
    case 'p':
        arguments->parallel = strtoull(arg, NULL, 10);
        break;
    case 'i':
        arguments->iterations = atoi(arg);
        break;
    case 'I':
        arguments->input_output = atoi(arg);
        break;
    case 'l':
        arguments->location = arg;
        break;
    case ARGP_KEY_ARG:
        return 0;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp_option options[] = {
    {"create file", 'c', "NUM", 0, "If to create a file"},
    {"benchmark",   'b', "NUM", 0, "If to run benchmark"},
    {"var_to_bm",   'v', "c",   0, "Variables within a file to benchmark"},
    {"variable",    'V', "c",   0, "Variables the file should contain"},
    {"size",        's', "NUM", 0, "Specifiy the size of the file to read"},
    {"shape",       'S', "c",   0, "Specifiy the shapes of the file you want to create as list of lists"},
    {"chunk",       'C', "c",   0, "Specifiy the chunksize of the file you want to create as list of lists"},
    {"datatype",    'D', "c",   0, "Data types each variable should have as list"},
    {"parallel",    'p', "NUM", 0, "If to use parallelism or not"},
    {"iterations",  'i', "NUM", 0, "Ammount of iterations the benchmark should run"},
    {"input-output",'I', "NUM", 0, "Set I/O to either use independent (default or False) or collective I/O (True)"},
    {"location",    'l', "c",   0, "Location where file is going to be created / read from"},
    {0}};

hsize_t word_count(char *smth, char delim)
{
    hsize_t count = 1;
    for (hsize_t x = 0; x < strlen(smth); x++)
        if (smth[x] == delim)
            count++;
    return count;
}

int get_chars(char *smth, hsize_t amount, char **buf)
{
    hsize_t i = 0;
    char *token;
    char *rest = smth;

    while ((token = strtok_r(rest, ",", &rest)))
    {
        if (i == amount)
            break;
        buf[i] = calloc(strlen(token), sizeof(char *));
        strcpy(buf[i], token);
        i++;
    }
    return 0;
}

int get_list_contents(char *smth, hsize_t *buf)
{
    hsize_t i = 0;
    char tmp_char[CHAR_MAX] = "";
    bool flag = false;
    for (hsize_t x = 0; x < strlen(smth); x++)
    {
        if (smth[x] != 44) // ASCII ","
        {
            if (smth[x] == 91) // ASCII "["
                flag = true;
            if (smth[x] == 93) // ASCII "]"
                flag = false;

            if (flag == true && smth[x] != 91)
            {
                char tmp = smth[x];
                strncat(tmp_char, &tmp, 1);
            }
        }
        else
        {
            buf[i] = (hsize_t)strtoull(tmp_char, NULL, 10);
            tmp_char[0] = '\\0';
            i++;
        }
    }
    buf[i] = (hsize_t)strtoull(tmp_char, NULL, 10);
    return 0;
}

int get_individual_as_jagged(char *smth, hsize_t size, hsize_t **buf, hsize_t *jagged_size)
{
    char *token;
    char *rest = smth;

    hsize_t current_var = 0;
    while ((token = strtok_r(rest, "-", &rest)))
    {
        if (current_var == size)
            break;
        hsize_t dims = word_count(token, ',');

        buf[current_var] = calloc(dims, sizeof(hsize_t));
        int res = get_list_contents(token, buf[current_var]);
        jagged_size[current_var] = dims;
        current_var++;
    }
    return 0;
}

int main(int argc, char *argv[])
{
    struct argp argp = {options, parse_opt};

    args_t arguments;
    arguments.create = -1;
    arguments.benchmark = -1;
    arguments.var_to_bm = "[]";
    arguments.variable = "[]";
    arguments.size = 134217728;
    arguments.shape = "[]";
    arguments.chunk = "[]";
    arguments.datatype = "[]";
    arguments.parallel = 1;
    arguments.iterations = 1;
    arguments.location = "test.c";

    printf("Parsing: %d, var_to_bm: %s, variables: %s, shapes: %s, chunks: %s, datatypes: %s, parallel: %d, iterations: %d\\n", arguments.benchmark, arguments.var_to_bm, arguments.variable, arguments.shape, arguments.chunk, arguments.datatype, arguments.parallel, arguments.iterations);
    argp_parse(&argp, argc, argv, 0, 0, &arguments);
    
    hsize_t size = arguments.size;

    char *location = arguments.location;
    int iterations = arguments.iterations;
    int res;

    // get variables to benchmark
    hsize_t var_bm_count = word_count(arguments.var_to_bm, ',');

    // get variables
    hsize_t var_count = word_count(arguments.variable, ',');

    // get shapes
    hsize_t **shapes = calloc(var_count, sizeof(hsize_t *));
    hsize_t shapes_size[var_count];

    // get chunks
    hsize_t **chunks = calloc(var_count, sizeof(hsize_t *));
    hsize_t chunks_size[var_count];

    printf("Parsing: %d, parallel: %d, iterations: %d\\n", arguments.benchmark, arguments.parallel, arguments.iterations);

    // arguments parsing for creation of file
    switch (arguments.create)
    {
    case -1:
        break;
    case 1:

        // get variables
        char **variables = calloc(var_count, sizeof(char *));
        res = get_chars(arguments.variable, var_count, variables);


        // get shapes
        res = get_individual_as_jagged(arguments.shape, var_count, shapes, shapes_size);


        // get chunks
        res = get_individual_as_jagged(arguments.chunk, var_count, chunks, chunks_size);


        // get datatypes
        printf("word count: %ld\\n", var_count);
        char **datatypes = calloc(var_count, sizeof(char *));
        res = get_chars(arguments.datatype, var_count, datatypes);


        printf("Creating hdf5 file\\n");
        create(argc, argv, false, variables, shapes, chunks, datatypes, location);

        // Free variables, datatypes, shape and chunks
        for (int i = 0; i < var_count; i++)
        {
            free(variables[i]);
        }
        free(variables);

        for (int i = 0; i < var_count; i++)
        {
            free(datatypes[i]);
        }
        free(datatypes);

        for (int i = 0; i < var_count; i++)
        {
            free(shapes[i]);
        }
        free(shapes);

        for (int i = 0; i < var_count; i++)
        {
            free(chunks[i]);
        }
        free(chunks);

        break;
    case ARGP_KEY_ARG:
        return 0;
    default:
        return ARGP_ERR_UNKNOWN;
    }

    // arguments parsing for benchmarks
    switch (arguments.benchmark)
    {
    case -1:
        printf("No benchmark specified, exiting programm now \\n");
        break;
    case 1:
        // get variables to benchmark
        char **vars_to_bm = calloc(var_bm_count, sizeof(char *));
        res = get_chars(arguments.var_to_bm, var_bm_count, vars_to_bm);

        
        double * result = calloc(iterations, sizeof(double));

        printf("Running hdf5 benchmark for %d iterations reading %ld elements\\n", iterations, size);
        bench(argc, argv, size, vars_to_bm, iterations, location, result);
        
        save_list_to_json(result, "<result-path>.json", iterations);

        // Free variables to benchmark
        for (int i = 0; i < var_bm_count; i++)
        {
            free(vars_to_bm[i]);
        }
        free(vars_to_bm);
        free(result);
        break;
    case ARGP_KEY_ARG:
        return 0;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}
"""         
                tmp = tmp.replace("<result-path>", f"{self.results_path.absolute()}/{self.id}")
                return tmp
           

    def __assemble_sbatch(self, path: str, slurm_options: str):
        
        slurm_options.replace("#SBATCH --nodes=", "  #")
        
        if "#SBATCH --nodes=" not in slurm_options:
            slurm_options = slurm_options + f"#SBATCH --nodes={self.nodes}"
        
        sbatch_location = f"{path}.sh"
        with open(Path(f"{self.dir_path}/{sbatch_location}"), "w") as file:
            file.write(f"""#!/bin/bash

{slurm_options}
                       
# Begin of section with executable commands
set -e
ls -l

$1

mpi_enabled=$2
w=true

if [ "$mpi_enabled" = "$w" ]; then

    export OMPI_MCA_osc="ucx"
    export OMPI_MCA_pml="ucx"
    export OMPI_MCA_btl="self"
    export UCX_HANDLE_ERRORS="bt"
    export OMPI_MCA_pml_ucx_opal_mem_hooks=1
    
    export OMPI_MCA_io="romio321"          # basic optimisation of I/O
    export UCX_TLS="shm,rc_mlx5,rc_x,self" # for jobs using LESS than 150 nodes
    #export UCX_TLS="shm,dc_mlx5,dc_x,self" # for jobs using MORE than 150 nodes
    export UCX_UNIFIED_MODE="y"            
    
    export OMPI_MCA_coll_tuned_use_dynamic_rules="true"
    export OMPI_MCA_coll_tuned_alltoallv_algorithm=2
    
fi                     
"""
)
            
            return sbatch_location