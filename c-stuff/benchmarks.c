#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <netcdf_par.h>
#include <time.h>
#include <hdf5.h>
#include <h5_async_lib.h>
#include <unistd.h>
#include <argp.h>

#define ANSI_COLOR_RED "\x1b[31m"
#define ANSI_COLOR_GREEN "\x1b[32m"
#define ANSI_COLOR_YELLOW "\x1b[33m"
#define ANSI_COLOR_BLUE "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN "\x1b[36m"
#define ANSI_COLOR_RESET "\x1b[0m"

#define ERRCODE 2
#define ERR(e)                                 \
    {                                          \
        printf("Error: %s\n", nc_strerror(e)); \
        exit(ERRCODE);                         \
    }

void save_to_json(double *arr, char *file_name, char *name, size_t size)
{
    FILE *fptr;

    fptr = fopen(file_name, "w");

    fprintf(fptr, "{\"%s\":{", name);

    for (size_t i = 0; i < size; i++)
    {
        fprintf(fptr, "\"%ld\":%f", i, arr[i]);

        if (i != size - 1)
        {
            fprintf(fptr, ",");
        }
    }

    fprintf(fptr, "}}");

    fclose(fptr);
}

void save_list_to_json(double *arr, char *file_name, char *name, size_t size)
{
    FILE *fptr;

    fptr = fopen(file_name, "w");

    fprintf(fptr, "{\"%s\":", name);
    fprintf(fptr, "[");

    for (size_t i = 0; i < size; i++)
    {
        fprintf(fptr, "%f", arr[i]);

        if (i != size - 1)
        {
            fprintf(fptr, ",");
        }
    }

    fprintf(fptr, "]}");

    fclose(fptr);
}

int check_vol_async_present(hid_t fapl_id)
{
    hid_t default_vol_id = H5I_INVALID_HID;
    hid_t native_vol_id = H5I_INVALID_HID;
    int cmp = -1;

    /* Get the ID of the VOL connector set on the default FAPL (H5P_DEFAULT) */
    if (H5Pget_vol_id(fapl_id, &default_vol_id) < 0)
    {
        fprintf(stderr, "error\n");
        return -1;
    }

    /* Get the ID for the native VOL connector by its name */
    if ((native_vol_id = H5VLget_connector_id_by_name("async")) < 0)
    {
        fprintf(stderr, "error\n");
        return -1;
    }

    /* Compare the two VOL connectors */
    if (H5VLcmp_connector_cls(&cmp, default_vol_id, native_vol_id) < 0)
    {
        fprintf(stderr, "error\n");
        return -1;
    }

    if (0 == cmp)
    {
        printf("Using the async VOL connector\n");
        return 0;
    }
    else
    {
        printf("NOT using the async VOL connector\n");
        return -1;
    }

    H5VLclose(default_vol_id);
    H5VLclose(native_vol_id);

    return 0;
}

void create_hdf5(bool with_chunking, hsize_t size, hsize_t chunk)
{
    printf(ANSI_COLOR_YELLOW "Create hdf5 file" ANSI_COLOR_RESET "\n");
    hid_t plist_id, file_id, filespace, dset_id; /* file identifier */
    herr_t status;
    hsize_t dims[1];
    hsize_t cdims[1];

    /* Create a new file using default properties. */
    file_id = H5Fcreate("data/datasets/test_dataset_hdf5-c.h5", H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    // setup dimensions
    hsize_t some_size = size;

    dims[0] = some_size;
    filespace = H5Screate_simple(1, dims, NULL);

    plist_id = H5Pcreate(H5P_DATASET_CREATE);

    if (chunk != 0)
    {
        // setup chunking
        cdims[0] = chunk;
        status = H5Pset_chunk(plist_id, 1, cdims);
    }

    // create Dataset
    dset_id = H5Dcreate(file_id, "/X", H5T_IEEE_F64LE, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);

    // fill buffer
    float *wbuf = calloc(some_size, sizeof(float));

    if (!wbuf)
    {
        fprintf(stderr, "Fatal: unable to allocate array\n");
        exit(EXIT_FAILURE);
    }

    hsize_t i;

    for (i = 0; i < some_size; i++)
    {
        wbuf[i] = (float)rand() / RAND_MAX;
    }

    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, wbuf);

    free(wbuf);

    /* Terminate access to the file. */
    status = H5Dclose(dset_id);
    status = H5Sclose(filespace);
    status = H5Fclose(file_id);
    printf(ANSI_COLOR_YELLOW "Finish creating hdf5 file" ANSI_COLOR_RESET "\n");
}

void create_hdf5_parallel(int argc, char **argv, bool with_chunking, hsize_t size, hsize_t chunk)
{
    printf(ANSI_COLOR_YELLOW "Create hdf5 file with parallel" ANSI_COLOR_RESET "\n");
    hid_t plist_id = 0;
    hid_t file_id = 0;
    hid_t filespace = 0;
    hid_t memspace = 0;
    hid_t dset_id = 0;
    herr_t status = -1;
    hsize_t dims[1];
    hsize_t cdims[1];
    hsize_t count[1]; /* hyperslab selection parameters */
    hsize_t offset[1];
    hsize_t block[1];

    /*
     * Initialize MPI
     */

    int mpi_ssize;
    int mpi_rrank;
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_thread_required = MPI_THREAD_MULTIPLE;
    int mpi_thread_provided = 0;
    /* Initialize MPI with threading support */
    MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);

    MPI_Comm_size(comm, &mpi_ssize);
    MPI_Comm_rank(comm, &mpi_rrank);

    hsize_t mpi_size = mpi_ssize;
    hsize_t mpi_rank = mpi_rrank;

    /*
     * Set up file access property list with parallel I/O access
     */
    plist_id = H5Pcreate(H5P_FILE_ACCESS);
    status = H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

    H5Pset_all_coll_metadata_ops(plist_id, true);
    H5Pset_coll_metadata_write(plist_id, true);

    /*
     * Create a new file collectively.
     */
    file_id = H5Fcreate("data/datasets/test_dataset_hdf5-c_parallel.h5", H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);

    /*
     * Close property list.
     */
    H5Pclose(plist_id);

    // setup dimensions
    hsize_t process_mem_size = size / mpi_size;

    dims[0] = size;
    filespace = H5Screate_simple(1, dims, NULL);

    plist_id = H5Pcreate(H5P_DATASET_CREATE);

    if (chunk != 0)
    {
        // setup chunking
        cdims[0] = chunk;
        status = H5Pset_chunk(plist_id, 1, cdims);
    }

    /*
     * Create the dataset with default properties and close filespace.
     */
    dset_id = H5Dcreate(file_id, "/X", H5T_IEEE_F64LE, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Sclose(filespace);

    /*
     * Close property list.
     */
    H5Pclose(plist_id);

    /*
     * Select hyperslab in the file.
     */
    count[0] = process_mem_size;
    offset[0] = count[0] * mpi_rank;
    memspace = H5Screate_simple(1, count, NULL);

    filespace = H5Dget_space(dset_id);

    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    /*
     * Initialize data buffer
     */
    float *wbuf = calloc(process_mem_size, sizeof(float));

    if (!wbuf)
    {
        fprintf(stderr, "Fatal: unable to allocate array\n");
        exit(EXIT_FAILURE);
    }

    hsize_t i;

    for (i = 0; i < process_mem_size; i++)
    {
        wbuf[i] = (float)rand() / RAND_MAX;
    }

    /*
     * Create property list for collective dataset write.
     */
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    /*
     * To write dataset independently use
     *
     * H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
     */
    status = H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, wbuf);

    /*
     * Close/release resources.
     */

    status = H5Dclose(dset_id);
    status = H5Sclose(filespace);
    status = H5Pclose(plist_id);
    status = H5Fclose(file_id);

    free(wbuf);
    printf(ANSI_COLOR_YELLOW "Finish creating hdf5 file" ANSI_COLOR_RESET "\n");
    MPI_Barrier(MPI_COMM_WORLD);
}

void create_hdf5_async(int argc, char **argv, bool with_chunking, hsize_t size, hsize_t chunk)
{
    printf(ANSI_COLOR_YELLOW "Create hdf5 file with vol-async" ANSI_COLOR_RESET "\n");
    hid_t cplist_id = 0;
    hid_t plist_id = 0;
    hid_t fapl_id = 0;
    hid_t file_id = 0;
    hid_t filespace = 0;
    hid_t memspace = 0;
    hid_t dset_id = 0;
    hid_t es_id = 0;
    herr_t status = -1;
    hsize_t dims[1];
    hsize_t cdims[1];
    hsize_t mem_chunk_dims[1];
    hsize_t count[1]; /* hyperslab selection parameters */
    hsize_t offset[1];
    hsize_t block[1];

    /*
     * Initialize MPI
     */

    int mpi_ssize;
    int mpi_rrank;
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_thread_required = MPI_THREAD_MULTIPLE;
    int mpi_thread_provided = 0;
    /* Initialize MPI with threading support */
    MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);

    MPI_Comm_size(comm, &mpi_ssize);
    MPI_Comm_rank(comm, &mpi_rrank);

    hsize_t mpi_size = mpi_ssize;
    hsize_t mpi_rank = mpi_rrank;

    es_id = H5EScreate();

    /*
     * Set up file access property list with parallel I/O access
     */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    status = H5Pset_vol_async(fapl_id);

    check_vol_async_present(fapl_id);

    status = H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

    H5Pset_all_coll_metadata_ops(fapl_id, true);
    H5Pset_coll_metadata_write(fapl_id, true);

    /*
     * Create a new file collectively.
     */
    file_id = H5Fcreate_async("data/datasets/test_dataset_hdf5-c_async.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id, es_id);

    // setup dimensions
    hsize_t process_mem_size = size / mpi_size;

    dims[0] = size;
    filespace = H5Screate_simple(1, dims, NULL);

    cplist_id = H5Pcreate(H5P_DATASET_CREATE);

    if (chunk != 0)
    {
        // setup chunking
        cdims[0] = chunk;
        status = H5Pset_chunk(cplist_id, 1, cdims);
    }

    /*
     * Create the dataset with default properties and close filespace.
     */
    dset_id = H5Dcreate_async(file_id, "/X", H5T_IEEE_F64LE, filespace, H5P_DEFAULT, cplist_id, H5P_DEFAULT, es_id);

    /*
     * Select hyperslab in the file.
     */
    count[0] = process_mem_size;
    offset[0] = count[0] * mpi_rank;
    memspace = H5Screate_simple(1, count, NULL);

    status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
    /*
     * Initialize data buffer
     */
    float *wbuf = calloc(process_mem_size, sizeof(float));

    if (!wbuf)
    {
        fprintf(stderr, "Fatal: unable to allocate array\n");
        exit(EXIT_FAILURE);
    }

    hsize_t i;

    for (i = 0; i < process_mem_size; i++)
    {
        wbuf[i] = (float)rand() / RAND_MAX;
    }

    /*
     * Create property list for collective dataset write.
     */
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    /*
     * To write dataset independently use
     *
     * H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
     */
    status = H5Dwrite_async(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, wbuf, es_id);

    /*
     * Close/release resources.
     */

    status = H5Dclose_async(dset_id, es_id);
    status = H5Fclose_async(file_id, es_id);

    size_t num_in_progress;
    hbool_t op_failed;
    printf("Wait for async answer \n");
    status = H5ESwait(es_id, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
    printf("Finish waiting for async, num in progess: %ld, failed: %d, status: %d \n", num_in_progress, op_failed, status);

    status = H5ESclose(es_id);

    status = H5Pclose(fapl_id);
    status = H5Sclose(memspace);
    status = H5Sclose(filespace);
    status = H5Pclose(cplist_id);
    status = H5Pclose(plist_id);

    free(wbuf);
    printf(ANSI_COLOR_YELLOW "Finish creating hdf5 file" ANSI_COLOR_RESET "\n");
    MPI_Barrier(MPI_COMM_WORLD);
}

void create_hdf5_subfiling(int argc, char **argv, bool with_chunking, hsize_t size, hsize_t chunk)
{
    printf(ANSI_COLOR_YELLOW "Create hdf5 file with subfiling" ANSI_COLOR_RESET "\n");
    H5FD_subfiling_config_t subf_config;
    H5FD_ioc_config_t ioc_config;
    ;
    hid_t file_id = H5I_INVALID_HID;
    hid_t fapl_id = H5I_INVALID_HID;

    /*
     * Initialize MPI
     */

    int mpi_ssize;
    int mpi_rrank;
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_thread_required = MPI_THREAD_MULTIPLE;
    int mpi_thread_provided = 0;
    /* Initialize MPI with threading support */
    MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);

    MPI_Comm_size(comm, &mpi_ssize);
    MPI_Comm_rank(comm, &mpi_rrank);

    hsize_t mpi_size = mpi_ssize;
    hsize_t mpi_rank = mpi_rrank;

    if (mpi_thread_provided != mpi_thread_required)
        MPI_Abort(comm, -1);
    /*
     * Set up File Access Property List with MPI
     * parameters for the Subfiling VFD to use
     */
    fapl_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_mpi_params(fapl_id, comm, MPI_INFO_NULL);

    // putenv("H5FD_SUBFILING_IOC_PER_NODE = 2");
    /*
     * Get a default Subfiling and IOC configuration
     */
    H5Pget_fapl_subfiling(fapl_id, &subf_config);
    H5Pget_fapl_ioc(fapl_id, &ioc_config);
    /*
     * Set Subfiling configuration to use a 1MiB
     * stripe size.
     */
    // subf_config.shared_cfg.stripe_size = 1048576;
    // subf_config.shared_cfg.ioc_selection = SELECT_IOC_ONE_PER_NODE;

    subf_config.shared_cfg.stripe_count = 4;
    /*
     * Set IOC configuration to use 2 worker threads
     * per IOC instead of the default setting.
     */
    // ioc_config.thread_pool_size = 2;

    /*
     * Set our new configuration on the IOC
     * FAPL used for Subfiling
     */
    H5Pset_fapl_ioc(subf_config.ioc_fapl_id, &ioc_config);
    /*
     * Finally, set our new Subfiling configuration
     * on the original FAPL
     */
    H5Pset_fapl_subfiling(fapl_id, &subf_config);

    H5Pset_alignment(fapl_id, 0, 33554432);

    H5Pset_all_coll_metadata_ops(fapl_id, true);
    H5Pset_coll_metadata_write(fapl_id, true);
    /*
     * Create a new Subfiling-based HDF5 file
     */
    file_id = H5Fcreate("data/datasets/test_dataset_subfiling.h5", H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);
    H5Pclose(fapl_id);

    hid_t plist_id = 0;
    hid_t dset_id = 0;
    hid_t filespace = 0;
    hid_t memspace = 0;
    hsize_t dims[1];
    hsize_t cdims[1];
    hsize_t count[1]; /* hyperslab selection parameters */
    hsize_t offset[1];
    hsize_t block[1];

    // setup dimensions
    hsize_t process_mem_size = size / mpi_size;

    dims[0] = size;
    filespace = H5Screate_simple(1, dims, NULL);

    plist_id = H5Pcreate(H5P_DATASET_CREATE);

    if (chunk != 0)
    {
        // setup chunking
        cdims[0] = chunk;
        H5Pset_chunk(plist_id, 1, cdims);
    }

    // create Dataset
    dset_id = H5Dcreate2(file_id, "/X", H5T_IEEE_F64LE, filespace, H5P_DEFAULT, plist_id, H5P_DEFAULT);
    H5Sclose(filespace);
    H5Pclose(plist_id);

    /*
     * Select hyperslab in the file.
     */
    count[0] = process_mem_size;
    offset[0] = count[0] * mpi_rank;
    memspace = H5Screate_simple(1, count, NULL);

    filespace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    // fill buffer
    float *wbuf = calloc(process_mem_size, sizeof(float));

    if (!wbuf)
    {
        fprintf(stderr, "Fatal: unable to allocate array\n");
        exit(EXIT_FAILURE);
    }

    hsize_t i;

    for (i = 0; i < process_mem_size; i++)
    {
        wbuf[i] = (float)rand() / RAND_MAX;
    }

    /*
     * Create property list for independent dataset write.
     */
    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

    H5Dwrite(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, wbuf);

    /* Close/free resources */

    H5Dclose(dset_id);
    H5Sclose(filespace);
    H5Pclose(plist_id);
    H5Fclose(file_id);

    free(wbuf);
    printf(ANSI_COLOR_YELLOW "Finish creating hdf5 file" ANSI_COLOR_RESET "\n");
    MPI_Barrier(MPI_COMM_WORLD);
}

void bench_variable_hdf5(hsize_t size, int iteration)
{
    double arr3[iteration];

    // create_hdf5(false, size);

    for (int i = 0; i < iteration; i++)
    {
        printf("i: %d for variable: X\n", i);

        hid_t file_id, dset_id;
        herr_t status;
        hsize_t some_size = size;
        float *rbuf = calloc(some_size, sizeof(float));

        file_id = H5Fopen("data/datasets/test_dataset_hdf5-c.h5", H5F_ACC_RDONLY, H5P_DEFAULT);
        dset_id = H5Dopen2(file_id, "/X", H5P_DEFAULT);

        struct timespec start, end;

        clock_gettime(CLOCK_MONOTONIC, &start);

        status = H5Dread(dset_id, H5T_NATIVE_FLOAT, H5S_BLOCK, H5S_ALL, H5P_DEFAULT, rbuf);

        status = H5Dclose(dset_id);
        status = H5Fclose(file_id);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = end.tv_sec - start.tv_sec;
        elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        arr3[i] = elapsed;

        free(rbuf);
        printf(ANSI_COLOR_GREEN "Finished" ANSI_COLOR_RESET "\n");
    }

    save_to_json(arr3, "data/results/test_hdf5-c.json", "hdf5-c-read", iteration);
}

void bench_variable_hdf5_parallel(int argc, char **argv, hsize_t size, hsize_t chunk, int iteration)
{
    double arr3[iteration];

    // create_hdf5_parallel(argc, argv, false, size);

    /*
     * Initialize MPI
     */

    int mpi_ssize;
    int mpi_rrank;
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_thread_required = MPI_THREAD_MULTIPLE;
    int mpi_thread_provided = 0;
    /* Initialize MPI with threading support */
    MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);

    MPI_Comm_size(comm, &mpi_ssize);
    MPI_Comm_rank(comm, &mpi_rrank);

    hsize_t mpi_size = mpi_ssize;
    hsize_t mpi_rank = mpi_rrank;

    for (int i = 0; i < iteration; i++)
    {
        printf("i: %d for variable: X rank: %ld\n", i, mpi_rank);

        hid_t plist_id = 0;
        hid_t file_id = 0;
        hid_t dset_id = 0;
        herr_t status = -1;
        hid_t filespace = 0;
        hid_t memspace = 0;
        hsize_t count[1]; /* hyperslab selection parameters */
        hsize_t offset[1];
        hsize_t block[1];

        /*
         * Set up file access property list with parallel I/O access
         */
        plist_id = H5Pcreate(H5P_FILE_ACCESS);
        status = H5Pset_fapl_mpio(plist_id, comm, MPI_INFO_NULL);

        /*
         * Open existing file collectively and release property list identifier.
         */
        file_id = H5Fopen("data/datasets/test_dataset_hdf5-c_parallel.h5", H5F_ACC_RDONLY, plist_id);
        H5Pclose(plist_id);

        struct timespec start, end;
        if (mpi_rank == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &start);
        }

        /*
         * open the dataset with default properties and close filespace.
         */
        dset_id = H5Dopen(file_id, "/X", H5P_DEFAULT);

        /*
         * Select hyperslab in the file.
         */

        hsize_t process_mem_size = size / mpi_size;

        count[0] = process_mem_size;
        offset[0] = count[0] * mpi_rank;
        memspace = H5Screate_simple(1, count, NULL);

        filespace = H5Dget_space(dset_id);
        status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
        /*
         * Initialize data buffer
         */
        float *rbuf = calloc(process_mem_size, sizeof(float));

        /*
         * Create property list for collective dataset reads.
         */
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

        /*
         * To read dataset independently use
         *
         * H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
         */

        status = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, rbuf);

        /*
         * Close/release resources.
         */
        status = H5Dclose(dset_id);
        status = H5Pclose(plist_id);
        status = H5Fclose(file_id);

        if (mpi_rank == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &end);
            double elapsed = end.tv_sec - start.tv_sec;
            elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
            arr3[i] = elapsed;
        }

        free(rbuf);
        MPI_Barrier(MPI_COMM_WORLD);
        printf(ANSI_COLOR_GREEN "Finished" ANSI_COLOR_RESET "\n");
    }

    if (mpi_rank == 0)
    {
        save_to_json(arr3, "data/results/test_hdf5-c_parallel.json", "hdf5-c-read-parallel", iteration);
    }
}

void bench_variable_netcdf4(hsize_t size, int iteration)
{
    double arr3[iteration];

    for (int i = 0; i < iteration; i++)
    {
        printf("i: %d for variable: X\n", i);
        int ncid, varid, retval;
        hsize_t some_size = size;
        float *rbuf = calloc(some_size, sizeof(float));

        if ((retval = nc_open("../data/datasets/test_dataset.nc", NC_NOWRITE, &ncid)))
        {
            printf("failure to open file");
            ERR(retval);
        }

        struct timespec start, end;

        clock_gettime(CLOCK_MONOTONIC, &start);

        if ((retval = nc_inq_varid(ncid, "X", &varid)))
        {
            printf("failure finding variable id");
            ERR(retval);
        }

        if ((retval = nc_get_var_float(ncid, varid, rbuf)))
        {
            printf("failure reading variable");
            ERR(retval);
        }

        if ((retval = nc_close(ncid)))
            ERR(retval);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = end.tv_sec - start.tv_sec;
        elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        arr3[i] = elapsed;

        free(rbuf);
        printf(ANSI_COLOR_GREEN "Finished" ANSI_COLOR_RESET "\n");
    }

    save_to_json(arr3, "data/results/test_netcdf4.json", "netcdf4-read", iteration);
}

void bench_variable_netcdf4_parallel(int argc, char **argv, hsize_t size, int iteration)
{
    double arr3[iteration];

    /*
     * Initialize MPI
     */

    int mpi_ssize;
    int mpi_rrank;
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_thread_required = MPI_THREAD_MULTIPLE;
    int mpi_thread_provided = 0;
    /* Initialize MPI with threading support */
    MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);

    MPI_Comm_size(comm, &mpi_ssize);
    MPI_Comm_rank(comm, &mpi_rrank);

    hsize_t mpi_size = mpi_ssize;
    hsize_t mpi_rank = mpi_rrank;

    for (int i = 0; i < iteration; i++)
    {
        printf("i: %d for variable: X rank: %ld\n", i, mpi_rank);
        int ncid, varid, retval;
        hsize_t count[1]; /* hyperslab selection parameters */
        hsize_t offset[1];

        if ((retval = nc_open_par("../data/datasets/test_dataset.nc", NC_NOWRITE, comm, MPI_INFO_NULL, &ncid)))
        {
            printf("failure to open file ");
            ERR(retval);
        }

        struct timespec start, end;

        if (mpi_rank == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &start);
        }

        if ((retval = nc_inq_varid(ncid, "X", &varid)))
        {
            printf("failure finding variable id ");
            ERR(retval);
        }

        if ((retval = nc_var_par_access(ncid, varid, NC_COLLECTIVE)))
        {
            printf("failure setting collective I/O ");
            ERR(retval);
        }

        hsize_t process_mem_size = size / mpi_size;

        count[0] = process_mem_size;
        offset[0] = count[0] * mpi_rank;

        float *rbuf = calloc(process_mem_size, sizeof(float));

        if ((retval = nc_get_vara_float(ncid, varid, offset, count, rbuf)))
        {
            printf("failure reading variable");
            ERR(retval);
        }

        if ((retval = nc_close(ncid)))
            ERR(retval);

        if (mpi_rank == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &end);
            double elapsed = end.tv_sec - start.tv_sec;
            elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
            arr3[i] = elapsed;
        }

        free(rbuf);
        MPI_Barrier(MPI_COMM_WORLD);
        printf(ANSI_COLOR_GREEN "Finished" ANSI_COLOR_RESET "\n");
    }

    if (mpi_rank == 0)
    {
        save_to_json(arr3, "data/results/test_netcdf4-c_parallel.json", "netcdf4-c-read-parallel", iteration);
    }
}

void bench_variable_nczarr(hsize_t size, int iteration)
{
    double arr3[iteration];

    for (int i = 0; i < iteration; i++)
    {

        printf("i: %d for variable: X\n", i);
        int ncid, varid, retval;
        hsize_t some_size = size;
        float *rbuf = calloc(some_size, sizeof(float));

        if ((retval = nc_open("file:///home/dev/dkrz_dev/c-stuff/data/datasets/test_dataset.zarr#mode=zarr,file", NC_NOWRITE, &ncid)))
        {
            printf("failure to open file ");
            ERR(retval);
        }


        struct timespec start, end;

        clock_gettime(CLOCK_MONOTONIC, &start);

        if ((retval = nc_inq_varid(ncid, "X", &varid)))
        {
            printf("failure finding variable id ");
            ERR(retval);
        }

        if ((retval = nc_get_var_float(ncid, varid, rbuf)))
        {
            printf("failure reading variable ");
            ERR(retval);
        }

        if ((retval = nc_close(ncid)))
            ERR(retval);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double elapsed = end.tv_sec - start.tv_sec;
        elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
        arr3[i] = elapsed;

        free(rbuf);
        printf(ANSI_COLOR_GREEN "Finished" ANSI_COLOR_RESET "\n");
    }

    save_to_json(arr3, "data/results/test_nczarr.json", "nczarr-read", iteration);
}

void bench_variable_async(int argc, char **argv, hsize_t size, hsize_t chunk, int iteration)
{
    double arr3[iteration];

    // create_hdf5_async(argc, argv, false, size);

    /*
     * Initialize MPI
     */

    int mpi_ssize;
    int mpi_rrank;
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_thread_required = MPI_THREAD_MULTIPLE;
    int mpi_thread_provided = 0;
    /* Initialize MPI with threading support */
    MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);

    MPI_Comm_size(comm, &mpi_ssize);
    MPI_Comm_rank(comm, &mpi_rrank);

    hsize_t mpi_size = mpi_ssize;
    hsize_t mpi_rank = mpi_rrank;

    for (int i = 0; i < iteration; i++)
    {
        printf("i: %d for variable: X rank: %ld\n", i, mpi_rank);

        hid_t fapl_id = 0;
        hid_t dxpl_id = 0;
        hid_t es_id = 0;
        hid_t file_id = 0;
        hid_t dset_id = 0;
        herr_t status = -1;
        hid_t filespace = 0;
        hid_t memspace = 0;
        hsize_t count[1]; /* hyperslab selection parameters */
        hsize_t offset[1];
        hsize_t block[1];

        es_id = H5EScreate();

        /*
         * Set up file access property list with parallel I/O access
         */
        fapl_id = H5Pcreate(H5P_FILE_ACCESS);

        status = H5Pset_vol_async(fapl_id);

        check_vol_async_present(fapl_id);

        status = H5Pset_fapl_mpio(fapl_id, comm, MPI_INFO_NULL);

        /*
         * Open existing file collectively and release property list identifier.
         */
        file_id = H5Fopen_async("data/datasets/test_dataset_hdf5-c_async.h5", H5F_ACC_RDONLY, fapl_id, es_id);

        struct timespec start, end;

        if (mpi_rank == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &start);
        }

        /*
         * open the dataset with default properties and close filespace.
         */
        dset_id = H5Dopen_async(file_id, "/X", H5P_DEFAULT, es_id);

        /*
         * Select hyperslab in the file.
         */

        hsize_t process_mem_size = size / mpi_size;

        count[0] = process_mem_size;
        offset[0] = count[0] * mpi_rank;
        memspace = H5Screate_simple(1, count, NULL);

        filespace = H5Dget_space(dset_id);
        status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
        /*
         * Initialize data buffer
         */
        float *rbuf = calloc(process_mem_size, sizeof(float));

        /*
         * Create property list for collective dataset reads.
         */
        dxpl_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(dxpl_id, H5FD_MPIO_COLLECTIVE);

        /*
         * To read dataset independently use
         *
         * H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);
         */
        status = H5Dread_async(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, dxpl_id, rbuf, es_id);
        // status = H5Dread_async(dset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, dxpl_id, rbuf, es_id);

        /*
         * Close/release resources.
         */
        status = H5Dclose_async(dset_id, es_id);
        status = H5Fclose_async(file_id, es_id);

        size_t num_in_progress = 0;
        hbool_t op_failed = 0;

        printf("Wait for async answer \n");
        status = H5ESwait(es_id, H5ES_WAIT_FOREVER, &num_in_progress, &op_failed);
        printf("Finish waiting for async, num in progess: %ld, failed: %d, status: %d \n", num_in_progress, op_failed, status);

        H5ESclose(es_id);

        status = H5Pclose(fapl_id);
        status = H5Pclose(dxpl_id);

        if (mpi_rank == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &end);
            double elapsed = end.tv_sec - start.tv_sec;
            elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
            arr3[i] = elapsed;
        }

        free(rbuf);
        MPI_Barrier(MPI_COMM_WORLD);
        printf(ANSI_COLOR_GREEN "Finished" ANSI_COLOR_RESET "\n");
    }

    if (mpi_rank == 0)
    {
        save_to_json(arr3, "data/results/test_hdf5-c_async.json", "hdf5-c-async-read", iteration);
    }
}

void bench_variable_subfiling(int argc, char **argv, hsize_t size, hsize_t chunk, int iteration)
{
    double arr3[iteration];

    // create_hdf5_subfiling(argc, argv, false, size);

    /*
     * Initialize MPI
     */

    int mpi_ssize;
    int mpi_rrank;
    MPI_Comm comm = MPI_COMM_WORLD;

    int mpi_thread_required = MPI_THREAD_MULTIPLE;
    int mpi_thread_provided = 0;
    /* Initialize MPI with threading support */
    MPI_Init_thread(&argc, &argv, mpi_thread_required, &mpi_thread_provided);

    MPI_Comm_size(comm, &mpi_ssize);
    MPI_Comm_rank(comm, &mpi_rrank);

    hsize_t mpi_size = mpi_ssize;
    hsize_t mpi_rank = mpi_rrank;

    for (int i = 0; i < iteration; i++)
    {
        printf("i: %d for variable: X rank: %ld\n", i, mpi_rank);

        H5FD_subfiling_config_t subf_config;
        H5FD_ioc_config_t ioc_config;
        hid_t fapl_id = H5I_INVALID_HID;

        fapl_id = H5Pcreate(H5P_FILE_ACCESS);
        H5Pset_mpi_params(fapl_id, comm, MPI_INFO_NULL);

        /*
         * Get a default Subfiling and IOC configuration
         */
        H5Pget_fapl_subfiling(fapl_id, &subf_config);
        H5Pget_fapl_ioc(fapl_id, &ioc_config);
        /*
         * Set Subfiling configuration to use a 1MiB
         * stripe size.
         */
        // subf_config.shared_cfg.stripe_size = 1048576;
        // subf_config.shared_cfg.ioc_selection = SELECT_IOC_ONE_PER_NODE;

        subf_config.shared_cfg.stripe_count = 4;
        /*
         * Set IOC configuration to use 2 worker threads
         * per IOC instead of the default setting.
         */
        // ioc_config.thread_pool_size = 2;

        /*
         * Set our new configuration on the IOC
         * FAPL used for Subfiling
         */
        H5Pset_fapl_ioc(subf_config.ioc_fapl_id, &ioc_config);
        /*
         * Finally, set our new Subfiling configuration
         * on the original FAPL
         */
        H5Pset_fapl_subfiling(fapl_id, &subf_config);

        H5Pset_alignment(fapl_id, 0, 33554432);

        H5Pset_all_coll_metadata_ops(fapl_id, true);
        H5Pset_coll_metadata_write(fapl_id, true);

        hid_t plist_id = 0;
        hid_t file_id = 0;
        hid_t dset_id = 0;
        herr_t status = -1;
        hid_t filespace = 0;
        hid_t memspace = 0;
        hsize_t count[1]; /* hyperslab selection parameters */
        hsize_t offset[1];
        hsize_t block[1];

        /*
         * Open existing file collectively and release property list identifier.
         */
        file_id = H5Fopen("data/datasets/test_dataset_subfiling.h5", H5F_ACC_RDONLY, fapl_id);
        H5Pclose(fapl_id);

        struct timespec start, end;

        if (mpi_rank == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &start);
        }

        dset_id = H5Dopen(file_id, "/X", H5P_DEFAULT);

        /*
         * Select hyperslab in the file.
         */

        hsize_t process_mem_size = size / mpi_size;

        count[0] = process_mem_size;
        offset[0] = count[0] * mpi_rank;
        memspace = H5Screate_simple(1, count, NULL);

        filespace = H5Dget_space(dset_id);
            status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);
        
        /*
         * Initialize data buffer
         */
        float *rbuf = calloc(process_mem_size, sizeof(float));

        /*
         * Create property list for independent dataset reads.
         */
        plist_id = H5Pcreate(H5P_DATASET_XFER);
        H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_INDEPENDENT);

        status = H5Dread(dset_id, H5T_NATIVE_FLOAT, memspace, filespace, plist_id, rbuf);

        /*
         * Close/release resources.
         */
        status = H5Dclose(dset_id);
        status = H5Pclose(plist_id);
        status = H5Fclose(file_id);

        if (mpi_rank == 0)
        {
            clock_gettime(CLOCK_MONOTONIC, &end);
            double elapsed = end.tv_sec - start.tv_sec;
            elapsed += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
            arr3[i] = elapsed;
        }

        free(rbuf);

        MPI_Barrier(MPI_COMM_WORLD);
        printf(ANSI_COLOR_GREEN "Finished" ANSI_COLOR_RESET "\n");
    }
    if (mpi_rank == 0)
    {
        save_to_json(arr3, "data/results/test_hdf5_subfiling.json", "hdf5-subfiling-read", iteration);
    }
}

typedef struct args_t
{
    int create;
    int benchmark;
    hsize_t size;
    hsize_t chunk;
    hsize_t factor;
    int iterations;
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
    case 'i':
        arguments->iterations = atoi(arg);
        break;
    case 's':
        arguments->size = strtoull(arg, NULL, 10);
        break;
    case 'k':
        arguments->chunk = strtoull(arg, NULL, 10);
        break;
    case 'f':
        arguments->factor = strtoull(arg, NULL, 10);
        break;
    case ARGP_KEY_ARG:
        return 0;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp_option options[] = {
    {"create file", 'c', "NUM", 0, "File to create from selection of 1-6, matches benchmarks"},
    {"benchmark", 'b', "NUM", 0, "Benchmark to run from a selection of 1-6"},
    {"base-filesize", 's', "NUM", 0, "Specifiy the base-filesize of the file you want to create"},
    {"chunksize", 'k', "NUM", 0, "Specifiy the base-filesize of the file you want to create"},
    {"factor", 'f', "NUM", 0, "Factor to multiply base-filesize with to increase / decrease size"},
    {"iterations", 'i', "NUM", 0, "Ammount of iterations the benchmark should run"},
    {0}};

int main(int argc, char **argv)
{

    struct argp argp = {options, parse_opt};

    args_t arguments;
    arguments.create = -1;
    arguments.benchmark = -1;
    arguments.size = 134217728;
    arguments.chunk = 0;
    arguments.factor = 1;
    arguments.iterations = 10;

    printf(ANSI_COLOR_CYAN "Default benchmark %d, filesize %lu, chunksize %lu, factor %lu, iterations %d" ANSI_COLOR_RESET "\n", arguments.benchmark, arguments.size, arguments.chunk, arguments.factor, arguments.iterations);

    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    printf(ANSI_COLOR_MAGENTA "Parsing %d, filesize %lu, chunksize %lu, factor %lu, iterations %d" ANSI_COLOR_RESET "\n", arguments.benchmark, arguments.size, arguments.chunk, arguments.factor, arguments.iterations);

    hsize_t size = arguments.size * arguments.factor;
    hsize_t chunk = arguments.chunk;
    int iterations = arguments.iterations;

    // arguments parsing for creation of file
    switch (arguments.create)
    {
    case -1:
        break;
    case 1:
        printf(ANSI_COLOR_BLUE "Creating hdf5 file with a filesize of %lu and chunksize of %lu" ANSI_COLOR_RESET "\n", size, chunk);
        create_hdf5(false, size, chunk);
        break;
    case 2:
        printf(ANSI_COLOR_BLUE "Creating netcdf4 file with a filesize of %lu and chunksize of %lu" ANSI_COLOR_RESET "\n", size, chunk);
        break;
    case 3:
        printf(ANSI_COLOR_BLUE "Creating nczarr file with a filesize of %lu and chunksize of %lu" ANSI_COLOR_RESET "\n", size, chunk);
        break;
    case 4:
        printf(ANSI_COLOR_BLUE "Creating hdf5-parallel file with a filesize of %lu and chunksize of %lu" ANSI_COLOR_RESET "\n", size, chunk);
        create_hdf5_parallel(argc, argv, false, size, chunk);
        MPI_Finalize();
        break;
    case 5:
        printf(ANSI_COLOR_BLUE "Creating hdf5-vol-async file with a filesize of %lu and chunksize of %lu" ANSI_COLOR_RESET "\n", size, chunk);
        create_hdf5_async(argc, argv, false, size, chunk);
        MPI_Finalize();
        break;
    case 6:
        printf(ANSI_COLOR_BLUE "Creating hdf5 subfiling file with a filesize of %lu and chunksize of %lu" ANSI_COLOR_RESET "\n", size, chunk);
        create_hdf5_subfiling(argc, argv, false, size, chunk);
        MPI_Finalize();
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
        printf("No benchmark specified, exiting programm now \n");
        break;
    case 1:
        printf(ANSI_COLOR_BLUE "Running hdf5 benchmark with a filesize of %lu for %d iterations" ANSI_COLOR_RESET "\n", size, iterations);
        bench_variable_hdf5(size, iterations);
        break;
    case 2:
        printf(ANSI_COLOR_BLUE "Running netcdf4 benchmark with a filesize of %lu for %d iterations" ANSI_COLOR_RESET "\n", size, iterations);
        bench_variable_netcdf4(size, iterations);
        break;
    case 3:
        printf(ANSI_COLOR_BLUE "Running nczarr benchmark with a filesize of %lu for %d iterations" ANSI_COLOR_RESET "\n", size, iterations);
        bench_variable_nczarr(size, iterations);
        break;
    case 4:
        printf(ANSI_COLOR_RED "CURRENTLY ONLY WORKS WITH EVEN MPI_RANKS" ANSI_COLOR_RESET "\n");
        printf(ANSI_COLOR_BLUE "Running hdf5 parallel benchmark with a filesize of %lu for %d iterations" ANSI_COLOR_RESET "\n", size, iterations);
        bench_variable_hdf5_parallel(argc, argv, size, chunk, iterations);
        MPI_Finalize();
        break;
    case 5:
        printf(ANSI_COLOR_RED "CURRENTLY ONLY WORKS WITH EVEN MPI_RANKS" ANSI_COLOR_RESET "\n");
        printf(ANSI_COLOR_RED "PLEASE ENSURE YOU LOADED HDF5-VOL-ASYNC AND SET THE ENVIRONMENTAL VARIABLES REQUIRED CORRECTLY" ANSI_COLOR_RESET "\n");
        printf(ANSI_COLOR_BLUE "Running hdf5-vol-async parallel benchmark with a filesize of %lu for %d iterations" ANSI_COLOR_RESET "\n", size, iterations);
        bench_variable_async(argc, argv, size, chunk, iterations);
        MPI_Finalize();
        break;
    case 6:
        printf(ANSI_COLOR_RED "CURRENTLY ONLY WORKS WITH EVEN MPI_RANKS" ANSI_COLOR_RESET "\n");
        printf(ANSI_COLOR_RED "PLEASE ENSURE THIS SCRIPT IS RUN WITH MPIEXEC / MPIRUN" ANSI_COLOR_RESET "\n");
        printf(ANSI_COLOR_BLUE "Running hdf5 subfiling benchmark with a filesize of %lu for %d iterations" ANSI_COLOR_RESET "\n", size, iterations);

        bench_variable_subfiling(argc, argv, size, chunk, iterations);
        MPI_Finalize();
        break;
    case 7:
        printf(ANSI_COLOR_RED "CURRENTLY ONLY WORKS WITH EVEN MPI_RANKS" ANSI_COLOR_RESET "\n");
        printf(ANSI_COLOR_BLUE "Running netcdf4 parallel benchmark with a filesize of %lu for %d iterations" ANSI_COLOR_RESET "\n", size, iterations);
        bench_variable_netcdf4_parallel(argc, argv, size, iterations);
        MPI_Finalize();
        break;
    case ARGP_KEY_ARG:
        return 0;
    default:
        return ARGP_ERR_UNKNOWN;
    }

    return 0;
}
