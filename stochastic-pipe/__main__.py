"""
This file is part of the openPMD-api.

This module provides functions that are wrapped into sys.exit(...()) calls by
the setuptools (setup.py) "entry_points" -> "console_scripts" generator.

Copyright 2021 openPMD contributors
Authors: Franz Poeschel
License: LGPLv3+
"""
import argparse
import os  # os.path.basename
import re
import sys  # sys.stderr.write
import time

import numpy as np
import openpmd_api as io


class DumpTimes:

    def __init__(self, filename):
        self.last_time_point = int(time.time() * 1000)
        self.out_stream = open(filename, 'w')

    def close(self):
        self.out_stream.close()

    def now(self, description, separator='\t'):
        current = int(time.time() * 1000)
        self.out_stream.write(
            str(current) + separator + str(current - self.last_time_point) +
            separator + description + '\n')
        self.last_time_point = current

    def flush(self):
        self.out_stream.flush()


def parse_args(program_name):
    parser = argparse.ArgumentParser(
        # we need this for line breaks
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
openPMD Pipe.

This tool connects an openPMD-based data source with an openPMD-based data sink
and forwards all data from source to sink.
Possible uses include conversion of data from one backend to another one
or multiplexing the data path in streaming setups.
Parallelization with MPI is optionally possible and is done automatically
as soon as the mpi4py package is found and this tool is called in an MPI
context.
Parallelization with MPI is optionally possible and can be switched on with
the --mpi switch, resp. switched off with the --no-mpi switch.
By default, openpmd-pipe will use MPI if all of the following conditions
are fulfilled:
1) The mpi4py package can be imported.
2) The openPMD-api has been built with support for MPI.
3) The MPI size is greater than 1.
   By default, the openPMD-api will be initialized without an MPI communicator
   if the MPI size is 1. This is to simplify the use of the JSON backend
   which is only available in serial openPMD.
With parallelization enabled, each dataset will be equally sliced according to
a chunk distribution strategy which may be selected via the environment variable
OPENPMD_CHUNK_DISTRIBUTION. Options include "roundrobin", "binpacking",
"slicedataset" and "hostname_<1>_<2>", where <1> should be replaced with a
strategy to be applied within a compute node and <2> with a secondary strategy
in case the hostname strategy does not distribute all chunks.
The default is `hostname_binpacking_slicedataset`.

Examples:
    {0} --infile simData.h5 --outfile simData_%T.bp
    {0} --infile simData.sst --inconfig @streamConfig.json \\
        --outfile simData_%T.bp
    {0} --infile uncompressed.bp \\
        --outfile compressed.bp --outconfig @compressionConfig.json
""".format(os.path.basename(program_name)))

    parser.add_argument('--infile', type=str, help='In file')
    parser.add_argument('--outfile', type=str, help='Out file')
    parser.add_argument('--inconfig',
                        type=str,
                        default='{}',
                        help='JSON config for the in file')
    parser.add_argument('--outconfig',
                        type=str,
                        default='{}',
                        help='JSON config for the out file')
    # MPI, default: Import mpi4py if available and openPMD is parallel,
    # but don't use if MPI size is 1 (this makes it easier to interact with
    # JSON, since that backend is unavailable in parallel)
    if io.variants['mpi']:
        parser.add_argument('--mpi', action='store_true')
        parser.add_argument('--no-mpi', dest='mpi', action='store_false')
        parser.set_defaults(mpi=None)

    return parser.parse_args()


args = parse_args(sys.argv[0])
# MPI is an optional dependency
if io.variants['mpi'] and (args.mpi is None or args.mpi):
    try:
        from mpi4py import MPI
        HAVE_MPI = True
    except (ImportError, ModuleNotFoundError):
        if args.mpi:
            raise
        else:
            print("""
    openPMD-api was built with support for MPI,
    but mpi4py Python package was not found.
    Will continue in serial mode.""",
                  file=sys.stderr)
            HAVE_MPI = False
else:
    HAVE_MPI = False

debug = True


class FallbackMPICommunicator:

    def __init__(self):
        self.size = 1
        self.rank = 0


class loaded_chunk:

    def __init__(self, dest_component: io.Record_Component, offset, extent,
                 chunk):
        self.dest_component = dest_component
        self.offset = offset
        self.extent = extent
        self.chunk = chunk


class loaded_chunks_record_component:

    def __init__(self):
        self.chunks = []

    def append(self, entry: loaded_chunk):
        self.chunks.append(entry)

    def sample(self, sample_size_total, my_out_chunk, random_sample, chunks):
        # Assert that all chunks are congruent across components
        assert (len(chunks) == len(self.chunks))
        for i in range(len(chunks)):
            assert chunks[i].offset == self.chunks[i].offset
            assert chunks[i].extent == self.chunks[i].extent

        reset_dataset = True
        offset = my_out_chunk.offset[0]

        for chunk in self.chunks:
            if reset_dataset:
                chunk.dest_component.reset_dataset(
                    io.Dataset(chunk.dest_component.dtype,
                               [sample_size_total]))
                reset_dataset = False
            chunk_len = chunk.extent[0]
            filter_now = random_sample < chunk_len
            filter_next = random_sample >= chunk_len
            filtered = chunk.chunk[random_sample[filter_now]]
            chunk.dest_component.store_chunk(filtered, [offset],
                                             [len(filtered)])

            random_sample = random_sample[filter_next] - chunk_len
            offset += len(filtered)


class loaded_chunks_record:

    def __init__(self):
        self.components = {}

    # e.g. key = "x", "y", "z"
    def insert_component(self, key: str,
                         loaded_chunks: loaded_chunks_record_component):
        self.components[key] = loaded_chunks

    def sample(self, sample_size_total, my_out_chunk, random_sample, chunks):
        for _, component in self.components.items():
            component.sample(sample_size_total, my_out_chunk, random_sample,
                             chunks)


class loaded_chunks_species:

    def __init__(self):
        self.records = {}
        self.chunks = None
        self.shape = None

    # e.g. key = "position", "positionOffset", ...
    def insert_record(self, key: str, record: loaded_chunks_record):
        self.records[key] = record

    def sample(self, communicator, percentage):
        total_size_this_rank = 0
        for chunk in self.chunks:
            assert (len(chunk.offset) == 1 and len(chunk.extent) == 1)
            total_size_this_rank += chunk.extent[0]

        assert (len(self.shape) == 1)
        sample_size_per_rank = int(percentage * self.shape[0] /
                                   communicator.size)
        sample_size_total = communicator.size * sample_size_per_rank

        my_out_chunk = io.ChunkInfo([communicator.rank * sample_size_per_rank],
                                    [sample_size_per_rank])
        assert(total_size_this_rank >= sample_size_per_rank)
        random_sample = np.random.choice(range(total_size_this_rank),
                                         sample_size_per_rank)

        for _, record in self.records.items():
            record.sample(sample_size_total, my_out_chunk, random_sample,
                          self.chunks)


class loaded_chunks_iteration:

    def __init__(self):
        self.particles = {}

    def insert_particle_species(self, key: str,
                                particle_species: loaded_chunks_species):
        self.particles[key] = particle_species

    def sample(self, communicator, percentage=0.05):
        for _, species in self.particles.items():
            species.sample(communicator, percentage)


# class particle_patch_load:
#     """
#     A deferred load/store operation for a particle patch.
#     Our particle-patch API requires that users pass a concrete value for
#     storing, even if the actual write operation occurs much later at
#     series.flush().
#     So, unlike other record components, we cannot call .store_chunk() with
#     a buffer that has not yet been filled, but must wait until the point where
#     we actual have the data at hand already.
#     In short: calling .store() must be deferred, until the data has been fully
#     read from the sink.
#     This class stores the needed parameters to .store().
#     """

#     def __init__(self, data, dest):
#         self.data = data
#         self.dest = dest

#     def run(self):
#         for index, item in enumerate(self.data):
#             self.dest.store(index, item)


def distribution_strategy(dataset_extent,
                          mpi_rank,
                          mpi_size,
                          strategy_identifier=None):
    if strategy_identifier is None or not strategy_identifier:
        if 'OPENPMD_CHUNK_DISTRIBUTION' in os.environ:
            strategy_identifier = os.environ[
                'OPENPMD_CHUNK_DISTRIBUTION'].lower()
        else:
            strategy_identifier = 'hostname_binpacking_slicedataset'  # default
    match = re.search('hostname_(.*)_(.*)', strategy_identifier)
    if match is not None:
        inside_node = distribution_strategy(dataset_extent,
                                            mpi_rank,
                                            mpi_size,
                                            strategy_identifier=match.group(1))
        second_phase = distribution_strategy(
            dataset_extent,
            mpi_rank,
            mpi_size,
            strategy_identifier=match.group(2))
        return io.FromPartialStrategy(io.ByHostname(inside_node), second_phase)
    elif strategy_identifier == 'roundrobin':
        return io.RoundRobin()
    elif strategy_identifier == 'binpacking':
        return io.BinPacking()
    elif strategy_identifier == 'slicedataset':
        return io.ByCuboidSlice(io.OneDimensionalBlockSlicer(), dataset_extent,
                                mpi_rank, mpi_size)
    elif strategy_identifier == 'fail':
        return io.FailingStrategy()
    else:
        raise RuntimeError("Unknown distribution strategy: " +
                           strategy_identifier)


class pipe:
    """
    Represents the configuration of one "pipe" pass.
    """

    def __init__(self, infile, outfile, inconfig, outconfig, comm):
        self.infile = infile
        self.outfile = outfile
        self.inconfig = inconfig
        self.outconfig = outconfig
        self.comm = comm
        if HAVE_MPI:
            hostinfo = io.HostInfo.HOSTNAME
            self.outranks = hostinfo.get_collective(self.comm)
        else:
            self.outranks = {i: str(i) for i in range(self.comm.size)}

    def run(self, loggingfile):
        if not HAVE_MPI or (args.mpi is None and self.comm.size == 1):
            print("Opening data source")
            sys.stdout.flush()
            inseries = io.Series(self.infile, io.Access.read_only,
                                 self.inconfig)
            print("Opening data sink")
            sys.stdout.flush()
            outseries = io.Series(self.outfile, io.Access.create,
                                  self.outconfig)
            print("Opened input and output")
            sys.stdout.flush()
        else:
            print("Opening data source on rank {}.".format(self.comm.rank))
            sys.stdout.flush()
            inseries = io.Series(self.infile, io.Access.read_only, self.comm,
                                 self.inconfig)
            print("Opening data sink on rank {}.".format(self.comm.rank))
            sys.stdout.flush()
            outseries = io.Series(self.outfile, io.Access.create, self.comm,
                                  self.outconfig)
            print("Opened input and output on rank {}.".format(self.comm.rank))
            sys.stdout.flush()
        dump_times = DumpTimes(loggingfile)
        self.__copy(inseries, outseries, dump_times)
        dump_times.close()
        del inseries
        del outseries

    def __copy(self, src, dest, dump_times, current_path="/data/"):
        """
        Worker method.
        Copies data from src to dest. May represent any point in the openPMD
        hierarchy, but src and dest must both represent the same layer.
        """
        if (type(src) != type(dest)
                and not isinstance(src, io.IndexedIteration)
                and not isinstance(dest, io.Iteration)):
            raise RuntimeError(
                "Internal error: Trying to copy mismatching types")
        attribute_dtypes = src.attribute_dtypes
        # The following attributes are written automatically by openPMD-api
        # and should not be manually overwritten here
        ignored_attributes = {
            io.Series:
            ["basePath", "iterationEncoding", "iterationFormat", "openPMD"],
            io.Iteration: ["snapshot"]
        }
        for key in src.attributes:
            ignore_this_attribute = False
            for openpmd_group, to_ignore_list in ignored_attributes.items():
                if isinstance(src, openpmd_group):
                    for to_ignore in to_ignore_list:
                        if key == to_ignore:
                            ignore_this_attribute = True
            if not ignore_this_attribute:
                attr = src.get_attribute(key)
                attr_type = attribute_dtypes[key]
                dest.set_attribute(key, attr, attr_type)
        container_types = [
            io.Mesh_Container, io.Particle_Container, io.ParticleSpecies,
            io.Record, io.Mesh, io.Particle_Patches, io.Patch_Record
        ]
        if isinstance(src, io.Series):
            # main loop: read iterations of src, write to dest
            write_iterations = dest.write_iterations()
            for in_iteration in src.read_iterations():
                dump_times.now("Received iteration {}".format(
                    in_iteration.iteration_index))
                if self.comm.rank == 0:
                    # print("Iteration {0} contains {1} meshes:".format(
                    #     in_iteration.iteration_index,
                    #     len(in_iteration.meshes)))
                    # for m in in_iteration.meshes:
                    #     print("\t {0}".format(m))
                    # print("")
                    print(
                        "Iteration {0} contains {1} particle species:".format(
                            in_iteration.iteration_index,
                            len(in_iteration.particles)))
                    for ps in in_iteration.particles:
                        print("\t {0}".format(ps))
                        print("With records:")
                        for r in in_iteration.particles[ps]:
                            print("\t {0}".format(r))
                # With linear read mode, we can only load the source rank table
                # inside `read_iterations()` since it's a dataset.
                # For scalability, maybe read mpi_ranks_meta_info in parallel?
                self.inranks = src.mpi_ranks_meta_info
                out_iteration = write_iterations[in_iteration.iteration_index]
                sys.stdout.flush()
                self.__particle_patches = []
                loaded_chunks = self.__copy(
                    in_iteration, out_iteration, dump_times,
                    current_path + str(in_iteration.iteration_index) + "/")
                for species_name, species in loaded_chunks.particles.items():
                    if not debug:
                        break
                    print("Species {},\tloaded chunks:".format(species_name))
                    for chunk in species.chunks:
                        print("\t{}\tto {}".format(chunk.offset, chunk.extent))
                    for record_name, record in species.records.items():
                        print("\tRecord:", record_name)
                        for component_name, component in record.components.items(
                        ):
                            print("\t\tComponent:", component_name)
                            for chunk in component.chunks:
                                print(
                                    "\t\t\tLoaded chunk from {} to {}".format(
                                        chunk.offset, chunk.extent))
                dump_times.now("Closing incoming iteration {}".format(
                    in_iteration.iteration_index))
                in_iteration.close()
                dump_times.now("Sampling iteration {}".format(
                    in_iteration.iteration_index))
                loaded_chunks.sample(self.comm, 0.005)
                dump_times.now("Closing outgoing iteration {}".format(
                    in_iteration.iteration_index))
                out_iteration.close()
                dump_times.now("Closed outgoing iteration {}".format(
                    in_iteration.iteration_index))
                dump_times.flush()
                self.__particle_patches.clear()
                sys.stdout.flush()
        elif isinstance(src, io.Record_Component):
            shape = src.shape
            dtype = src.dtype
            dest.reset_dataset(io.Dataset(dtype, shape))
            if src.empty:
                # empty record component automatically created by
                # dest.reset_dataset()
                return None, shape
            elif src.constant:
                dest.make_constant(src.get_attribute("value"))
                return None, shape
            else:
                if not self.__my_chunks__:
                    chunk_table = src.available_chunks()
                    strategy = distribution_strategy(shape, self.comm.rank,
                                                     self.comm.size)
                    assignment = strategy.assign_chunks(
                        chunk_table, self.inranks, self.outranks)
                    self.__my_chunks__ = assignment[
                        self.comm.
                        rank] if self.comm.rank in assignment else []
                loaded_chunks = loaded_chunks_record_component()
                for chunk in self.__my_chunks__:
                    if debug:
                        end = chunk.offset.copy()
                        for i in range(len(end)):
                            end[i] += chunk.extent[i]
                        print("{}\t{}/{}:\t{} -- {}".format(
                            current_path, self.comm.rank, self.comm.size,
                            chunk.offset, end))
                    loaded_chunks.append(
                        loaded_chunk(
                            dest, chunk.offset, chunk.extent,
                            src.load_chunk(chunk.offset, chunk.extent)))
                return loaded_chunks, shape
        # elif isinstance(src, io.Patch_Record_Component):
        #     dest.reset_dataset(io.Dataset(src.dtype, src.shape))
        #     if self.comm.rank == 0:
        #         self.__particle_patches.append(
        #             particle_patch_load(src.load(), dest))
        elif isinstance(src, io.Iteration):
            # self.__copy(src.meshes, dest.meshes, dump_times,
            #             current_path + "meshes/")
            return self.__copy(src.particles, dest.particles, dump_times,
                               current_path + "particles/")
        elif isinstance(src, io.Particle_Container):
            res = loaded_chunks_iteration()
            for key in src:
                item = self.__copy(src[key], dest[key], dump_times,
                                   current_path + key + "/")
                res.insert_particle_species(key, item)
            return res
        elif isinstance(src, io.ParticleSpecies):
            res = loaded_chunks_species()
            self.__my_chunks__ = None
            for key in src:
                item, shape = self.__copy(src[key], dest[key], dump_times,
                                          current_path + key + "/")
                if item.components:
                    res.insert_record(key, item)
            # self.__copy(src.particle_patches, dest.particle_patches,
            #                 dump_times)
            res.chunks = self.__my_chunks__
            res.shape = shape
            self.__my_chunks__ = None
            return res
        elif isinstance(src, io.Record):
            res = loaded_chunks_record()
            for key in src:
                item, shape = self.__copy(src[key], dest[key], dump_times,
                                          current_path + key + "/")
                if item is not None:
                    res.insert_component(key, item)
            return res, shape
        elif any([
                isinstance(src, container_type)
                for container_type in container_types
        ]):
            raise RuntimeError("Unsupported openPMD container class: " +
                               str(src))
        else:
            raise RuntimeError("Unknown openPMD class: " + str(src))


def main():
    if not args.infile or not args.outfile:
        print("Please specify parameters --infile and --outfile.")
        sys.exit(1)
    if HAVE_MPI:
        communicator = MPI.COMM_WORLD
    else:
        communicator = FallbackMPICommunicator()
    run_pipe = pipe(args.infile, args.outfile, args.inconfig, args.outconfig,
                    communicator)

    max_logs = 20
    stride = (communicator.size + max_logs) // max_logs - 1  # sdiv, ceil(a/b)
    if stride == 0:
        stride += 1
    if communicator.rank % stride == 0:
        loggingfile = "./PIPE_times_{}.txt".format(communicator.rank)
    else:
        loggingfile = "/dev/null"
    print("Logging file on rank {} of {} is \"{}\".".format(
        communicator.rank, communicator.size, loggingfile))

    run_pipe.run(loggingfile)


if __name__ == "__main__":
    main()
    sys.exit()
