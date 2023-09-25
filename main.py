import math
import random
import numpy as np

FILE_PATH = "./data.txt"

machines = np.array([])
jobs = np.array([], object)
particles = np.array([], object)
global_best_position = np.array([])
global_best_make_span = math.inf
dimension = 0

# parameters
mutation_probability = 0.01
C1 = 2.0
C2 = 2.0
max_iter = 1000
inertia_weight = 1.4
inertia_weight_max = 1.4
inertia_weight_min = 0.4


class Operation:

    def __init__(self, operation_id, job_id, machine_id, duration):
        self.operation_id = operation_id
        self.job_id = job_id
        self.machine_id = machine_id
        self.duration = duration

    def __str__(self):
        return f"\t\tOPERATION #{self.operation_id + 1}: MACHINE={self.machine_id + 1} DURATION={self.duration}\n"


class Job:

    def __init__(self, job_id, operations):
        self.job_id = job_id
        self.operations = operations

    def __str__(self):
        job_str = f"JOB #{job.job_id + 1}:\n"
        for operation in self.operations:
            job_str += operation.__str__()
        return job_str


class PSOParticle:

    def __init__(self):
        self.encoded_position = None
        self.position = np.zeros(dimension)
        self.velocity = np.zeros(dimension)
        self.make_span = math.inf
        self.local_best_position = np.zeros(dimension)
        self.local_best_make_span = math.inf

    def __str__(self):
        return (f"POSITION={self.position} MAKE SPAN={self.make_span} " +
                f"LOCAL BEST POSITION={self.local_best_position} LOCAL BEST MAKE SPAN={self.local_best_make_span}\n")

    # updates velocity according to current position and velocity
    def update_velocity(self):
        first_part = self.velocity * inertia_weight
        second_part = C1 * random.random() * (self.local_best_position - self.position)
        third_part = C2 * random.random() * (global_best_position - self.position)
        self.velocity = first_part + second_part + third_part

    # moves particle according to current position and velocity
    def update_position(self):
        self.position += self.velocity
        self.rk_encoding()
        self.make_span = get_make_span_by_position(self.encoded_position)

    def rk_encoding(self):
        # turn integer series into permutation of job indexes
        integer_series = self.position_integer_series()
        for index, value in enumerate(integer_series.copy()):
            integer_series[index] = value % len(jobs)

        # turn job index series into operation sequence
        operation_sequence = np.array([None] * dimension, dtype=Operation)
        job_occurrences = {}
        for index, job_index in enumerate(integer_series):
            if job_index in job_occurrences:
                job_occurrences[job_index] += 1
            else:
                job_occurrences[job_index] = 0
            operation_sequence[index] = jobs[job_index].operations[job_occurrences[job_index]]

        self.encoded_position = operation_sequence
        return operation_sequence

    # turn real numbers into integer series
    def position_integer_series(self):
        integer_series = np.zeros(dimension)
        position_copy = self.position.copy()
        for order in range(dimension):
            smallest_value = math.inf
            smallest_value_index = 0
            for index, value in enumerate(position_copy):
                if value < smallest_value:
                    smallest_value = value
                    smallest_value_index = index
            position_copy[smallest_value_index] = math.inf
            integer_series[smallest_value_index] = order + 1
        return integer_series

    def swapping_operation(self):
        indexes = random.sample(range(dimension), 2)
        temp = self.position[indexes[0]]
        self.position[indexes[0]] = self.position[indexes[1]]
        self.position[indexes[1]] = temp
        self.rk_encoding()

    def insertion_operation(self):
        indexes = random.sample(range(dimension), 2)
        temp = self.position[indexes[0]]
        deleted = np.delete(self.position, indexes[0])
        self.position = np.insert(deleted, indexes[1], temp)
        self.rk_encoding()

    # swapping operation for local search
    # def swapping_operation(self):
    #     first_index, second_index, first_operation, second_operation = self.get_randoms_for_local_search()
    #     operations_copy = self.encoded_position.copy()
    #     operations_copy[first_index] = second_operation
    #     operations_copy[second_index] = first_operation
    #     return operations_copy

    # insertion operation for local search
    # def insertion_operation(self):
    #     first_index, second_index, first_operation, second_operation = self.get_randoms_for_local_search()
    #     operations_copy = self.encoded_position.copy()
    #     operations_copy.pop(first_index)
    #     operations_copy.insert(second_index, first_operation)
    #     return operations_copy

    # Helper function for local searches, gets dimensions and their operations
    # def get_randoms_for_local_search(self):
    #     chosen_jobs = random.sample(jobs, 2)
    #     first_operation = random.sample(chosen_jobs[0].operations, 1)
    #     second_operation = random.sample(chosen_jobs[1].operations, 1)
    #
    #     first_index = 0
    #     second_index = 0
    #     first_occurrences_found = 0
    #     second_occurrences_found = 0
    #     for index, operation in enumerate(self.encoded_position):
    #         if operation.job_id == first_operation.job_id:
    #             if first_occurrences_found == first_operation.operation_id:
    #                 first_index = index
    #             first_occurrences_found += 1
    #         elif operation.job_id == second_operation.job_id:
    #             if second_occurrences_found == second_operation.operation_id:
    #                 second_index = index
    #             second_occurrences_found += 1
    #     return first_index, second_index, first_operation, second_operation

    # mutate
    def mutation(self):
        if random.random() <= mutation_probability:
            old_position = self.position
            old_encoded_position = self.encoded_position

            if random.random() < 0.5:
                self.swapping_operation()
            else:
                self.insertion_operation()

            mutated_make_span = get_make_span_by_position(self.encoded_position)

            if self.make_span >= mutated_make_span:
                self.make_span = mutated_make_span
            else:
                self.position = old_position
                self.encoded_position = old_encoded_position

    # update the local best position
    def update_local_best(self):
        if self.local_best_make_span > self.make_span:
            self.local_best_position = self.position
            self.local_best_make_span = self.make_span


def get_make_span_by_position(position):
    machine_dictionary = dict(zip(machines, [{"elapsed_time": 0, "current_operation": None}] * len(machines)))
    for operation in position:
        machine = machine_dictionary[operation.machine_id]
        if machine["current_operation"] is None:
            machine["elapsed_time"] = operation.duration
        else:
            added = False
            for machine_dict in machine_dictionary:
                if machine_dict["current_operation"].job_id == operation.job_id:
                    machine["elapsed_time"] = machine_dict["elapsed_time"] + operation.duration
                    added = True
            if not added:
                machine["elapsed_time"] += operation.duration
        machine["current_operation"] = operation

    longest_machine_time = 0
    for machine_dict in machine_dictionary:
        if longest_machine_time < machine_dict["elapsed_time"]:
            longest_machine_time = machine_dict["elapsed_time"]
    return longest_machine_time


def update_global_position():
    for particle in particles:
        global global_best_make_span
        global global_best_position
        if particle.make_span < global_best_make_span:
            global_best_make_span = particle.make_span
            global_best_position = particle.position


def update_inertia_weight(current_iteration):
    global inertia_weight
    inertia_weight = inertia_weight_max - (current_iteration * (inertia_weight_max - inertia_weight_min) / max_iter)


def iteration(current_iteration):
    print()
    print(f"ITERATION #{current_iteration}")
    print("############################################")
    for particle in particles:
        particle.mutation()
        particle.update_local_best()

    update_global_position()
    print(f"Global best make span: {global_best_make_span}")
    update_inertia_weight(current_iteration)

    for particle in particles:
        particle.update_velocity()
        particle.update_position()
        print(particle.__str__())


def read_file_dataset(dataset_name):
    global machines
    global jobs
    global dimension
    with open(FILE_PATH, 'r') as f:
        while True:
            line = f.readline()
            if line.find(f"instance {dataset_name}") != -1:
                f.readline()
                f.readline()
                f.readline()
                line = f.readline().strip()
                numbers = line.split(" ")
                num_of_jobs = int(numbers[0])
                num_of_machines = int(numbers[1])
                machines = np.arange(num_of_machines)
                jobs = np.array([None] * num_of_jobs, dtype=Job)

                for job_index in range(num_of_jobs):
                    line = f.readline().strip()
                    numbers = line.split(" ")
                    operations = np.array([], dtype=Operation)

                    for i in range(0, len(numbers), 2):
                        operations = np.append(operations,
                                               [Operation(i - len(operations), job_index,
                                                          int(numbers[i]), int(numbers[i + 1]))])
                        dimension += 1

                    jobs[job_index] = Job(job_index, operations)
            elif line.find("+ EOF +") != -1:
                break


# def generate_random_problem(self, num_machines, num_jobs):
#     self.machines = np.arange(num_machines)
#     times = np.arange(1, 10)
#     for job_id in np.arange(num_jobs):
#         operations = []
#         for machine_id in np.random.permutation(self.machines):
#             operations.append(Operation(machine_id, job_id, np.random.choice(times)))
#         jobs.append(Job(job_id, operations))


if __name__ == '__main__':
    dataset = input("Please type the name of the dataset of your desired JSS problem:")
    read_file_dataset(dataset)

    print()
    print("Job Shop Scheduling problem definition")
    print("############################################")
    print(f"Number of machines: {len(machines)}")
    print(f"Number of jobs: {len(jobs)}")
    for job in jobs:
        print(job.__str__())
    print("############################################")
