import math
import random
import numpy as np

FILE_PATH = "./data.txt"

# parameters
population_size = 50
mutation_probability = 0.01
C1 = 2.0
C2 = 2.0
max_iter = 500
inertia_weight = 1.4
inertia_weight_max = 1.4
inertia_weight_min = 0.4

# global variables
machines = np.array([])
jobs = np.array([], dtype=object)
particles = np.array([None] * population_size, dtype=object)
global_best_position = np.array([], dtype=np.float64)
global_best_encoded_position = np.array([], dtype=object)
global_best_make_span = math.inf
dimension = 0


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
        self.position = np.random.uniform(-dimension, dimension, dimension)
        self.velocity = np.zeros(dimension, dtype=np.float64)
        self.make_span = math.inf
        self.local_best_position = np.zeros(dimension, dtype=np.float64)
        self.local_best_make_span = math.inf

    def __str__(self):
        return (f"POSITION={self.position} MAKE SPAN={self.make_span} " +
                f"LOCAL BEST POSITION={self.local_best_position} LOCAL BEST MAKE SPAN={self.local_best_make_span}\n")

    def update_velocity(self):
        first_part = self.velocity * inertia_weight
        second_part = C1 * random.random() * (self.local_best_position - self.position)
        third_part = C2 * random.random() * (global_best_position - self.position)
        self.velocity = first_part + second_part + third_part
        fit_values_into_range(self.velocity, dimension*0.1, -dimension*0.1)

    def update_position(self):
        self.position += self.velocity
        fit_values_into_range(self.position, dimension, -dimension)
        self.rk_encoding()
        self.make_span = get_make_span_by_position(self.encoded_position)

    def rk_encoding(self):
        integer_series = self.position_integer_series()
        for index, value in enumerate(integer_series.copy()):
            integer_series[index] = value % len(jobs)

        operation_sequence = np.array([None] * dimension, dtype=Operation)
        job_occurrences = {}
        for index, job_index in enumerate(integer_series):
            if job_index in job_occurrences.keys():
                job_occurrences[job_index] += 1
            else:
                job_occurrences[job_index] = 0
            operation_sequence[index] = jobs[int(job_index)].operations[int(job_occurrences[job_index])]

        self.encoded_position = operation_sequence
        return operation_sequence

    def position_integer_series(self):
        integer_series = np.zeros(dimension, dtype=int)
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


def fit_values_into_range(values, maximum, minimum):
    for index, value in enumerate(values):
        if math.isnan(value):
            values[index] = 0
        elif value > maximum:
            values[index] = maximum
        elif value < minimum:
            values[index] = minimum


def get_make_span_by_position(position):
    machine_dictionary = {}
    for machine in machines:
        machine_dictionary[machine] = []

    for operation in position:
        machine = machine_dictionary[operation.machine_id]
        added = False
        for machine_dict in machine_dictionary.values():
            for machine_schedule in machine_dict:
                if (machine_schedule["operation"].job_id == operation.job_id
                        and machine_schedule["operation"].operation_id == (operation.operation_id - 1)):

                    if len(machine) == 0 or machine[-1]["elapsed_time"] < machine_schedule["elapsed_time"]:
                        machine.append({"operation": operation,
                                        "elapsed_time": machine_schedule["elapsed_time"] + operation.duration})
                    else:
                        machine.append({"operation": operation,
                                        "elapsed_time": machine[-1]["elapsed_time"] + operation.duration})
                    added = True
                    break
        if not added:
            if len(machine) == 0:
                machine.append({"operation": operation,
                                "elapsed_time": operation.duration})
            else:
                machine.append({"operation": operation,
                                "elapsed_time": machine[-1]["elapsed_time"] + operation.duration})

    longest_machine_time = 0
    for machine_dict in machine_dictionary.values():
        if longest_machine_time < machine_dict[-1]["elapsed_time"]:
            longest_machine_time = machine_dict[-1]["elapsed_time"]
    return longest_machine_time


def update_global_position():
    global global_best_make_span
    global global_best_position
    global global_best_encoded_position
    for particle in particles:
        if particle.make_span < global_best_make_span:
            global_best_make_span = particle.make_span
            global_best_position = particle.position
            global_best_encoded_position = particle.encoded_position


def update_inertia_weight(current_iteration):
    global inertia_weight
    inertia_weight = inertia_weight_max - (current_iteration * (inertia_weight_max - inertia_weight_min) / max_iter)


def iteration(current_iteration):
    print()
    print(f"ITERATION #{current_iteration + 1}")
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

    print("############################################")


def read_file_dataset(dataset_name):
    global machines
    global jobs
    global dimension
    global particles
    global global_best_position
    with open(FILE_PATH, 'r') as f:
        while True:
            line = f.readline()
            if line.find(f"instance {dataset_name}") != -1:
                f.readline()
                f.readline()
                f.readline()
                line = f.readline().strip()
                numbers = [number for number in line.split(" ") if number != ""]
                num_of_jobs = int(numbers[0])
                num_of_machines = int(numbers[1])
                machines = np.arange(num_of_machines)
                jobs = np.array([None] * num_of_jobs, dtype=Job)

                for job_index in range(num_of_jobs):
                    line = f.readline().strip()
                    numbers = [number for number in line.split(" ") if number != ""]
                    operations = np.array([], dtype=Operation)

                    for i in range(0, len(numbers), 2):
                        operations = np.append(operations,
                                               [Operation(i - len(operations), job_index,
                                                          int(numbers[i]), int(numbers[i + 1]))])
                        dimension += 1

                    jobs[job_index] = Job(job_index, operations)

                particles = np.array([PSOParticle()] * population_size)
                global_best_position = np.zeros(dimension)
                break
            elif line.find("+ EOF +") != -1:
                break


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

    for it in range(max_iter):
        iteration(it)

    for op in global_best_encoded_position:
        print(f"JOB: {op.job_id + 1}" + op.__str__())
    print(f"Best position: {global_best_position}")
    print(f"Best make span: {global_best_make_span}")
