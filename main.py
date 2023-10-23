import math
import random
import numpy as np
import docplex.cp.utils_visu as visu
from pylab import rcParams

FILE_PATH = "./data.txt"

optimal_values = {
    "ft06": 55,
    "ft10": 930,
    "ft20": 1165,
    "la01": 666,
    "la02": 655,
    "la03": 597,
    "la04": 590,
    "la05": 593,
    "la06": 926,
    "la07": 890,
    "la08": 863,
    "la09": 951,
    "la10": 958,
    "la11": 1222,
    "la12": 1039,
    "la13": 1150,
    "la14": 1292,
    "la15": 1207,
    "la16": 945,
    "la17": 784,
    "la18": 848,
    "la19": 842,
    "la20": 902,
    "la21": 1046,
    "la22": 927,
    "la23": 1032,
    "la24": 935,
    "la25": 977,
    "la26": 1218,
    "la27": 1235,
    "la28": 1216,
    "la29": 1152,
    "la30": 1355,
    "la31": 1784,
    "la32": 1850,
    "la33": 1719,
    "la34": 1721,
    "la35": 1888,
    "la36": 1268,
}

# parameters
population_size = 50
mutation_probability = 0.05
C1 = 2.5
C2 = 0.5
C3 = 1.5
C4 = 1.5
max_iter = 1000
inertia_weight = 1.4
inertia_weight_max = 1.4
inertia_weight_min = 0.4
remap_iterations = 100
bifurcation = 4
neighbours = 9
q_c = 0.2
q_u = 0.7
max_idle_time = 0.4

# global variables
machines = np.array([])
jobs = np.array([], dtype=object)
particles = np.array([None] * population_size, dtype=object)
global_best_position = np.array([], dtype=np.float64)
global_best_encoded_position = np.array([], dtype=object)
global_best_make_span = math.inf
dimension = 0
optimal_value = 0


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
        self.encoded_position = np.array([], dtype=Operation)
        self.position = np.random.uniform(0, 1, dimension)
        self.velocity = np.zeros(dimension, dtype=np.float64)
        self.make_span = math.inf
        self.personal_best_position = np.zeros(dimension, dtype=np.float64)
        self.personal_best_make_span = math.inf
        self.local_best_position = np.zeros(dimension, dtype=np.float64)
        self.local_best_make_span = math.inf
        self.near_neighbour_best_position = np.zeros(dimension, dtype=np.float64)
        self.near_neighbour_make_span = math.inf

    def chaotically_remap_particle(self):
        for index, pos in enumerate(self.position):
            self.position[index] = bifurcation * pos * (1 - pos)

    def update_velocity(self):
        first_part = self.velocity * inertia_weight
        personal_part = (C1 * random.random()) * (self.personal_best_position - self.position)
        global_part = (C2 * random.random()) * (global_best_position - self.position)
        local_part = (C3 * random.random()) * (self.local_best_position - self.position)
        nearest_neighbor_part = (C4 * random.random()) * (self.near_neighbour_best_position - self.position)
        self.velocity = first_part + personal_part + global_part + local_part + nearest_neighbor_part
        fit_values_into_range(self.velocity, dimension * 0.1, -dimension * 0.1)

    def update_position(self):
        self.position += self.velocity
        self.rk_encoding()

    def crossover(self):
        if random.random() > q_u:
            self.position = global_best_position
            self.encoded_position = global_best_encoded_position
            self.make_span = global_best_make_span

    def rk_encoding(self):
        operation_priorities = self.operation_priorities()
        operation_sequence = np.array([], dtype=Operation)
        machine_schedules = dict((machine_id, []) for machine_id in machines)
        scheduleable_operations = dict([(j.job_id, {"operation": j.operations[0], "fi": 0, "delta": 0}) for j in jobs])

        for i in range(dimension):
            best_fi = math.inf
            best_delta = math.inf
            for job_id in scheduleable_operations:
                operation = scheduleable_operations[job_id]["operation"]
                machine = machine_schedules[operation.machine_id]
                delta = 0
                added = False
                if operation.operation_id > 0:
                    successor_machine = jobs[operation.job_id].operations[operation.operation_id - 1].machine_id
                    for machine_schedule in machine_schedules[successor_machine]:
                        if (machine_schedule["operation"].job_id == operation.job_id and
                                machine_schedule["operation"].operation_id == (operation.operation_id - 1)):

                            if len(machine) == 0 or machine[-1]["elapsed_time"] < machine_schedule["elapsed_time"]:
                                delta = machine_schedule["elapsed_time"]
                            else:
                                delta = machine[-1]["elapsed_time"]

                            added = True
                            break

                if not added:
                    if len(machine) == 0:
                        delta = 0
                    else:
                        delta = machine[-1]["elapsed_time"]

                fi = delta + operation.duration
                if fi < best_fi:
                    best_fi = fi

                if delta < best_delta:
                    best_delta = delta

                scheduleable_operations[job_id]["delta"] = delta
                scheduleable_operations[job_id]["fi"] = fi

            requirement = best_delta + max_idle_time * (best_fi - best_delta)
            possible_schedules = [scheduleable_operation for scheduleable_operation in scheduleable_operations.values()
                                  if scheduleable_operation["delta"] <= requirement]

            operation = None
            if len(possible_schedules) > 1:
                smallest_index = math.inf
                smallest_fi = 0
                for possible_schedule in possible_schedules:
                    op = possible_schedule['operation']
                    operation_priority = operation_priorities[op.job_id][op.operation_id]
                    if operation_priority < smallest_index:
                        smallest_index = operation_priority
                        smallest_fi = possible_schedule["fi"]
                        operation = op

                operation_sequence = np.append(operation_sequence, operation)
                machine_schedules[operation.machine_id].append({"operation": operation, "elapsed_time": smallest_fi})
            else:
                operation = possible_schedules[0]["operation"]
                operation_sequence = np.append(operation_sequence, operation)
                machine_schedules[operation.machine_id].append({"operation": operation,
                                                                "elapsed_time": possible_schedules[0]["fi"]})

            if operation.operation_id != len(jobs[operation.job_id].operations) - 1:
                scheduleable_operations[operation.job_id] = {
                    "operation": jobs[operation.job_id].operations[operation.operation_id + 1]
                }
            else:
                scheduleable_operations.pop(operation.job_id)

        self.encoded_position = operation_sequence
        self.make_span = max(machine_schedule[-1]["elapsed_time"] for machine_schedule in machine_schedules.values())

    def operation_priorities(self):
        operation_priorities = dict([(j.job_id, []) for j in jobs])
        position_copy = self.position.copy()
        job_num = len(jobs)
        for order in range(dimension):
            smallest_value = math.inf
            smallest_value_index = 0
            for index, value in enumerate(position_copy):
                if value < smallest_value:
                    smallest_value = value
                    smallest_value_index = index
            position_copy[smallest_value_index] = math.inf
            operation_priorities[(order + 1) % job_num].append(smallest_value_index)

        for operation_list in operation_priorities.values():
            operation_list.sort()
        return operation_priorities

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
            if random.random() < 0.5:
                self.swapping_operation()
            else:
                self.insertion_operation()

    def update_personal_best(self):
        if self.personal_best_make_span > self.make_span:
            self.personal_best_position = self.position
            self.personal_best_make_span = self.make_span

    def update_local_best(self, index):
        local_particles = np.array([], dtype=PSOParticle)

        for i in range((neighbours - 1) // 2):
            local_particles = np.append(local_particles, particles[index - i - 1])

        local_particles = np.append(local_particles, [self])

        for i in range((neighbours - 1) // 2):
            overflow = index + i + 1 - population_size
            if overflow >= 0:
                local_particles = np.append(local_particles, particles[overflow])
            else:
                local_particles = np.append(local_particles, particles[index + i + 1])

        for particle in local_particles:
            if self.local_best_make_span > particle.personal_best_make_span:
                self.local_best_position = particle.personal_best_position
                self.local_best_make_span = particle.personal_best_make_span

    def update_nearest_neighbor_best(self, index):
        for pos_index in range(dimension):
            best = 0
            best_fdr = -math.inf
            for i, particle in enumerate(particles):
                if i != index:
                    fdr = ((self.make_span - particle.personal_best_make_span) /
                           abs(particle.personal_best_position[pos_index] - self.position[pos_index] + 1))
                    if fdr > best_fdr:
                        best_fdr = fdr
                        best = particle.personal_best_position[pos_index]
            self.near_neighbour_best_position[pos_index] = best


def fit_values_into_range(values, maximum, minimum):
    for index, value in enumerate(values):
        if math.isnan(value):
            values[index] = 0
        elif value > maximum:
            values[index] = maximum
        elif value < minimum:
            values[index] = minimum


def update_global_best_position():
    global global_best_make_span
    global global_best_position
    global global_best_encoded_position
    for particle in particles:
        if particle.make_span < global_best_make_span:
            global_best_make_span = particle.make_span
            global_best_position = particle.position
            global_best_encoded_position = particle.encoded_position


def update_iter_sensitive_params(current_iteration):
    global inertia_weight
    global C1
    global C2
    inertia_weight = (inertia_weight_max -
                      (current_iteration * (inertia_weight_max - inertia_weight_min) / max_iter))
    C1 = 2.5 - (2 * current_iteration / max_iter)
    C2 = 0.5 + (2 * current_iteration / max_iter)


def iteration(current_iteration):
    for index, particle in enumerate(particles):
        particle.mutation()
        particle.update_personal_best()
        particle.update_local_best(index)
        particle.update_nearest_neighbor_best(index)

    update_global_best_position()
    update_iter_sensitive_params(current_iteration)

    if current_iteration % 50 == 0:
        print()
        print(f"ITERATION #{current_iteration}")
        print("############################################")
        print(f"Global best make span: {global_best_make_span}")
        print("############################################")

    for particle in particles:
        if random.random() > q_c:
            particle.update_velocity()
            particle.update_position()
        else:
            particle.crossover()


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


def display_gantt():
    machine_dictionary = {}
    for machine in machines:
        machine_dictionary[machine] = []

    for operation in global_best_encoded_position:
        machine = machine_dictionary[operation.machine_id]
        added = False
        for machine_dict in machine_dictionary.values():
            for machine_schedule in machine_dict:
                if (machine_schedule["operation"].job_id == operation.job_id
                        and machine_schedule["operation"].operation_id == (operation.operation_id - 1)):

                    if len(machine) == 0 or machine[-1]["end"] < machine_schedule["end"]:
                        machine.append({"operation": operation,
                                        "start": machine_schedule["end"],
                                        "end": machine_schedule["end"] + operation.duration})
                    else:
                        machine.append({"operation": operation,
                                        "start": machine[-1]["end"],
                                        "end": machine[-1]["end"] + operation.duration})
                    added = True
                    break
        if not added:
            if len(machine) == 0:
                machine.append({"operation": operation,
                                "start": 0,
                                "end": operation.duration})
            else:
                machine.append({"operation": operation,
                                "start": machine[-1]["end"],
                                "end": machine[-1]["end"] + operation.duration})

    longest_machine_time = 0
    for machine_dict in machine_dictionary.values():
        if longest_machine_time < machine_dict[-1]["end"]:
            longest_machine_time = machine_dict[-1]["end"]

    rcParams['figure.figsize'] = 20, 5
    rcParams['font.size'] = 6
    visu.timeline('Job Shop Solution', 0, longest_machine_time)
    for index in range(len(machine_dictionary)):
        visu.sequence(name=str(index + 1))
        for machine_schedule in machine_dictionary[index]:
            operation = machine_schedule["operation"]
            visu.interval(machine_schedule["start"], machine_schedule["end"], "salmon",
                          f"O-{operation.job_id + 1}-{operation.operation_id + 1}")
    visu.show()


if __name__ == '__main__':
    while len(jobs) == 0:
        dataset = input("Please type the name of the dataset of your desired JSS problem:")
        read_file_dataset(dataset)
        if len(jobs) == 0:
            print()
            print("A Dataset with this name doesn't exist")
        elif optimal_values.keys().__contains__(dataset):
            optimal_value = optimal_values[dataset]

    print()
    print("Job Shop Scheduling problem definition")
    print("############################################")
    print(f"Number of machines: {len(machines)}")
    print(f"Number of jobs: {len(jobs)}")
    for job in jobs:
        print(job.__str__())
    print("############################################")

    for x in range(remap_iterations):
        for p in particles:
            p.chaotically_remap_particle()

    for p in particles:
        p.rk_encoding()

    print()
    print("Initialization complete")
    print()

    for it in range(1, max_iter + 1):
        iteration(it)
        if global_best_make_span == optimal_value:
            break

    print()
    print("############################################")
    print(f"Best make span: {global_best_make_span}")
    print("############################################")

    display_gantt()
