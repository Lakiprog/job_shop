# job_shop
Job Shop Optimization

This project tries to find the optimal solution for Job Shop Scheduling problems using the Particle Swarm Optimization algorithm.
The file that contains the application is **main.py**.
The libraries that need to be installed before running the application are:
  **numpy**
  **docplex.cp.utils_visu**
  **matplotlib**

The data for the Job Shop Scheduling problems can be found inside the **data.txt** file, these are standard problems that are used in many other papers.
When the application is started you have to type the name of a JSS problem in no-caps, for Example (ft06, la01 etc..).
When an optimal solution is found, or the project ran for a set of max iterations, a Gantt chart will be displayed showing the best schedule found.
The achieved results in 10 runs is found in the excel sheet **Results.xslx**.
