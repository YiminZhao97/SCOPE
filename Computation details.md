**Computing partition:** Jobs were run on the Fred Hutch  SLURM partition. Compute nodes are dual-socket **Intel Xeon Gold 6154 @ 3.00 GHz (2 × 18 physical cores, 72 threads per node)**.

Setty dataset: 

- For preprocessing, the scVI was trained using single NVIDIA L40S. The estimated running time is about 5 minutes.
- For running SCOPE, each job was executed on a single node with **3 CPU cores** and **90 GB RAM**, with an estimated runtime of approximately **10 minutes**.

Persad dataset:  

- For preprocessing of RNA modality, the scVI was trained using single NVIDIA L40S. The estimated running time is about 5 minutes.
- For running SCOPE on RNA modality, each job was executed on a single node with **3 CPU cores** and **90 GB RAM**, with an estimated runtime of approximately **10 minutes**.
- For preprocessing of ATAC modality, the PeakVI was trained using single NVIDIA L40S. The estimated running time is about **20 minutes**.
- For running SCOPE on ATAC modality, each job was executed on a single node with **3 CPU cores** and **90 GB RAM**, with an estimated runtime of approximately **10 minutes**.

Weinreb dataset:

- For running SCOPE, each job was executed on a single node with **8 CPU cores** and **160 GB RAM**, using **multi-threaded parallelization across the 8 cores**, with a total runtime of approximately **4 hours**.

Wohlschlegel dataset:

- For running SCOPE on RNA modality, each job was executed on a single node with **3 CPU cores** and **90 GB RAM**, with an estimated runtime of approximately **30 minutes**.
- For preprocessing, the PeakVI was trained using single NVIDIA L40S. The estimated running time is about **30 minutes**.
- For running SCOPE on ATAC modality, each job was executed on a single node with **3 CPU cores** and **90 GB RAM**, with an estimated runtime of approximately **1 hour**.

