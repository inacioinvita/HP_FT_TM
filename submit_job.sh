   #!/bin/bash

   # Convert line endings to Unix format
   sed -i 's/\r$//' run_sage_train_inf_eval.job

   # Submit the job
   sbatch run_sage_train_inf_eval.job