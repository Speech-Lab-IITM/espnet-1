#! /bin/bash
#SBATCH --nodes=1
#SBATCH --partition=nltmp
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:A100-SXM4:1
#SBATCH --time=160:00:00
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running $SLURM_NTASKS tasks."
echo "Job id is $SLURM_JOBID"
echo "Job submission directory is : $SLURM_SUBMIT_DIR"
export ftp_proxy=http://172.50.0.50:9090
export https_proxy=http://172.50.0.50:9090
export http_proxy=http://172.50.0.50:9090
source path.sh
echo "joint 1h 0.3 ctc finetuning"
#srun time ./run_pre_char.sh --asr_tag lr0.0001_3.5Mbb_4ag_char_loadpre_epoch31_onlyctc --stop_stage 12 --stage 13
srun time ./run_pre_char.sh \
--train_set "train_1h" \
--asr_config conf/tuning/train_joint_char_0.3ctc_1h.yaml \
--asr_tag lr0.00002_3.2Mbb_1ag_char_jointscratch_epoch101_1h_0.3ctc_load_pre_decoder \
--stage 10 \
--stop_stage 11
