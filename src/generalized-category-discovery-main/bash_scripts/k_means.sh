PYTHON='/opt/conda/bin/python'

hostname
nvidia-smi

export CUDA_VISIBLE_DEVICES=0

# Get unique log file
SAVE_DIR=/home/padl22t4/padl/src/generalized-category-discovery-main/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.clustering.k_means --dataset 'cifar100' --semi_sup 'True' --use_ssb_splits 'True' \
 --use_best_model 'True' --max_kmeans_iter 200 --k_means_init 100 --warmup_model_exp_id '(20.06.2022_|_22.105)' \
 > ${SAVE_DIR}logfile_${EXP_NUM}.out