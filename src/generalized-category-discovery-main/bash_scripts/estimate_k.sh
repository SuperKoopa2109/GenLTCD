PYTHON='/opt/conda/bin/python'

hostname

# Get unique log file
SAVE_DIR=/home/padl22t4/padl/src/generalized-category-discovery-main/osr_novel_categories/dev_outputs/

EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
EXP_NUM=$((${EXP_NUM}+1))
echo $EXP_NUM

${PYTHON} -m methods.estimate_k.estimate_k --max_classes 100 --dataset_name cifar100 --warmup_model_exp_id '(20.06.2022_|_22.105)' --search_mode other \
        > ${SAVE_DIR}logfile_${EXP_NUM}.out