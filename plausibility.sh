set -e

##
# Only for datasets with annotated ground truth
##

MODEL=GSAT
for DATASET in GOODMotif2/basis GOODMotif/size; do
       goodtg --config_path final_configs/${DATASET}/covariate/${MODEL}.yaml \
              --seeds "1/2/3/4/5" \
              --task eval_metric \
              --metrics "plaus" \
              --average_edge_attn mean \
              --mitigation_sampling feat \
              --gpu_idx 1 \
              --save_metrics \
              --log_id plausibility
       echo "DONE ${MODEL} ${DATASET}"
done
