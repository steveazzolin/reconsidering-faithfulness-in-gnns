##
# MOTIF2
##
# LECI - ACC
goodtg --config_path final_configs/GOODMotif2/basis/covariate/LECI.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# LECI - FAITH
goodtg --config_path final_configs/GOODMotif2/basis/covariate/LECI.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT - ACC
goodtg --config_path final_configs/GOODMotif2/basis/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# GSAT - FAITH
goodtg --config_path final_configs/GOODMotif2/basis/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# CIGA - ACC
goodtg --config_path final_configs/GOODMotif2/basis/covariate/CIGA.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# CIGA - FAITH
goodtg --config_path final_configs/GOODMotif2/basis/covariate/CIGA.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset


##
# CMNIST
##
# LECI - ACC
goodtg --config_path final_configs/GOODCMNIST/color/covariate/LECI.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# GSAT - ACC
goodtg --config_path final_configs/GOODCMNIST/color/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# CIGA - ACC
goodtg --config_path final_configs/GOODCMNIST/color/covariate/CIGA.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1


##
# LBAPcore
##
# LECI - ACC
goodtg --config_path final_configs/LBAPcore/assay/covariate/LECI.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# GSAT - ACC
goodtg --config_path final_configs/LBAPcore/assay/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# CIGA - ACC
goodtg --config_path final_configs/LBAPcore/assay/covariate/CIGA.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1


##
# SST2
##
# LECI - ACC
goodtg --config_path final_configs/GOODSST2/length/covariate/LECI.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# LECI - FAITH
goodtg --config_path final_configs/GOODSST2/length/covariate/LECI.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT - ACC
goodtg --config_path final_configs/GOODSST2/length/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# GSAT - FAITH
goodtg --config_path final_configs/GOODSST2/length/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# CIGA - ACC
goodtg --config_path final_configs/GOODSST2/length/covariate/CIGA.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# CIGA - FAITH
goodtg --config_path final_configs/GOODSST2/length/covariate/CIGA.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset


################## SEGNNs ##################


##
# BAMs
##
# GSAT - ACC
goodtg --config_path final_configs/MultiShapes/basis/no_shift/GSAT.yaml --seeds "1/2/3/10/11" --task test --average_edge_attn mean --gpu_idx 1
# GSAT+ER - ACC
goodtg --config_path final_configs/MultiShapes/basis/no_shift/GSAT.yaml --seeds "1/2/3/10/11" --task test --average_edge_attn mean --gpu_idx 1 --mitigation_readout weighted
# GSAT+HS - ACC
goodtg --config_path final_configs/MultiShapes/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --mitigation_expl_scores hard
# GSAT+ER+HS - ACC
goodtg --config_path final_configs/MultiShapes/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --mitigation_expl_scores hard --mitigation_readout weighted
# GSAT - FAITH
goodtg --config_path final_configs/MultiShapes/basis/no_shift/GSAT.yaml --seeds "1/2/3/10/11" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT+ER - FAITH
goodtg --config_path final_configs/MultiShapes/basis/no_shift/GSAT.yaml --seeds "1/2/3/10/11" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset  --mitigation_readout weighted
# GSAT+HS - FAITH
goodtg --config_path final_configs/MultiShapes/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset  --mitigation_expl_scores hard
# GSAT+ER+HS - FAITH
goodtg --config_path final_configs/MultiShapes/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset  --mitigation_expl_scores hard --mitigation_readout weighted

##
# MOTIF-size
##
# GSAT - ACC
goodtg --config_path final_configs/GOODMotif/size/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --extra_param True 10 0.2 --ood_param 10.0
# GSAT+ER - ACC
goodtg --config_path final_configs/GOODMotif/size/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --extra_param True 10 0.2 --ood_param 10.0 --mitigation_readout weighted
# GSAT+HS - ACC
goodtg --config_path final_configs/GOODMotif/size/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --extra_param True 10 0.2 --ood_param 10.0 --mitigation_expl_scores hard
# GSAT+ER+HS - ACC
goodtg --config_path final_configs/GOODMotif/size/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --extra_param True 10 0.2 --ood_param 10.0 --mitigation_readout weighted --mitigation_expl_scores hard
# GSAT - FAITH
goodtg --config_path final_configs/GOODMotif/size/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --extra_param True 10 0.2 --ood_param 10.0 --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT+ER - FAITH
goodtg --config_path final_configs/GOODMotif/size/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --extra_param True 10 0.2 --ood_param 10.0  --mitigation_readout weighted --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT+HS - FAITH
goodtg --config_path final_configs/GOODMotif/size/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --extra_param True 10 0.2 --ood_param 10.0  --mitigation_expl_scores hard --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT+HS - FAITH
goodtg --config_path final_configs/GOODMotif/size/covariate/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --extra_param True 10 0.2 --ood_param 10.0  --mitigation_readout weighted --mitigation_expl_scores hard --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset


##
# BBBP
##
# GSAT - ACC
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1
# GSAT+ER - ACC
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --mitigation_readout weighted
# GSAT+HS - ACC
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --mitigation_expl_scores hard
# GSAT+ER+HS - ACC
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task test --average_edge_attn mean --gpu_idx 1 --mitigation_readout weighted --mitigation_expl_scores hard
# GSAT - FAITH
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT+ER - FAITH
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --mitigation_readout weighted --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT+HS - FAITH
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --mitigation_expl_scores hard --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset
# GSAT+ER+HS - FAITH
goodtg --config_path final_configs/BBBP/basis/no_shift/GSAT.yaml --seeds "1/2/3/4/5" --task eval_metric --average_edge_attn mean --gpu_idx 1 --mitigation_readout weighted --mitigation_expl_scores hard --metrics "suff++/nec" --splits "id_test" --mask --debias --samplingtype deconfounded --nec_number_samples prop_G_dataset


