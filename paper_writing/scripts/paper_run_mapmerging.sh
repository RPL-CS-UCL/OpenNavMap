# SeqMatch + RawConf - in_order (NO KF Culling)
# rosrun litevloc run_map_merging.sh 0 nokf_spgo_rc_seqmatch master_nocalib_pretrain;

# SeqMatch + CalibConf - in_order (NO KF Culling)
# rosrun litevloc run_map_merging.sh 0 nokf_spgo_cc_seqmatch master_calib_pretrain;

# Proposed (SeqMatch + CalibConf + KF Culling) - in_order (0) random_order (others)
# for ((i=0; i<=9; i++)); do
#   rosrun litevloc run_map_merging.sh "$i" kf_spgo_cc_seqmatch master_calib_pretrain;
# done
