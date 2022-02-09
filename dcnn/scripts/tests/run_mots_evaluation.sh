python scripts/tests/standard_rcnn_tracker_test.py --mots_evaluation mots_tools/mots_eval/val.seqmap
echo "Calculating MOTS metrics for sequences..."
python mots_tools/mots_eval/eval.py output/evaluation_results datasets/data_tracking_image_2/instances_txt mots_tools/mots_eval/val.seqmap > output/evaluation_results/MOTS_metrics.txt