source ~/.bashrc
source /usr/wipp/conda/24.5.0u/etc/profile.d/conda.sh
conda activate common

cd $RUN_DIR
python compute_substructures.py -fp $FILE_PATH -estart $ENTRY_START -estop $ENTRY_STOP -sd $SAVE_DIR

rm $SAVE_DIR/status/job_${ENTRY_START}_${ENTRY_STOP}.status