#!/bin/bash

# run_baseframe.py
# for each csv
# for specific generator-window_size
# for each : ./output/<dataset_basename>/<generator_name>

DATA_DIR="./database/kline_processed_data"
PYTHON_SCRIPT="run_multi_gan.py"

# define generator-window_size
declare -a generators=("gru" "lstm" "transformer")
declare -a window_sizes=(5 10 15)

# FILE NAME to start_timestamp map <source_id data="0" title="run_all.sh" />
declare -A START_MAP
declare -A END_MAP
START_MAP["processed_PTA_day.csv"]=213
END_MAP["processed_PTA_day.csv"]=3120
START_MAP["processed_SP500_day.csv"]=1077
END_MAP["processed_SP500_day.csv"]=4284
END_MAP["processed_ShippingIndex_day.csv"]=332
START_MAP["processed_SSE50_day.csv"]=1172
END_MAP["processed_SSE50_day.csv"]=4270
END_MAP["processed_China10YBond_day.csv"]=2073
START_MAP["processed_Corn_day.csv"]=1760
END_MAP["processed_Corn_day.csv"]=4858
START_MAP["processed_Rubber_day.csv"]=546
END_MAP["processed_Rubber_day.csv"]=3651
START_MAP["processed_US10YTreasury_day.csv"]=1608
END_MAP["processed_US10YTreasury_day.csv"]=4938
END_MAP["processed_Lumber_day.csv"]=606
START_MAP["processed_Oil_day.csv"]=1065
END_MAP["processed_Oil_day.csv"]=4275
START_MAP["processed_Wheat_day.csv"]=51
END_MAP["processed_Wheat_day.csv"]=3057
START_MAP["processed_Bitcoin_day.csv"]=820
END_MAP["processed_Bitcoin_day.csv"]=5476
START_MAP["processed_CSI300_day.csv"]=649
END_MAP["processed_CSI300_day.csv"]=3747
END_MAP["processed_Methanol_day.csv"]=2504
START_MAP["processed_Pulp_day.csv"]=1073
END_MAP["processed_Pulp_day.csv"]=4280
START_MAP["processed_USDIndex_day.csv"]=1060
END_MAP["processed_USDIndex_day.csv"]=4284
START_MAP["processed_Soybean_day.csv"]=1087
END_MAP["processed_Soybean_day.csv"]=4297
START_MAP["processed_Rebar_day.csv"]=546
END_MAP["processed_Rebar_day.csv"]=3652
START_MAP["processed_DowJones_day.csv"]=1077
END_MAP["processed_DowJones_day.csv"]=4284

# DEFAULT_ start_timestamp <source_id data="0" title="run_all.sh" />
DEFAULT_START=31
DEFAULT_END=-1

# FOR each <source_id data="0" title="run_all.sh" />
for FILE in "$DATA_DIR"/processed_*_day.csv; do
    FILENAME=$(basename "$FILE")
    BASENAME="${FILENAME%.csv}" # e.g. : processed_PTA_day

    # if map in START_TIMESTAMP <source_id data="0" title="run_all.sh" />
    if [[ -v START_MAP["$FILENAME"] ]]; then
        START_TIMESTAMP=${START_MAP["$FILENAME"]}
        END_TIMESTAMP=${END_MAP["$FILENAME"]}
    else
        START_TIMESTAMP=$DEFAULT_START
        END_TIMESTAMP=$DEFAULT_END
    fi

    echo "Processing data file: $FILENAME"
    echo "-------------------------------------"

    # each generator-window_size
    for i in "${!generators[@]}"; do
        generator=${generators[$i]}
        window_size=${window_sizes[$i]}

        # ./output/<dataset_basename>/<generator_name>
        # e.g. : ./output/processed_PTA_day/gru
        OUTPUT_DIR_COMBINED="./output/gan/${BASENAME}/${generator}"

        mkdir -p "$OUTPUT_DIR_COMBINED"

        echo "Running with generator=$generator, window_size=$window_size, start=$START_TIMESTAMP..."
        echo "Output directory: $OUTPUT_DIR_COMBINED"

        python "$PYTHON_SCRIPT" \
            --data_path "$FILE" \
            --output_dir "$OUTPUT_DIR_COMBINED" \
            --start_timestamp "$START_TIMESTAMP" \
            --end_timestamp "$END_TIMESTAMP" \
            --generator "$generator" \
            --feature_columns 2 19\
            --window_size "$window_size" \
            --N_pairs 1 \
            --distill_epochs 0 \
            --cross_finetune_epochs 0 \

        echo "Finished run for generator=$generator, window_size=$window_size."
        echo ""

    done
    echo "-------------------------------------"
    echo "Finished processing file: $FILENAME"
    echo ""
done

echo "All tasks completed."