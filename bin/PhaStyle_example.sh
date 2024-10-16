# Running the prokBERT PhaStyle

torchrun PhaStyle.py \
        --fastain /project/c_evolm/training_datasets/phagelifestyleds/benchmarkdb/metadata_sample1000.fasta \
        --out prediction.txt \
        --per_device_eval_batch_size 196 \
        --kmer 6 \
        --shift 1 \
        --ftmodel neuralbioinfo/PhaStyle-mini  \
        --modelclass BertForBinaryClassificationWithPooling \
        --min_length 200
