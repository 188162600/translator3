# Training Command for WMT17 German-English Translation Model

This guide details the command used to train a German-to-English translation model using Fairseq. The model utilizes the Transformer architecture adapted for WMT English to German translation, with specific training parameters.

## Command

Run the following command from your terminal to start the training process:

```bash
fairseq-train \
    --user-dir transformer \
    C:/translator3/data-bin/wmt17_de_en_pretrain \
    --arch transformer_wmt_en_de --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 --activation-fn relu \
    --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
    --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0001 \
    --max-tokens 4000 --label-smoothing 0.1 \
    --save-dir checkpoints/2_wmt17_de_en_pretrain_ --log-interval 1000 \
    --keep-interval-updates -1 --save-interval-updates 0 \
    --criterion label_smoothed_cross_entropy \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric

## Train result (epoch 1 @ 32746 updates, score 26.91) for dataset pairs<5m
