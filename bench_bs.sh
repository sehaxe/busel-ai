#!/usr/bin/env bash
# bs/GA sweep — 10 steps each
set -e
cd /home/sehaxe/busel-ai/burn_migration
BIN=./target/release/busel-burn
STEPS=10

echo "bs,ga,tok_s,ms_step,loss_last"

for BS in 2 4 8 16; do
    for GA in 1 2 5; do
        # Build YAML with specific GA
        cat > /tmp/bsweep.yaml <<YAML
model:
  d_model: 512
  n_layers: 12
  n_heads: 8
  expert_hidden: 1024
  num_experts: 1
  top_k: 1
  vocab_size: 326
  num_mtp_heads: 3
  n_patches: 32
  sct_rank: 32
  dropbp_prob: 0
  progressive_freeze: false
  ascii_curriculum: false
  chunk_growth: false
training:
  warmup_steps: 1
  learning_rate_muon: 0.002
  weight_decay: 0.1
  grad_accum_steps: ${GA}
  wsd_decay_frac: 0.1
  lotus_rank: 32
  max_steps: auto
  target_tok_per_param: 80
data:
  batch_size: 128
  chunk_size: 16384
YAML
        rm -rf checkpoints/vorobey/ 2>/dev/null
        output=$(timeout 180 $BIN vorobey --yaml /tmp/bsweep.yaml --steps $STEPS --bs $BS --cs 256 2>&1 || true)
        tok_s=$(echo "$output" | grep "tok/s:" | awk '{print $2}')
        ms_step=$(echo "$output" | grep "ms/step:" | awk '{print $2}')
        loss_last=$(echo "$output" | grep "loss last:" | awk '{print $2}')
        # Fallback to last tr line
        if [ -z "$tok_s" ]; then
            tok_s=$(echo "$output" | grep "step $STEPS" | grep -oP 'tok/s \K[0-9]+')
        fi
        echo "$BS,$GA,$tok_s,$ms_step,$loss_last"
    done
done