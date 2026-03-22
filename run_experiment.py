import os
import subprocess
import sys
import re
import json

def parse_output(stdout_str):
    bpb = None
    size = None
    
    # Matches: final_int8_zlib_roundtrip val_loss:2.8596 val_bpb:1.6841 eval_time:1874ms
    bpb_match = re.search(r'final_int8_zlib_roundtrip.*?val_bpb:([\d\.]+)', stdout_str)
    if bpb_match:
        bpb = float(bpb_match.group(1))
        
    # Matches: Total submission size int8+zlib: 11355121 bytes
    size_match = re.search(r'Total submission size int8\+zlib:\s*(\d+)', stdout_str)
    if size_match:
        size = int(size_match.group(1))
        
    return bpb, size

def run_experiment():
    experiments = [
        {"name": "0. Control", "script": "train_gpt.py", "run_id": "control_baseline"},
        {"name": "1. SwiGLU", "script": "train docs/train_gpt_swiglu.py", "run_id": "var1_swiglu"},
        {"name": "2. Recurrence", "script": "train docs/train_gpt_recurrence.py", "run_id": "var2_recurrence"},
        {"name": "3. Low-Rank", "script": "train docs/train_gpt_low_rank.py", "run_id": "var3_low_rank"},
        {"name": "4. MQA", "script": "train docs/train_gpt_mqa.py", "run_id": "var4_mqa"},
        {"name": "5. Wide Model (6L, 640D)", "script": "train_gpt.py", "run_id": "var5_wide"},
        {"name": "6. Deep & Narrow (14L, 384D)", "script": "train_gpt.py", "run_id": "var6_deep"},
        {"name": "7. Tiny MLP (12L, Mult=1)", "script": "train_gpt.py", "run_id": "var7_tinymlp"},
        {"name": "8. Untied Embeddings (6L)", "script": "train_gpt.py", "run_id": "var8_untied"},
        {"name": "9. Weak Softcap (100.0)", "script": "train_gpt.py", "run_id": "var9_softcap"},
        {"name": "10. High RoPE (500k)", "script": "train_gpt.py", "run_id": "var10_rope"}
    ]

    base_env = os.environ.copy()
    base_env["ITERATIONS"] = "20"
    base_env["TRAIN_BATCH_TOKENS"] = "8192"
    base_env["VAL_LOSS_EVERY"] = "0"
    base_env["VAL_BATCH_SIZE"] = "8192"

    results = []

    print("\nStarting experiments (20 iterations each)...\n")

    for exp in experiments:
        print(f"Running {exp['name']} ({exp['script']})...")
        env = base_env.copy()
        env["RUN_ID"] = exp["run_id"]
        
        # Override hyperparams depending on the run
        if exp["run_id"] == "var5_wide":
            env["NUM_LAYERS"] = "6"
            env["MODEL_DIM"] = "640"
            env["NUM_HEADS"] = "10"
            env["NUM_KV_HEADS"] = "2"
        elif exp["run_id"] == "var6_deep":
            env["NUM_LAYERS"] = "14"
            env["MODEL_DIM"] = "384"
            env["NUM_HEADS"] = "6"
            env["NUM_KV_HEADS"] = "2"
        elif exp["run_id"] == "var7_tinymlp":
            env["NUM_LAYERS"] = "12"
            env["MLP_MULT"] = "1"
        elif exp["run_id"] == "var8_untied":
            env["TIE_EMBEDDINGS"] = "0"
            env["NUM_LAYERS"] = "6"
        elif exp["run_id"] == "var9_softcap":
            env["LOGIT_SOFTCAP"] = "100.0"
        elif exp["run_id"] == "var10_rope":
            env["ROPE_BASE"] = "500000.0"
        
        try:
            result = subprocess.run(
                [sys.executable, exp["script"]],
                env=env,
                capture_output=True,
                text=True,
                check=True
            )
            bpb, size = parse_output(result.stdout)
            
            if bpb is not None and size is not None:
                status = "✅ Success"
            else:
                status = "⚠️ Missing Metrics"
                
            results.append({
                "variation": exp["name"],
                "val_bpb": bpb,
                "size_bytes": size,
                "status": status,
                "notes": "Completed successfully."
            })
            print(f"  -> SUCCESS! bpb: {bpb}, size: {size}")
            
        except subprocess.CalledProcessError as e:
            print(f"  -> FAILED with exit code {e.returncode}")
            # Try to grab whatever it printed before dying
            bpb, size = parse_output(e.stdout)
            
            error_preview = e.stderr.strip().split('\n')[-3:] if e.stderr else ["No stderr output"]
            
            results.append({
                "variation": exp["name"],
                "val_bpb": bpb,
                "size_bytes": size,
                "status": "❌ Failed",
                "notes": "Error: " + " | ".join(error_preview)
            })

    # Save to JSON for programatic use
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Print a markdown table ready to copy
    print("\n\n" + "="*50)
    print("🎉 ALL RUNS COMPLETE! 🎉")
    print("="*50 + "\n")
    print("Here is your markdown table for 'Testing the Modifications.md':\n")
    
    print("| Variation | val_bpb (20 iters) | Total Size int8+zlib (bytes) | Status | Notes / Observations |")
    print("|---|---|---|---|---|")
    for r in results:
        bpb_str = f"{r['val_bpb']:.4f}" if r['val_bpb'] is not None else "N/A"
        size_str = f"{r['size_bytes']:,}" if r['size_bytes'] is not None else "N/A"
        print(f"| **{r['variation']}** | {bpb_str} | {size_str} | {r['status']} | {r['notes']} |")

if __name__ == "__main__":
    run_experiment()
