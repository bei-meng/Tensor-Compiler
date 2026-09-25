# RUN: python3 %s

import numpy as np
import onnxruntime as ort
import subprocess
import os

def verify_matmul(onnx_path, m, n, k):
    # Generate data
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    # Save to binary
    a.tofile("a.bin")
    b.tofile("b.bin")

    # Get ORT Ground Truth
    sess = ort.InferenceSession(onnx_path)
    input_names = [i.name for i in sess.get_inputs()]
    expected = sess.run(None, {input_names[0]: a, input_names[1]: b})[0]

    # Run
    driver_path = "../../../build/driver"
    stages = ["mlir", "mlir-tensor", "mlir-memref", "mlir-llvm", "llvm"]

    # Configs to compare
    M,N,K = 32,32,64
    configs = [
        {"label": "standard", "active": True},
        # {"label": f"standard+tiling({M},{K},{N})", "tiling": True, "tilesize": f"{M},{K},{N}","active": True},
        {"label": "transposed", "transpose": True, "active": True},
        {"label": "loopExchange", "exchange": True, "active": True},
        {"label": f"transposed+tiling(M_{M},N_{N},K_{K})","transpose": True,"tiling": True, "tilesize": f"{M},{N},{K}","active": True},
        {"label": f"loopExchange+tiling(M_{M},K_{K},N_{N})","exchange": True,"tiling": True, "tilesize": f"{M},{K},{N}","active": True}
        # {"label": "tiled", "tiling": "64,64,64", "transpose": False, "active": True},
        # {"label": "transposed+tiled", "tiling": "64,64,64", "transpose": True, "active": True},
    ]

    performance_results = {}

    for cfg in configs:
        if not cfg.get("active", True):
            continue

        prefix = cfg["label"]
        print(f"\n" + "="*50)
        print(f"配置: {prefix.upper()}")
        print("="*50)

        input_files_arg = "-input-files=a.bin,b.bin,bias.bin"
       
        # Build common flags list once per config
        common_flags = []
        if cfg.get("tiling",False): common_flags.append(f"-tiling")
        if cfg.get("transpose",False): common_flags.append("-transpose")
        if cfg.get("exchange",False): common_flags.append("-exchange")
        if cfg.get("tilesize",False): common_flags.append(f"-tilesize=\"{cfg.get('tilesize')}\"")
        # 

        # Capture IR for all stages
        for stage in stages:
            ext = "ll" if stage == "llvm" else "mlir"
            os.makedirs(f"matmul_{prefix}/",exist_ok=True)
            ir_filename = f"matmul_{prefix}/{stage}.{ext}"

            cmd = [driver_path, onnx_path, f"-emit={stage}", input_files_arg] + common_flags

            print(f"降级: {stage} -> {ir_filename}")

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                with open(ir_filename, "w") as f:
                    f.write(result.stdout)
            except subprocess.CalledProcessError as e:
                print(f"Error during stage{stage}: {e.stderr}")

        # Run JIT (with Valgrind)
        print(f"Launching JIT (with Cachegrind) for {prefix}")
        cache_out_file = f"matmul_cachegrind_{prefix}.out"
        valgrind_prefix = [
            "valgrind",
            "--tool=cachegrind",
            f"--cachegrind-out-file={cache_out_file}",
            # --CacheLevel=notActualSize(actualsize),Associativity,LineSize
            "--I1=32768,8,64",     # Instruction L1: 32KB, 8-way, 64B line
            "--D1=32768,8,64",     # Data L1: 32KB, 8-way, 64B line
            "--LL=4194304,16,64"   # Last Level (L2): 4MB, 16-way, 64B line
        ]

        jit_cmd = [driver_path, onnx_path, "-emit=jit", input_files_arg] + common_flags
        # jit_cmd = valgrind_prefix + [driver_path, onnx_path, "-emit=jit", input_files_arg] + common_flags
        jit_proc = subprocess.run(jit_cmd, capture_output=True, text=True)

        # Parse Math Time from output
        for line in jit_proc.stdout.split('\n'):
            if "实际计算时间:" in line:
                print(f"{line.strip()}")
                performance_results[prefix] = line.split(":")[1].strip()
            
        # Load JIT result
        if not os.path.exists("output.bin"):
            print("driver未生成输出")
            return
        
        actual = np.fromfile("output.bin", dtype=np.float32)

        if actual.size != expected.size:
            print(f"大小不匹配! 期望：{expected.size}; 实际：{actual.size}")
            return

        actual = actual.reshape(expected.shape)

        # Compare elements
        max_diff = np.max(np.abs(actual - expected))
        print("-" * 30)
        print(f"期望 (ORT): {expected.flatten()[:4]}")
        print(f"实际 (JIT): {actual.flatten()[:4]}")
        if max_diff == 0:
            print("RESULT: 100% Bit-Identical Match")
        elif max_diff < 1e-5:
            print(f"RESULT: Perfect Pass (Max Diff: {max_diff:.2e})")
        elif max_diff < 1e-3:
            print(f"结果（浮点计算带来的误差）: 可接受的误差(最大差距: {max_diff:.2e})")
        else:
            print(f"RESULT: FAILURE (Max Diff: {max_diff:.2e})")
            print("Check for logic errors or precision loss.")
        print("-" * 30)

    # Final summary
    print("总结: ")
    for label, timing in performance_results.items():
        print(f"{label}: {timing}")

M,N,K = 2048,2048,2048
shape = f"M{M}_N{N}_K{K}"
verify_matmul(f"../../onnx_files/matmul_{shape}.onnx",M,N,K)



# M,N,K = 4096,4096,4096
# standard: 7.917255e+05 ms
# transposed: 7.602615e+04 ms
# loopExchange: 5.947335e+04 ms
# transposed+tiling(M_32,N_32,K_64): 8.942295e+04 ms
# loopExchange+tiling(M_32,K_64,N_32): 7.972177e+04 ms