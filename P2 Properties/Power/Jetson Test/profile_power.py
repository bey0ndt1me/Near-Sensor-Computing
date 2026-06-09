import numpy as np
import torch
import torch.fft
import time
import math
import sys
import os
import glob
import threading
import csv

# ==========================================
# 1. 硬件底层节点发现函数 (绕过 jtop，实现 1000Hz 采样)
# ==========================================
def find_vdd_in_sysfs():
    """自动寻找 Jetson Orin NX 的 VDD_IN (系统总功耗) sysfs 节点"""
    hwmon_dirs = glob.glob('/sys/class/hwmon/hwmon*')
    for hwmon in hwmon_dirs:
        # 遍历该目录下的所有 label 文件
        label_files = glob.glob(os.path.join(hwmon, 'in*_label'))
        for label_file in label_files:
            with open(label_file, 'r') as f:
                label = f.read().strip()
            if 'VDD_IN' in label.upper():
                # 找到了 VDD_IN，对应的输入节点是 inX_input (mV) 和 currX_input (mA)
                base_name = os.path.basename(label_file).replace('_label', '')
                vol_path = os.path.join(hwmon, f"{base_name}_input")
                curr_path = os.path.join(hwmon, f"curr{base_name.replace('in', '')}_input")
                if os.path.exists(vol_path) and os.path.exists(curr_path):
                    return vol_path, curr_path
    
    # 备用方案：有时 Jetson 会直接暴露 powerX_input (微瓦 uW)
    for hwmon in hwmon_dirs:
        name_file = os.path.join(hwmon, 'name')
        if os.path.exists(name_file):
            with open(name_file, 'r') as f:
                name = f.read().strip()
            if 'ina3221' in name:
                power_path = os.path.join(hwmon, 'power1_input') # 通常 1 是 VDD_IN
                if os.path.exists(power_path):
                    return None, power_path
                    
    raise RuntimeError("无法在 /sys/class/hwmon/ 中找到 VDD_IN 功耗节点！")

# 全局变量，用于线程间通信
power_data_log =[]
current_phase = "Init"
is_running = True

# ==========================================
# 2. 高频功耗监控线程
# ==========================================
def power_monitor_thread(vol_path, curr_path, power_path):
    global current_phase, is_running, power_data_log
    
    while is_running:
        try:
            timestamp = time.perf_counter()
            power_w = 0.0
            
            # 读取底层文件计算功耗
            if power_path:
                with open(power_path, 'r') as f:
                    power_w = float(f.read().strip()) / 1_000_000.0 # uW to W
            else:
                with open(vol_path, 'r') as fv, open(curr_path, 'r') as fc:
                    vol_mv = float(fv.read().strip())
                    curr_ma = float(fc.read().strip())
                    power_w = (vol_mv * curr_ma) / 1_000_000.0 # mV * mA = uW -> W
                    
            power_data_log.append((timestamp, power_w, current_phase))
            
            # 休眠 1ms，防止把 CPU 跑满，实现约 1000Hz 采样率
            time.sleep(0.001) 
        except Exception as e:
            pass

# --- 保持原来的数学计算函数不变 ---
def dst_torch(x, type=2, n=None, axis=-1, norm=None):
    x = torch.as_tensor(x, dtype=torch.float32)
    if n is not None:
        x = torch.nn.functional.pad(x, (0, max(0, n - x.shape[axis])))
        x = x.narrow(axis, 0, n)
    N = x.shape[axis]
    if type == 2:
        x = torch.cat([x, -x.flip([axis])], dim=axis)
        result = torch.fft.fft(x, dim=axis).imag.index_select(axis, torch.arange(1, N + 1, device=x.device))
        if norm == "ortho":
            result.mul_(math.sqrt(2 / N))
            result[..., 0].mul_(math.sqrt(2))
    return result

def idst_torch(input_tensor, norm=None, axis=-1):
    n = input_tensor.shape[axis]
    if norm == 'ortho':
        input_tensor = input_tensor * math.sqrt(2 / (n + 1))
    else:
        input_tensor = input_tensor * 2
    extended_shape = list(input_tensor.shape)
    extended_shape[axis] = 2 * (n + 1)
    extended = torch.zeros(extended_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    slices = [slice(None)] * input_tensor.dim()
    slices[axis] = slice(1, n + 1)
    extended[tuple(slices)] = -input_tensor
    slices[axis] = slice(n + 2, 2 * n + 2)
    extended[tuple(slices)] = input_tensor.flip([axis])
    ifft_result = torch.fft.ifft(extended, dim=axis)
    slices[axis] = slice(1, n + 1)
    return ifft_result[tuple(slices)].imag

def poisson_reconstruct_pytorch(grady, gradx, boundarysrc):
    if boundarysrc is None:
        boundarysrc = torch.zeros_like(grady)
    gyy = grady[:, 1:, :-1] - grady[:, :-1, :-1]
    gxx = gradx[:, :-1, 1:] - gradx[:, :-1, :-1]
    f = torch.zeros_like(boundarysrc)
    f[:, :-1, 1:] += gxx
    f[:, 1:, :-1] += gyy
    boundary = boundarysrc.clone()
    boundary[:, 1:-1, 1:-1] = 0
    f_bp = -4 * boundary[:, 1:-1, 1:-1] + boundary[:, 1:-1, 2:] + boundary[:, 1:-1, :-2] + boundary[:, 2:, 1:-1] + boundary[:, :-2, 1:-1]
    f = f[:, 1:-1, 1:-1] - f_bp
    tt = dst_torch(f, norm='ortho', axis=-1)
    fsin = dst_torch(tt.transpose(-1, -2), norm='ortho', axis=-1).transpose(-1, -2)
    h, w = f.shape[1], f.shape[2]
    y, x = np.ogrid[1:h+1, 1:w+1]
    denom = (2 * np.cos(math.pi * x / (w+1)) - 2) + (2 * np.cos(math.pi * y / (h+1)) - 2)
    denom = torch.tensor(denom, dtype=f.dtype, device=f.device).unsqueeze(0)
    f = fsin / denom
    tt = idst_torch(f, norm='ortho', axis=-1)
    img_tt = idst_torch(tt.transpose(-1, -2), norm='ortho', axis=-1).transpose(-1, -2)
    result = boundary
    result[:, 1:-1, 1:-1] += img_tt
    return result

# ==========================================
# 3. 主仿真逻辑 (人为注入延时的时间拉伸法)
# ==========================================
def run_simulation():
    global current_phase, is_running
    
    RESOLUTION = 256
    BATCH_SIZE = 1
    h, w = RESOLUTION, RESOLUTION

    print("--- 启动高频功耗抓取模式 (Sustained Workload) ---")
    
    try:
        vol_path, curr_path = find_vdd_in_sysfs()
        power_path = None
    except ValueError:
        vol_path, curr_path, power_path = None, None, find_vdd_in_sysfs()[1]
        
    monitor_thread = threading.Thread(target=power_monitor_thread, args=(vol_path, curr_path, power_path))
    monitor_thread.start()

    current_phase = "Idle"
    gradx_cpu = torch.randn(BATCH_SIZE, h, w, dtype=torch.float32)
    grady_cpu = torch.randn(BATCH_SIZE, h, w, dtype=torch.float32)
    boundary_cpu = torch.zeros(BATCH_SIZE, h, w, dtype=torch.float32)
    time.sleep(2.0) # 稳定基础功耗

    # 预热
    current_phase = "Warmup"
    _ = poisson_reconstruct_pytorch(grady_cpu.to("cuda"), gradx_cpu.to("cuda"), boundary_cpu.to("cuda"))
    torch.cuda.synchronize()
    current_phase = "Idle"
    time.sleep(1.0) 

    # 我们只抓取 1 个完美的周期，但拉长它的阶段
    print("开始抓取 1 帧超长延展波形...")
    
    # --- 阶段 1: 数据搬运开销 (持续轰炸 150ms) ---
    current_phase = "H2D_Transfer"
    end_time = time.perf_counter() + 0.15 
    while time.perf_counter() < end_time:
        # 疯狂往 GPU 里塞数据，填满 PCIe 带宽
        gradx_t = gradx_cpu.to("cuda", non_blocking=False)
        grady_t = grady_cpu.to("cuda", non_blocking=False)
        boundary_t = boundary_cpu.to("cuda", non_blocking=False)
        torch.cuda.synchronize()
        
    # --- 阶段 2: 纯 GPU 计算开销 (持续轰炸 200ms) ---
    current_phase = "GPU_Compute"
    end_time = time.perf_counter() + 0.20
    while time.perf_counter() < end_time:
        # 疯狂让 GPU 算数学题，满载 CUDA Core
        result_t = poisson_reconstruct_pytorch(grady_t, gradx_t, boundary_t)
        torch.cuda.synchronize()
        
    # --- 阶段 3: 写回 CPU 开销 (持续轰炸 150ms) ---
    current_phase = "D2H_Transfer"
    end_time = time.perf_counter() + 0.15
    while time.perf_counter() < end_time:
        # 疯狂把数据拉回 CPU
        result_cpu = result_t.to("cpu", non_blocking=False)
        torch.cuda.synchronize()

    # --- 阶段 4: 空闲 ---
    current_phase = "Idle"
    time.sleep(1.0) 

    is_running = False
    monitor_thread.join()

    with open('power_profile_sustained.csv', 'w', newline='') as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(['Time_s', 'Power_W', 'Phase'])
        start_time = power_data_log[0][0]
        for row in power_data_log:
            writer.writerow([row[0] - start_time, row[1], row[2]])
    print("抓取完成！已保存为 power_profile_sustained.csv")

if __name__ == "__main__":
    run_simulation()


""" Note: To successfully capture the high-frequency power spikes of individual execution phases on the GPU using software polling, 
artificial delays were inserted between the host-to-device transfer, compute kernel execution, and device-to-host transfer. 
The time axis is scaled purely for phase demarcation and visualization purposes. """