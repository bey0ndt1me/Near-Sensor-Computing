import numpy as np
import torch
import torch.fft
import time
import math
import sys
from jtop import jtop  # 导入 Jetson 专用库

def get_jetson_stats(jetson):
    """
    针对你的 Orin NX 硬件优化的功耗读取
    """
    # 内存 (MB)
    mem_used = jetson.memory['RAM']['used'] / 1024.0

    # 总功耗 (W) - 从 'tot' 字典的 'power' 键读取
    total_power_raw = jetson.power.get('tot', {}).get('power', 0)
    total_power = total_power_raw / 1000.0

    # 核心总功耗 (CPU+GPU+CV) (W)
    # 你的硬件上这个轨道叫 'VDD_CPU_GPU_CV'
    core_rail = jetson.power.get('rail', {}).get('VDD_CPU_GPU_CV', {})
    core_power_raw = core_rail.get('power', 0)
    core_power = core_power_raw / 1000.0

    return mem_used, total_power, core_power

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

def run_simulation():
    RESOLUTION = 256
    BATCH_SIZE = 1
    TARGET_FPS = 90
    TARGET_TIME_PER_BATCH = 1.0 / TARGET_FPS
    h, w = RESOLUTION, RESOLUTION
    frame_count = 0

    print("--- Jetson Orin NX Real-Time Simulation ---")
    print(f"Resolution: {w}x{h} | Target FPS: {TARGET_FPS}")

    # 初始化 GPU 张量
    gradx_t = torch.randn(BATCH_SIZE, h, w, device="cuda", dtype=torch.float32)
    grady_t = torch.randn(BATCH_SIZE, h, w, device="cuda", dtype=torch.float32)
    boundary_t = torch.zeros(BATCH_SIZE, h, w, device="cuda", dtype=torch.float32)

    # 预热
    _ = poisson_reconstruct_pytorch(grady_t, gradx_t, boundary_t)
    torch.cuda.synchronize()

    # 使用 jtop 上下文管理器读取硬件信息
    with jtop() as jetson:
        try:
            while True:
                loop_start_time = time.perf_counter()

                # 主计算步骤
                _ = poisson_reconstruct_pytorch(grady_t, gradx_t, boundary_t)
                torch.cuda.synchronize() 

                processing_time = time.perf_counter() - loop_start_time

                # 帧率控制
                sleep_time = TARGET_TIME_PER_BATCH - processing_time
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # 获取硬件监控数据
                gpu_mem, pwr_tot, pwr_core = get_jetson_stats(jetson)

                actual_fps = 1.0 / (time.perf_counter() - loop_start_time)
                frame_count += BATCH_SIZE

                sys.stdout.write(
                    f"\r\033[KFrames: {frame_count:<6} | "
                    f"FPS: {actual_fps:>5.2f} | "
                    f"Proc: {processing_time*1000:>6.2f}ms | "
                    f"Mem: {gpu_mem:>7.2f}MB | "
                    f"Core(CPU+GPU): {pwr_core:>5.2f}W | " # 改为 Core 功耗
                    f"TOT: {pwr_tot:>5.2f}W"
                )
                sys.stdout.flush()

        except KeyboardInterrupt:
            print("\n停止运行。")

if __name__ == "__main__":
    run_simulation()

