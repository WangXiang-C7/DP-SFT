import os
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

def perform_svd(trajectory_dir, config, model):
    """
    执行SVD分解，确保参数顺序与模型一致
    
    参数:
        trajectory_dir: 轨迹文件目录
        config: 配置对象
        model: 模型实例已训练的模型实例（用于获取参数顺序）
    """
    # 检查并修复异常值
    def check_and_fix(tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            config.logger.warning(f"警告: {name} 包含NaN/Inf值，正在修复...")
            return torch.nan_to_num(tensor, nan=0.0, posinf=1e30, neginf=-1e30)
        return tensor
    
    config.logger.info(f"[SVD] 从目录加载轨迹: {trajectory_dir}")
    
    # 1. 从模型获取原始参数顺序
    param_names = list(model.state_dict().keys())
    param_shapes = {name: model.state_dict()[name].shape for name in param_names}
    
    config.logger.info(f"[SVD] 使用模型原始参数顺序 ({len(param_names)} 个参数)")
    config.logger.debug(f"[SVD] 参数顺序: {param_names[:3]}...{param_names[-3:]}")
    
    # 加载文件索引
    index_path = os.path.join(trajectory_dir, "index.pt")
    if os.path.exists(index_path):
        snapshot_files = torch.load(index_path)
    else:
        # 向后兼容：扫描目录
        snapshot_files = sorted(
            [f for f in os.listdir(trajectory_dir) if f.endswith(".pt") and f != "index.pt"],
            key=lambda x: int(x.split(".")[0])
        )
        snapshot_files = [os.path.join(trajectory_dir, f) for f in snapshot_files]
    
    # 读取并堆叠参数
    snapshots = []
    config.logger.info(f"[SVD] 加载 {len(snapshot_files)} 个轨迹快照...")
    
    # 验证第一个快照的参数一致性
    first_data = torch.load(snapshot_files[0])
    snapshot_param_names = set(first_data["params"].keys())
    model_param_names = set(param_names)
    
    if snapshot_param_names != model_param_names:
        config.logger.error(f"[SVD] 参数名称不一致!")
        missing = model_param_names - snapshot_param_names
        extra = snapshot_param_names - model_param_names
        
        if missing:
            config.logger.error(f"[SVD] 缺失参数: {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        if extra:
            config.logger.error(f"[SVD] 多余参数: {list(extra)[:5]}{'...' if len(extra) > 5 else ''}")
        
        raise ValueError("轨迹快照与模型参数不一致")
    
    # 处理所有快照
    for file_path in tqdm(snapshot_files, desc="加载轨迹快照"):
        data = torch.load(file_path)
        
        # 按模型原始顺序展平参数
        flat_params = torch.cat([
            data["params"][name].flatten() 
            for name in param_names
        ])
        snapshots.append(flat_params)
    
    # 获取初始参数（第0步）
    initial_params = snapshots[0].clone()
    config.logger.info(f"[SVD] 初始参数向量形状: {initial_params.shape}")
    
    # 计算相对于初始参数的变化量（排除初始参数本身）
    config.logger.info("[SVD] 计算相对于初始参数的增量...")
    trajectory_deltas = []
    
    for i in tqdm(range(1, len(snapshots)), desc="计算参数增量"):
        delta = snapshots[i] - initial_params
        trajectory_deltas.append(delta)
    
    # 释放原始快照以节省内存
    del snapshots
    torch.cuda.empty_cache() if initial_params.is_cuda else None
    
    # 堆叠变化矩阵 [n-1, D]
    delta_matrix = torch.stack(trajectory_deltas)
    delta_matrix = check_and_fix(delta_matrix, "参数增量矩阵")
    
    # 释放增量列表以节省内存
    del trajectory_deltas
    torch.cuda.empty_cache() if delta_matrix.is_cuda else None
    
    # 打印矩阵信息
    config.logger.info(f"[SVD] 参数增量矩阵形状: {delta_matrix.shape}")
    
    # 计算内存占用
    mem_size = delta_matrix.element_size() * delta_matrix.nelement() / (1024**3)
    config.logger.info(f"[SVD] 增量矩阵内存占用: {mem_size:.2f} GB")
    
    # 分析增量范数
    delta_norms = torch.norm(delta_matrix, dim=1)
    config.logger.info(
        f"[SVD] 参数增量范数统计: "
        f"最小={delta_norms.min().item():.4e}, "
        f"平均={delta_norms.mean().item():.4e}, "
        f"最大={delta_norms.max().item():.4e}"
    )
    
    # 转移数据到CPU（使用detach()切断梯度计算）
    config.logger.info("[SVD] 将增量矩阵转移到CPU...")
    if delta_matrix.is_cuda:
        delta_matrix_cpu = delta_matrix.detach().cpu()
    else:
        delta_matrix_cpu = delta_matrix.detach().clone()
    
    # 释放GPU内存
    del delta_matrix
    torch.cuda.empty_cache()
    
    # 执行SVD分解
    config.logger.info(f"[SVD] 开始增量矩阵的SVD分解...")
    svd_start = time.time()
    
    # 紧凑型SVD分解
    config.logger.info("[SVD] 使用紧凑型SVD...")
    U, S, Vt = torch.linalg.svd(delta_matrix_cpu, full_matrices=False)
    config.logger.info(f"[SVD] SVD结果: U.shape={U.shape}, S.shape={S.shape}, Vt.shape={Vt.shape}")
    
    svd_time = time.time() - svd_start
    config.logger.info(f"[SVD] SVD分解完成，耗时: {svd_time:.2f} 秒")
    
    # 分析奇异值能量分布
    total_energy = torch.sum(S**2)
    config.logger.info("[SVD] 奇异值能量分布:")
    cumulative_energy = 0.0
    for i in range(len(S)):
        energy = S[i]**2
        energy_ratio = energy / total_energy
        cumulative_energy += energy_ratio
        config.logger.info(
            f"  S[{i}] = {S[i].item():.4e} "
            f"能量占比: {energy_ratio.item()*100:.2f}% "
            f"累计: {cumulative_energy.item()*100:.2f}%"
        )
    
    # 保存奇异值谱图
    if config.save_svd_plots:
        plot_dir = os.path.join(trajectory_dir, "svd_plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, "svd_spectrum.png")
        
        plt.figure(figsize=(10, 6))
        plt.plot(S.numpy()[:50], 'o-', markersize=4)
        plt.xlabel('奇异值索引')
        plt.ylabel('奇异值大小')
        plt.title('参数增量奇异值谱')
        plt.grid(True, alpha=0.3)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        config.logger.info(f"[SVD] 奇异值谱图已保存至: {plot_path}")
    
    # 提取前stage2_subspace_dim个主方向
    config.logger.info(f"[SVD] 提取前{config.stage2_subspace_dim}个主方向...")
    principal_directions = Vt[:config.stage2_subspace_dim].T
    config.logger.info(f"[SVD] 提取的主方向矩阵形状: {principal_directions.shape}")
    
    # 计算主方向覆盖率
    top_k_energy = torch.sum(S[:config.stage2_subspace_dim]**2) / total_energy
    config.logger.info(
        f"[SVD] 前 {config.stage2_subspace_dim} 个主方向能量覆盖率: "
        f"{top_k_energy.item()*100:.2f}%"
    )
    
    # 保存参数映射信息
    mapping_path = os.path.join(trajectory_dir, "param_mapping.pt")
    name_to_index = {}
    start_idx = 0
    for name in param_names:
        num_params = param_shapes[name].numel()
        name_to_index[name] = (start_idx, start_idx + num_params)
        start_idx += num_params
    
    torch.save({
        "param_names": param_names,
        "param_shapes": param_shapes,
        "name_to_index": name_to_index,
        "principal_directions_shape": principal_directions.shape
    }, mapping_path)
    
    config.logger.info(f"[SVD] 参数映射信息已保存: {mapping_path}")
    
    # 转回GPU（如果需要）
    result = principal_directions
    if config.device == "cuda":
        config.logger.info("[SVD] 将主方向矩阵转回GPU...")
        result = result.to(torch.cuda.current_device())
    
    return result