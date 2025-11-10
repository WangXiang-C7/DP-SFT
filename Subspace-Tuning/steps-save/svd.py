import os
import torch
import time

def perform_svd(trajectory_matrix, config):
    """
    执行SVD分解：
    1. 使用轨迹的第一个点（预训练模型参数）作为中心点
    2. 计算参数变化量：Δw = w_t - w_0
    3. 对变化量矩阵进行SVD分解
    """
    # 配置日志记录器
    logger = config.logger
    
    # 检查并修复异常值
    def check_and_fix(tensor, name):
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            logger.warning(f"警告: {name} 包含NaN/Inf值，正在修复...")
            return torch.nan_to_num(tensor, nan=0.0, posinf=1e30, neginf=-1e30)
        return tensor
    
    trajectory_matrix = check_and_fix(trajectory_matrix, "轨迹矩阵")
    
    logger.info("\n[SVD] 开始执行SVD分解...")
    logger.info(f"[SVD] 轨迹矩阵形状: {trajectory_matrix.shape}")
    logger.info(f"[SVD] 轨迹点数: {trajectory_matrix.size(0)}")
    logger.info(f"[SVD] 参数维度: {trajectory_matrix.size(1)}")
    
    # 计算轨迹矩阵的内存占用
    mem_size = trajectory_matrix.element_size() * trajectory_matrix.nelement() / (1024**3)
    logger.info(f"[SVD] 轨迹矩阵内存占用: {mem_size:.2f} GB")
    
    # ===== 使用轨迹矩阵中的第一个点作为起点 =====
    logger.info("[SVD] 提取初始点（轨迹矩阵中的第一个点）...")
    initial_point = trajectory_matrix[0].clone()
    logger.info(f"[SVD] 初始点形状: {initial_point.shape}")
    
    # 计算相对于起点的变化量
    logger.info("[SVD] 计算参数变化量 Δw = w_t - w_0...")
    delta_trajectory = trajectory_matrix - initial_point.unsqueeze(0)
    delta_trajectory = check_and_fix(delta_trajectory, "变化量矩阵")
    
    # 打印变化量统计信息
    delta_norm = torch.norm(delta_trajectory, dim=1)
    logger.info(f"[SVD] 平均变化量范数: {delta_norm.mean().item():.4f} ± {delta_norm.std().item():.4f}")
    logger.info(f"[SVD] 最大变化量: {delta_norm.max().item():.4f}")
    logger.info(f"[SVD] 最小变化量: {delta_norm.min().item():.4f}")
    
    # 执行SVD分解
    logger.info(f"[SVD] 开始SVD分解 (use_full_svd={config.use_full_svd})...")
    svd_start = time.time()
    
    if config.use_full_svd:
        # 非紧凑型SVD分解
        logger.info("[SVD] 使用非紧凑型SVD...")
        U, S, Vt = torch.linalg.svd(delta_trajectory, full_matrices=True)
        logger.info(f"[SVD] 非紧凑型SVD结果: U.shape={U.shape}, S.shape={S.shape}, Vt.shape={Vt.shape}")
    else:
        # 紧凑型SVD分解（推荐）
        logger.info("[SVD] 使用紧凑型SVD...")
        U, S, Vt = torch.linalg.svd(delta_trajectory, full_matrices=False)
        logger.info(f"[SVD] 紧凑型SVD结果: U.shape={U.shape}, S.shape={S.shape}, Vt.shape={Vt.shape}")
    
    svd_time = time.time() - svd_start
    logger.info(f"[SVD] SVD分解完成，耗时: {svd_time:.2f} 秒")
    
    # 提取前subspace_dim个主方向
    principal_directions = Vt[:config.subspace_dim].T
    logger.info(f"[SVD] 提取的主方向矩阵形状: {principal_directions.shape}")
    
    # 计算奇异值占比（解释方差）
    total_energy = torch.sum(S**2)
    explained_energy = torch.sum(S[:config.subspace_dim]**2)
    explained_ratio = explained_energy / total_energy
    logger.info(f"[SVD] 前{config.subspace_dim}个奇异值解释的能量占比: {explained_ratio.item()*100:.2f}%")
    
    # 转回原始设备（如果需要）
    if config.device == "cuda":
        logger.info("[SVD] 将主方向矩阵转回GPU...")
        principal_directions = principal_directions.to(torch.cuda.current_device())

    # 创建输出目录
    output_dir = os.path.join(
        config.output_dir,
        f"{config.model_name.replace('/', '-')}_{config.dataset_name}_{config.subspace_dim}"
    )
    os.makedirs(output_dir, exist_ok=True)
    # 保存SVD结果
    torch.save(principal_directions, os.path.join(output_dir, "principal_directions.pt"))
    logger.info(f"[SVD] SVD结果已保存到 {output_dir}")
    
    return principal_directions