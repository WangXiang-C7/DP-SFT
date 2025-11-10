import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import os
import evaluate

def set_seed(seed=42):
    """设置随机种子确保可复现性"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def train_with_dpsgd(model, train_loader, val_loader, optimizer, scheduler, 
                    device, privacy_engine, target_epsilon, target_delta, epochs,
                    output_dir="./output"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'epsilon': []}
    
    # 提前终止标志
    privacy_budget_reached = False
    
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        
        # 训练循环
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress_bar:
            # 数据转移到设备
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪和噪声添加（由Opacus自动处理）
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # 每100步检查一次隐私消耗
            if progress_bar.n % 100 == 0:
                current_epsilon = privacy_engine.get_epsilon(delta=target_delta)
                # 如果达到或超过目标隐私预算，提前终止训练
                if current_epsilon >= target_epsilon:
                    privacy_budget_reached = True
                    print(f"\nPrivacy budget reached! Current ε={current_epsilon:.2f} >= target ε={target_epsilon:.2f}")
                    break
        
        # 如果隐私预算已达到，提前终止训练
        if privacy_budget_reached:
           break
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证评估
        val_results = evaluate_model(model, val_loader, device)
        
        # 确保正确提取数值
        val_loss = val_results.get('loss', 0.0)
        val_acc = val_results.get('accuracy', 0.0)
        
        # 确保 val_acc 是数值类型
        if isinstance(val_acc, (str, np.str_)):
            try:
                val_acc = float(val_acc)
            except ValueError:
                val_acc = 0.0
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 获取当前隐私消耗
        current_epsilon = privacy_engine.get_epsilon(delta=target_delta)
        history['epsilon'].append(current_epsilon)
    
        if val_acc > best_acc:
            best_acc = val_acc
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"ε: {current_epsilon:.2f} (δ={target_delta})")
        
        # 检查是否达到隐私预算（在epoch结束时）
        if current_epsilon >= target_epsilon:
            print(f"\nPrivacy budget reached at end of epoch! Current ε={current_epsilon:.2f} >= target ε={target_epsilon:.2f}")
            privacy_budget_reached = True
            break
    
    # 获取最终隐私消耗
    final_epsilon = privacy_engine.get_epsilon(delta=target_delta)
    
    # 如果训练提前终止，打印最佳指标
    if privacy_budget_reached:
        print("Training terminated early due to privacy budget constraints.")
        print(f"Loading best model from epoch {history['val_acc'].index(best_acc)} with accuracy {best_acc:.4f}")
    return history, best_acc, final_epsilon

def evaluate_model(model, data_loader, device):
    """使用 Hugging Face 的 evaluate 库计算评估指标"""
    
    model.eval()
    metric = evaluate.load("accuracy")
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(inputs, attention_mask=masks)
            logits = outputs.logits
            
            # 计算损失
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            
            # 收集预测和标签
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    results = metric.compute(references=all_labels, predictions=all_preds)
    avg_loss = total_loss / len(data_loader)
    results['loss'] = avg_loss
    
    return results