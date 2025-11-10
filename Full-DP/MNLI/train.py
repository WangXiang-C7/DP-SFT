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
                    accumulation_steps, output_dir="./output"):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    model.train()
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'epsilon': []}
    
    # 提前终止标志
    privacy_budget_reached = False
    
    for epoch in range(epochs):
        train_loss = 0.0
        model.train()
        
        # 训练循环
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for step, batch in enumerate(progress_bar):
            # 数据转移到设备
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            
            # 反向传播
            loss.backward()
            
            # 每累加指定的小批次后进行一次参数更新
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证评估
        val_results = evaluate_model(model, val_loader, device)
        
        # 确保正确提取数值
        val_loss = val_results.get('loss', 0.0)
        val_acc = val_results.get('accuracy', 0.0)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 获取当前隐私消耗
        current_epsilon = privacy_engine.get_epsilon(delta=target_delta)
        history['epsilon'].append(current_epsilon)
    
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"ε: {current_epsilon:.2f} (δ={target_delta})")
        
    # 获取最终隐私消耗
    final_epsilon = privacy_engine.get_epsilon(delta=target_delta)

    return history, val_acc, final_epsilon

def evaluate_model(model, data_loader, device):
    model.eval()
    metric = evaluate.load("./eval/accuracy")
    total_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for batch in data_loader:
            inputs = batch['input_ids'].to(device)
            masks = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 统一使用模型计算损失
            outputs = model(inputs, attention_mask=masks, labels=labels)
            loss = outputs.loss
            
            batch_size = len(inputs)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            preds = torch.argmax(outputs.logits, dim=1)
            metric.add_batch(predictions=preds.cpu(), references=labels.cpu())
    
    results = metric.compute()
    results['loss'] = total_loss / total_samples
    return results
