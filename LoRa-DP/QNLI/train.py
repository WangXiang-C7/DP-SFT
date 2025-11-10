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

def train_model(model, train_loader, val_loader, optimizer, scheduler, 
               device, epochs, output_dir="./output"):
    """支持DP的LoRA微调训练函数"""
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        steps = 0
        
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
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # 学习率调度
            scheduler.step()
            
            train_loss += loss.item()
            steps += 1

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
        
        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 验证评估
        val_results = evaluate_model(model, val_loader, device)
        
        # 记录验证指标
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        history['val_f1'].append(val_results['f1'])
        
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_results['loss']:.4f} | "
              f"Val Acc: {val_results['accuracy']:.4f} | "
              f"Val F1: {val_results['f1']:.4f}")
    
    return history, val_results

def evaluate_model(model, data_loader, device):
    """模型评估函数（增加F1指标）"""
    model.eval()
    # 加载多个指标
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
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
    results = {
        'accuracy': accuracy_metric.compute(
            references=all_labels, predictions=all_preds)['accuracy'],
        'f1': f1_metric.compute(
            references=all_labels, predictions=all_preds, average='binary')['f1'],
        'loss': total_loss / len(data_loader)
    }
    
    return results