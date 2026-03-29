import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from vci.model.model import load_VCI
from vci.model.classifier import load_classifier
from vci.dataset.dataset import load_dataset_splits
from vci.utils.data_utils import data_collate, move_tensors

# 工具函数：张量转图像
def tensor_to_image(tensor):
    """将张量转换为PIL图像 (输入可能是分布对象或张量)"""
    # 如果是分布对象，取mean
    if hasattr(tensor, 'mean') and callable(tensor.mean):
        # 分布对象有mean方法
        try:
            if hasattr(tensor, 'rsample'):  # 是概率分布
                tensor = tensor.mean()  # 调用mean方法获取均值
        except:
            pass
    
    # 确保是张量
    if not isinstance(tensor, torch.Tensor):
        try:
            tensor = torch.tensor(tensor, dtype=torch.float32)
        except:
            return Image.new('RGB', (64, 64))  # 失败时返回空图
    
    # 取batch第一个样本如果有的话
    while tensor.dim() > 3:
        tensor = tensor[0]
    
    # 确保有通道维度
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0)
    
    # 转换到[0,1]范围
    tensor = tensor.detach().cpu().float()
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转为numpy并转换通道顺序
    try:
        if tensor.size(0) == 3:  # CHW格式
            img = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:  # 已经是HW或HWC
            img = (tensor.numpy() * 255).astype(np.uint8)
            if img.ndim == 2:  # 灰度图
                img = np.stack([img] * 3, axis=-1)
    except:
        return Image.new('RGB', (64, 64))  # 失败时返回空图
    
    return Image.fromarray(img)

def save_difference_map(original, counterfactual, save_path):
    """
    生成并保存残差热力图
    original: 原始图像张量 (C, H, W)
    counterfactual: 反事实图像张量 (C, H, W)
    """
    # 计算差值
    diff = torch.abs(counterfactual - original)
    # 取平均通道得到灰度热力图
    heatmap = diff.mean(dim=0).cpu().detach().numpy()
    
    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap='hot')
    plt.title('Difference Map (Model Disentanglement)')
    plt.colorbar()
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

def save_interpolation_sequence(vci_model, classifier, device, x, t_original, c, 
                                treatment_dim=1, num_steps=7, save_dir='./interpolation_results'):
    """
    生成微笑程度线性插值序列
    treatment_dim: 要插值的属性维度（针对微笑，通常是第二维）
    num_steps: 插值步数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        # 确保x和t的batch维度
        if x.dim() == 3:  # 如果只有(C,H,W)，添加batch维度
            x = x.unsqueeze(0)
        if t_original.dim() == 1:  # 如果只有(treatments,)，添加batch维度
            t_original = t_original.unsqueeze(0)
        
        # 处理协变量维度
        if isinstance(c, torch.Tensor):
            if c.dim() == 1:
                c = c.unsqueeze(0)
        elif isinstance(c, list):
            if len(c) > 0 and isinstance(c[0], torch.Tensor) and c[0].dim() == 0:
                c = [item.unsqueeze(0) for item in c]
        
        # 获取原始微笑程度 t_original[treatment_dim]
        t_start = t_original.clone()
        t_end = t_original.clone()
        t_end[:, treatment_dim] = 1.0  # 将微笑维度设为1（最高）
        
        images = []
        # 生成插值序列
        for alpha in np.linspace(0, 1, num_steps):
            t_interp = t_start * (1 - alpha) + t_end * alpha
            result = vci_model(x, t_original, c, cf_treatments=t_interp)
            cf_x = result[1]
            images.append(cf_x)
        
        # 拼接为长条图并保存
        fig, axes = plt.subplots(1, num_steps, figsize=(20, 3))
        for i, img_tensor in enumerate(images):
            img = tensor_to_image(img_tensor)
            axes[i].imshow(img)
            axes[i].set_title(f'α={i/(num_steps-1):.1f}', fontsize=10)
            axes[i].axis('off')
        
        plt.suptitle('Smile Intensity: Linear Interpolation in Latent Space', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'smile_interpolation.png'), dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✨ 插值序列已保存到: {save_dir}/smile_interpolation.png")


# ================= 配置区 =================
args = {
    "data_name": "celebA",
    "data_path": "/dev/shm/celebA-HQ",
    "label_names": "Male,Smiling",
    "dist_mode": "discriminate",
    "dist_outcomes": "bernoulli",
    "batch_size": 32,
    "device": "cuda:0",
    "omega0": 10.0, "omega1": 0.05, "omega2": 0.01,
    "hparams": {
        "outcome_emb_dim": 32, "treatment_emb_dim": 8, "covariate_emb_dim": 2,
        "defuse_steps": 3, "encoder_resolution": "64*64,32*32,16*16,8*8,4*4,1*1",
        "encoder_width": "32,64,128,256,512,1024", "encoder_depth": "3,12,12,6,3,3",
        "decoder_resolution": "1*1,4*4,8*8,16*16,32*32,64*64", "decoder_width": "1024,512,256,128,64,32",
        "decoder_depth": "3,3,6,12,12,3", "classifier_resolution": "64*64,32*32,16*16,8*8,4*4,1*1",
        "classifier_width": "32,64,128,256,512,1024", "classifier_depth": "3,12,12,6,3,9"
    },
    "num_outcomes": [3, 64, 64], "num_treatments": 4, "num_covariates": [1]
}

device = torch.device(args["device"])

# 自动检查路径
if not os.path.exists(args["data_path"]):
    print("⚠️ 内存盘数据缺失，切换至硬盘...")
    args["data_path"] = "/workspace/VCI_Project/datasets/celebA-HQ"

# ================= 加载数据与模型 =================
print("📦 正在加载测试集...")
datasets = load_dataset_splits(args["data_name"], args["data_path"], 
                               label_names=args["label_names"].split(","), sample_cf=True)
test_loader = torch.utils.data.DataLoader(
    datasets["test"], batch_size=args["batch_size"], shuffle=False,
    collate_fn=(lambda batch: data_collate(batch, nb_dims=datasets["test"].nb_dims))
)

print("🧠 正在加载 VCI 生成器和分类器...")
vci_path = "artifact/saves/celebA-HQ-test_2026.03.21_10:57:35/model_seed=None_epoch=99.pt"
clf_path = "artifact/classifier/saves/celebA-classifier_2026.03.21_10:59:07/model_seed=None_epoch=49.pt"

# 模型加载函数
def smart_load(model, path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到权重文件: {path}")
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, tuple):
        checkpoint = checkpoint[0]
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    return model

vci_model = smart_load(load_VCI(args, state_dict=None, device=device), vci_path)
classifier = smart_load(load_classifier(args, state_dict=None, device=device), clf_path)

# ================= 定性展示：微笑程度线性插值 =================
print("\n🎬 生成微笑程度线性插值序列...")
try:
    with torch.no_grad():
        # 获取第一个batch用于示例
        sample_batch = next(iter(test_loader))
        # sample_batch是: [x, t, c, cf_t, ...]
        x_sample = sample_batch[0][:1].to(device)  # 取出第一个样本
        t_sample = sample_batch[1][:1].to(device)
        c_sample = sample_batch[2]  # 协变量是list，不需要slicing
        
        # 处理协变量
        if isinstance(c_sample, list):
            c_sample = [item[:1].to(device) if isinstance(item, torch.Tensor) else item for item in c_sample]
        
        # 生成线性插值 - 将微笑维度从当前值逐渐增加到1
        os.makedirs('./experiment_results', exist_ok=True)
        
        num_steps = 7
        images = []
        for alpha in np.linspace(0, 1, num_steps):
            t_interp = t_sample.clone()
            t_interp[:, 1] = alpha  # 第二维是微笑
            result = vci_model(x_sample, t_sample, c_sample, cf_treatments=t_interp)
            cf_x = result[1]
            images.append(cf_x)
        
        # 保存插值序列可视化
        fig, axes = plt.subplots(1, num_steps, figsize=(20, 3))
        if num_steps == 1:
            axes = [axes]
        for i, img_tensor in enumerate(images):
            img = tensor_to_image(img_tensor)
            axes[i].imshow(img)
            axes[i].set_title(f'Smile α={i/(num_steps-1):.2f}', fontsize=9)
            axes[i].axis('off')
        plt.suptitle('Linear Interpolation in Smile Space', fontsize=12, y=1.02)
        plt.tight_layout()
        plt.savefig('./experiment_results/smile_interpolation.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"✨ 插值序列已保存到: ./experiment_results/smile_interpolation.png")
except Exception as e:
    print(f"⚠️ 插值演示生成失败: {str(e)}")

# ================= 定量评估：公理性检验 =================
print("\n📊 开始定量评估 (有效性 & 可逆性)...")
print(f"样本总数: {len(test_loader.dataset)}")

effectiveness_correct = 0
reversibility_mse_sum = 0.0
total_samples = 0
difference_maps_count = 0

# 创建结果目录
viz_dir = './experiment_results/difference_maps'
os.makedirs(viz_dir, exist_ok=True)

with torch.no_grad():
    for i, batch in enumerate(test_loader):
        # data_collate返回: [x, t, c, cf_t, ...]
        # c是一个列表 (来自原始数据的协变量)
        x = batch[0].to(device)       # (B, 3, 64, 64)
        t = batch[1].to(device)       # (B, 4)
        c = batch[2]                  # list of tensors
        cf_t = batch[3].to(device)    # (B, 4)
        
        # 处理协变量：将其中的tensors转移到device
        if isinstance(c, list):
            c = [item.to(device) if isinstance(item, torch.Tensor) else item for item in c]
        
        # VCI模型需要参数 (outcomes, treatments, covariates, cf_treatments)
        result = vci_model(x, t, c, cf_treatments=cf_t)
        cf_x = result[1]  # HVCIConv返回8个值，第二个是反事实输出
        
        # 评估有效性 (Effectiveness)
        # 分类器需要图像和协变量
        if isinstance(c, list) and len(c) > 0:
            c_for_clf = c[0]  # 取list中的第一个tensor
        else:
            c_for_clf = c
        preds = classifier(cf_x, c_for_clf)
        pred_labels = (torch.sigmoid(preds) > 0.5).float()
        effectiveness_correct += (pred_labels == cf_t).all(dim=1).sum().item()
        
        # 评估可逆性 (Reversibility)
        result_rev = vci_model(cf_x, cf_t, c, cf_treatments=t)
        rev_x = result_rev[1]
        mse = F.mse_loss(rev_x, x, reduction='sum').item()
        reversibility_mse_sum += mse
        
        # 保存前3个样本的差异热力图
        if difference_maps_count < 3:
            for j in range(min(2, x.size(0))):
                if difference_maps_count < 3:
                    diff_path = os.path.join(viz_dir, f'sample_{difference_maps_count:02d}_diff_map.png')
                    save_difference_map(x[j], cf_x[j], diff_path)
                    difference_maps_count += 1
        
        total_samples += x.size(0)
        
        if (i + 1) % 10 == 0:
            print(f"  进度: {total_samples} / {len(test_loader.dataset)}")

# ================= 结果汇总 =================
effectiveness_acc = effectiveness_correct / total_samples
reversibility_mse = reversibility_mse_sum / (total_samples * x.size(1) * x.size(2) * x.size(3))

print("\n" + "="*70)
print("🎉 CelebA-HQ 因果生成模型 - 公理性检验结果")
print("="*70)
print(f"\n📈 定量指标:")
print(f"   ✅ 有效性 (Effectiveness):  {effectiveness_acc * 100:.2f}%")
print(f"      └─ 说明: 反事实后，分类器对目标属性的识别准确率")
print(f"\n   🔄 可逆性 (Reversibility):  {reversibility_mse:.6f}")
print(f"      └─ 说明: 变过去再变回，原始与重构图像的归一化MSE")
print(f"\n📝 关键发现:")
print(f"   • 考虑到CelebA属性间的强相关性，本文重点考察了因果生成的两项核心公理:")
print(f"   • 有效性确保了干预的可靠性与可控性")
print(f"   • 可逆性验证了身份信息的循环一致性(Cycle Consistency)")
print(f"\n🎬 定性展示:")
print(f"   • 线性插值序列: ./experiment_results/smile_interpolation.png")
print(f"   • 残差热力图:   ./experiment_results/difference_maps/")
print(f"      └─ 若模型解耦良好，残差应仅在修改属性区域着色")
print("="*70 + "\n")
