import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import scipy.stats
import scipy.fft
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

# 设置matplotlib中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

# 设置随机种子
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ================== 第一部分：目标域数据准备 ==================

def load_and_preprocess_target(file_path, sampling_rate=32000, rpm=600):
    """
    加载和预处理目标域.mat文件

    Parameters:
    -----------
    file_path : str
        .mat文件路径
    sampling_rate : int
        采样率（目标域为32kHz）
    rpm : float
        转速（目标域约600rpm）

    Returns:
    --------
    normalized_signal : np.array
        归一化后的信号
    rpm : float
        转速
    """
    # 加载.mat文件
    mat_data = scipy.io.loadmat(file_path)

    # 动态查找数据键
    data_key = None
    for key in mat_data.keys():
        if not key.startswith('__'):  # 跳过元数据
            data_key = key
            break

    if data_key is None:
        raise ValueError(f"无法在文件 {file_path} 中找到数据键")

    # 提取信号数据并展平为1D数组
    signal = mat_data[data_key].flatten()

    # Z-score归一化
    normalized_signal = (signal - np.mean(signal)) / np.std(signal)

    return normalized_signal, rpm


def calculate_fault_frequencies_target(rpm=600):
    """
    计算目标域的理论故障特征频率

    Parameters:
    -----------
    rpm : float
        转速（转/分钟）

    Returns:
    --------
    fault_freqs : dict
        包含各种故障特征频率的字典
    """
    # SKF 6205轴承参数（与源域相同）
    N_BALLS = 9
    D_BALL = 0.3126
    D_PITCH = 1.537
    ALPHA_CONTACT = 0

    # 转频
    fr = rpm / 60.0

    # 接触角转换为弧度
    alpha_rad = np.deg2rad(ALPHA_CONTACT)

    # 外圈故障频率 (BPFO)
    f_bpfo = (N_BALLS / 2) * fr * (1 - (D_BALL / D_PITCH) * np.cos(alpha_rad))

    # 内圈故障频率 (BPFI)
    f_bpfi = (N_BALLS / 2) * fr * (1 + (D_BALL / D_PITCH) * np.cos(alpha_rad))

    # 滚动体故障频率 (BSF)
    f_bsf = (D_PITCH / (2 * D_BALL)) * fr * (1 - ((D_BALL / D_PITCH) * np.cos(alpha_rad)) ** 2)

    return {
        'fr': fr,
        'BPFO': f_bpfo,
        'BPFI': f_bpfi,
        'BSF': f_bsf
    }


def extract_features_target(signal, rpm=600, sampling_rate=32000):
    """
    为目标域提取所有特征（复用第一问的特征提取逻辑）

    Parameters:
    -----------
    signal : np.array
        输入信号
    rpm : float
        转速
    sampling_rate : int
        采样率

    Returns:
    --------
    features : dict
        所有特征的字典
    """
    # 时域特征
    rms = np.sqrt(np.mean(signal ** 2))
    kurtosis = scipy.stats.kurtosis(signal)
    skewness = scipy.stats.skew(signal)
    peak = np.max(np.abs(signal))
    clearance_factor = peak / (np.mean(np.sqrt(np.abs(signal))) ** 2)
    impulse_factor = peak / np.mean(np.abs(signal))
    shape_factor = rms / np.mean(np.abs(signal))

    # 频域特征
    fault_freqs = calculate_fault_frequencies_target(rpm)

    # 计算包络谱
    analytical_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytical_signal)
    fft_envelope = scipy.fft.fft(envelope)
    fft_magnitude = np.abs(fft_envelope)
    n_samples = len(envelope)
    freq_axis = scipy.fft.fftfreq(n_samples, 1 / sampling_rate)[:n_samples // 2]
    fft_magnitude = fft_magnitude[:n_samples // 2]

    # 定义搜索窗口
    window_hz = 5

    # 提取故障频率幅值
    bpfo_amp = 0
    for harmonic in [1, 2, 3]:
        target_freq = fault_freqs['BPFO'] * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bpfo_amp = max(bpfo_amp, np.max(fft_magnitude[idx_range]))

    bpfi_amp = 0
    for harmonic in [1, 2, 3]:
        target_freq = fault_freqs['BPFI'] * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bpfi_amp = max(bpfi_amp, np.max(fft_magnitude[idx_range]))

    bsf_amp = 0
    for harmonic in [1, 2, 3]:
        target_freq = fault_freqs['BSF'] * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bsf_amp = max(bsf_amp, np.max(fft_magnitude[idx_range]))

    fr_amp = 0
    target_freq = fault_freqs['fr']
    idx_range = np.where((freq_axis >= target_freq - window_hz) &
                         (freq_axis <= target_freq + window_hz))[0]
    if len(idx_range) > 0:
        fr_amp = np.max(fft_magnitude[idx_range])

    # 频谱质心
    spectral_centroid = np.sum(freq_axis * fft_magnitude) / np.sum(fft_magnitude)

    # 时频域特征（小波包）
    wavelet = 'db1'
    max_level = 3
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)
    nodes = [node.path for node in wp.get_level(max_level, 'natural')]

    # 计算每个节点的能量
    energies = []
    for node_path in nodes:
        node = wp[node_path]
        coeffs = node.data
        energy = np.sum(coeffs ** 2)
        energies.append(energy)

    # 返回所有特征
    features = {
        'RMS': rms,
        'Kurtosis': kurtosis,
        'Skewness': skewness,
        'Peak': peak,
        'Clearance_Factor': clearance_factor,
        'Impulse_Factor': impulse_factor,
        'Shape_Factor': shape_factor,
        'BPFO_Amplitude': bpfo_amp,
        'BPFI_Amplitude': bpfi_amp,
        'BSF_Amplitude': bsf_amp,
        'Fr_Amplitude': fr_amp,
        'Spectral_Centroid': spectral_centroid,
        'WPT_E1': energies[0],
        'WPT_E2': energies[1],
        'WPT_E3': energies[2],
        'WPT_E4': energies[3],
        'WPT_E5': energies[4],
        'WPT_E6': energies[5],
        'WPT_E7': energies[6],
        'WPT_E8': energies[7]
    }

    return features


def process_target_domain():
    """
    处理目标域数据集

    Returns:
    --------
    target_features : pd.DataFrame
        目标域特征数据框
    """
    target_dir = '数据集/目标域数据集/'
    target_files = [f'{chr(65 + i)}.mat' for i in range(16)]  # A.mat to P.mat

    target_data = []

    print("处理目标域数据...")
    for filename in tqdm(target_files):
        file_path = os.path.join(target_dir, filename)
        try:
            # 加载目标域信号
            signal, rpm = load_and_preprocess_target(file_path,
                                                     sampling_rate=32000,
                                                     rpm=600)

            # 提取特征
            features = extract_features_target(signal, rpm=600, sampling_rate=32000)

            # 添加文件名
            features['Filename'] = filename
            features['RPM'] = rpm

            target_data.append(features)

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue

    # 创建DataFrame
    target_df = pd.DataFrame(target_data)

    # 重新排列列的顺序以匹配源域
    feature_columns = ['RMS', 'Kurtosis', 'Skewness', 'Peak', 'Clearance_Factor',
                       'Impulse_Factor', 'Shape_Factor', 'BPFO_Amplitude',
                       'BPFI_Amplitude', 'BSF_Amplitude', 'Fr_Amplitude',
                       'Spectral_Centroid', 'WPT_E1', 'WPT_E2', 'WPT_E3',
                       'WPT_E4', 'WPT_E5', 'WPT_E6', 'WPT_E7', 'WPT_E8']

    target_features = target_df[feature_columns + ['Filename', 'RPM']]

    return target_features


# ================== 第二部分：DANN模型实现 ==================

class GradientReversalFunction(Function):
    """梯度反转层的实现"""

    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_val
        return output, None


class GradientReversalLayer(nn.Module):
    """梯度反转层封装"""

    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)


class DANN(nn.Module):
    """领域对抗神经网络模型"""

    def __init__(self, input_dim=20, num_classes=4, lambda_val=1.0):
        super(DANN, self).__init__()

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 标签预测器
        self.label_predictor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

        # 梯度反转层
        self.grl = GradientReversalLayer(lambda_val)

        # 领域判别器
        self.domain_discriminator = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # 提取特征
        features = self.feature_extractor(x)

        # 标签预测
        class_logits = self.label_predictor(features)

        # 领域判别（通过GRL）
        reversed_features = self.grl(features)
        domain_logits = self.domain_discriminator(reversed_features)

        return class_logits, domain_logits, features


# ================== 第三部分：数据加载与训练 ==================

def load_and_prepare_data():
    """
    加载并准备源域和目标域数据

    Returns:
    --------
    数据和标签的元组
    """
    print("加载源域数据...")
    # 加载源域数据
    source_df = pd.read_csv('bearing_features.csv')

    # 选择特征列
    feature_columns = ['RMS', 'Kurtosis', 'Skewness', 'Peak', 'Clearance_Factor',
                       'Impulse_Factor', 'Shape_Factor', 'BPFO_Amplitude',
                       'BPFI_Amplitude', 'BSF_Amplitude', 'Fr_Amplitude',
                       'Spectral_Centroid', 'WPT_E1', 'WPT_E2', 'WPT_E3',
                       'WPT_E4', 'WPT_E5', 'WPT_E6', 'WPT_E7', 'WPT_E8']

    X_source = source_df[feature_columns].values
    y_source = source_df['Label'].values

    # 处理目标域数据
    print("处理目标域数据...")
    target_df = process_target_domain()
    X_target = target_df[feature_columns].values

    # 保存目标域的原始特征
    np.save('target_features.npy', X_target)
    print("目标域特征已成功保存到 target_features.npy")

    # 创建领域标签
    domain_source = np.zeros(len(X_source))  # 源域为0
    domain_target = np.ones(len(X_target))  # 目标域为1

    # 合并数据进行标准化
    scaler = StandardScaler()
    X_combined = np.vstack([X_source, X_target])
    X_combined_scaled = scaler.fit_transform(X_combined)

    # 分离标准化后的数据
    X_source_scaled = X_combined_scaled[:len(X_source)]
    X_target_scaled = X_combined_scaled[len(X_source):]

    # 编码标签
    label_encoder = LabelEncoder()
    y_source_encoded = label_encoder.fit_transform(y_source)

    return (X_source_scaled, y_source_encoded, domain_source,
            X_target_scaled, domain_target,
            target_df['Filename'].values, label_encoder, scaler)


def train_dann(model, source_data, target_data, epochs=200, batch_size=32, lr=0.001):
    """
    训练DANN模型

    Parameters:
    -----------
    model : DANN
        DANN模型
    source_data : tuple
        源域数据(X, y, domain)
    target_data : tuple
        目标域数据(X, domain)
    epochs : int
        训练轮数
    batch_size : int
        批大小
    lr : float
        学习率

    Returns:
    --------
    train_losses : list
        训练损失历史
    """
    X_source, y_source, domain_source = source_data
    X_target, domain_target = target_data

    # 转换为张量
    X_source = torch.FloatTensor(X_source)
    y_source = torch.LongTensor(y_source)
    domain_source = torch.FloatTensor(domain_source)
    X_target = torch.FloatTensor(X_target)
    domain_target = torch.FloatTensor(domain_target)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 损失函数
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.BCEWithLogitsLoss()

    # 训练历史
    train_losses = []

    n_source = len(X_source)
    n_target = len(X_target)
    n_batches = max(n_source // batch_size, n_target // batch_size)

    print("开始训练DANN模型...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_class_loss = 0
        epoch_domain_loss = 0
        correct = 0
        total = 0

        for batch_idx in range(n_batches):
            # 采样源域数据
            source_idx = np.random.choice(n_source, batch_size, replace=True)
            batch_X_source = X_source[source_idx]
            batch_y_source = y_source[source_idx]
            batch_domain_source = domain_source[source_idx]

            # 采样目标域数据
            target_idx = np.random.choice(n_target, batch_size, replace=True)
            batch_X_target = X_target[target_idx]
            batch_domain_target = domain_target[target_idx]

            # 合并批次
            batch_X = torch.cat([batch_X_source, batch_X_target], dim=0)
            batch_domain = torch.cat([batch_domain_source, batch_domain_target], dim=0)

            # 前向传播
            class_logits, domain_logits, _ = model(batch_X)

            # 计算损失
            # 标签损失（仅源域）
            class_loss = class_criterion(class_logits[:batch_size], batch_y_source)

            # 领域损失（所有样本）
            domain_loss = domain_criterion(domain_logits.squeeze(), batch_domain)

            # 总损失
            lambda_p = float(epoch) / epochs  # 逐渐增加领域损失的权重
            loss = class_loss + lambda_p * domain_loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            epoch_loss += loss.item()
            epoch_class_loss += class_loss.item()
            epoch_domain_loss += domain_loss.item()

            # 计算准确率
            _, predicted = torch.max(class_logits[:batch_size], 1)
            total += batch_y_source.size(0)
            correct += (predicted == batch_y_source).sum().item()

        # 记录平均损失
        avg_loss = epoch_loss / n_batches
        avg_class_loss = epoch_class_loss / n_batches
        avg_domain_loss = epoch_domain_loss / n_batches
        accuracy = 100 * correct / total

        train_losses.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Total Loss: {avg_loss:.4f}, '
                  f'Class Loss: {avg_class_loss:.4f}, '
                  f'Domain Loss: {avg_domain_loss:.4f}, '
                  f'Source Acc: {accuracy:.2f}%')

    return train_losses


# ================== 第四部分：可视化与分析 ==================

def visualize_domain_adaptation(model, X_source, y_source, X_target,
                                label_encoder, title_prefix=""):
    """
    可视化领域适应效果

    Parameters:
    -----------
    model : DANN
        训练好的DANN模型
    X_source : np.array
        源域特征
    y_source : np.array
        源域标签
    X_target : np.array
        目标域特征
    label_encoder : LabelEncoder
        标签编码器
    """
    # 设置颜色映射
    colors = ['blue', 'green', 'orange', 'red']
    label_names = label_encoder.classes_
    label_names_cn = ['正常', '内圈故障', '外圈故障', '滚动体故障']

    # 创建图表
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    # 上图：迁移前（原始特征空间）
    print("生成迁移前的t-SNE可视化...")
    X_combined_before = np.vstack([X_source, X_target])

    # t-SNE降维
    tsne_before = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded_before = tsne_before.fit_transform(X_combined_before)

    # 分离源域和目标域的嵌入
    X_source_embedded_before = X_embedded_before[:len(X_source)]
    X_target_embedded_before = X_embedded_before[len(X_source):]

    # 绘制源域样本
    for i, label in enumerate(np.unique(y_source)):
        mask = y_source == label
        axes[0].scatter(X_source_embedded_before[mask, 0],
                        X_source_embedded_before[mask, 1],
                        c=colors[label], label=f'源域-{label_names_cn[label]}',
                        alpha=0.6, s=30)

    # 绘制目标域样本
    axes[0].scatter(X_target_embedded_before[:, 0],
                    X_target_embedded_before[:, 1],
                    c='black', marker='*', s=100,
                    label='目标域(未标记)', alpha=0.8)

    axes[0].set_title('迁移前：原始特征空间的t-SNE可视化', fontsize=14)
    axes[0].set_xlabel('t-SNE维度1')
    axes[0].set_ylabel('t-SNE维度2')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # 下图：迁移后（DANN共享特征空间）
    print("生成迁移后的t-SNE可视化...")
    model.eval()
    with torch.no_grad():
        # 提取共享特征
        X_source_torch = torch.FloatTensor(X_source)
        X_target_torch = torch.FloatTensor(X_target)

        _, _, features_source = model(X_source_torch)
        _, _, features_target = model(X_target_torch)

        features_source = features_source.numpy()
        features_target = features_target.numpy()

    # 合并特征
    X_combined_after = np.vstack([features_source, features_target])

    # 保存用于t-SNE可视化的共享特征
    np.save('tsne_features_after_transfer.npy', X_combined_after)
    print("共享特征已成功保存到 tsne_features_after_transfer.npy")

    # t-SNE降维
    tsne_after = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded_after = tsne_after.fit_transform(X_combined_after)

    # 分离源域和目标域的嵌入
    X_source_embedded_after = X_embedded_after[:len(X_source)]
    X_target_embedded_after = X_embedded_after[len(X_source):]

    # 绘制源域样本
    for i, label in enumerate(np.unique(y_source)):
        mask = y_source == label
        axes[1].scatter(X_source_embedded_after[mask, 0],
                        X_source_embedded_after[mask, 1],
                        c=colors[label], label=f'源域-{label_names_cn[label]}',
                        alpha=0.6, s=30)

    # 绘制目标域样本
    axes[1].scatter(X_target_embedded_after[:, 0],
                    X_target_embedded_after[:, 1],
                    c='black', marker='*', s=100,
                    label='目标域(未标记)', alpha=0.8)

    axes[1].set_title('迁移后：DANN共享特征空间的t-SNE可视化', fontsize=14)
    axes[1].set_xlabel('t-SNE维度1')
    axes[1].set_ylabel('t-SNE维度2')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('DANN_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("可视化完成，已保存为 DANN_visualization.png")


def predict_target_labels(model, X_target, target_filenames, label_encoder):
    """
    预测目标域标签

    Parameters:
    -----------
    model : DANN
        训练好的DANN模型
    X_target : np.array
        目标域特征
    target_filenames : np.array
        目标域文件名
    label_encoder : LabelEncoder
        标签编码器

    Returns:
    --------
    results_df : pd.DataFrame
        预测结果
    """
    model.eval()
    with torch.no_grad():
        X_target_torch = torch.FloatTensor(X_target)
        class_logits, _, _ = model(X_target_torch)
        _, predicted = torch.max(class_logits, 1)
        predicted_labels = predicted.numpy()

    # 解码标签
    predicted_labels_str = label_encoder.inverse_transform(predicted_labels)

    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'Filename': target_filenames,
        'Predicted_Label': predicted_labels_str
    })

    # 添加中文标签
    label_mapping = {
        'Normal': '正常',
        'Inner_Ring_Fault': '内圈故障',
        'Outer_Ring_Fault': '外圈故障',
        'Ball_Fault': '滚动体故障'
    }
    results_df['预测故障类型'] = results_df['Predicted_Label'].map(label_mapping)

    return results_df


# ================== 第五部分：主函数 ==================

def main():
    """
    主函数：执行完整的DANN迁移学习流程
    """
    print("=" * 80)
    print("领域对抗神经网络(DANN)轴承故障诊断系统")
    print("=" * 80)

    # 1. 加载和准备数据
    (X_source, y_source, domain_source,
     X_target, domain_target,
     target_filenames, label_encoder, scaler) = load_and_prepare_data()

    print(f"\n数据统计:")
    print(f"源域样本数: {len(X_source)}")
    print(f"目标域样本数: {len(X_target)}")
    print(f"特征维度: {X_source.shape[1]}")
    print(f"故障类别数: {len(np.unique(y_source))}")

    # 2. 创建DANN模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")

    model = DANN(input_dim=20, num_classes=4, lambda_val=1.0)
    model = model.to(device)

    # 3. 训练模型
    source_data = (X_source, y_source, domain_source)
    target_data = (X_target, domain_target)

    train_losses = train_dann(model, source_data, target_data,
                              epochs=300, batch_size=32, lr=0.001)

    # 保存训练好的模型权重
    torch.save(model.state_dict(), 'dann_model.pth')
    print("\nDANN模型权重已成功保存到 dann_model.pth")

    # 4. 可视化领域适应效果
    print("\n生成领域适应可视化...")
    visualize_domain_adaptation(model, X_source, y_source, X_target, label_encoder)

    # 5. 预测目标域标签
    print("\n预测目标域样本标签...")
    results_df = predict_target_labels(model, X_target, target_filenames, label_encoder)

    # 6. 显示预测结果
    print("\n" + "=" * 80)
    print("目标域故障诊断结果")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # 7. 保存结果
    results_df.to_csv('target_domain_predictions.csv', index=False)
    print("\n预测结果已保存到 target_domain_predictions.csv")

    # 8. 统计分析
    print("\n预测标签分布:")
    label_counts = results_df['预测故障类型'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(results_df)) * 100
        print(f"  {label}: {count} 个样本 ({percentage:.1f}%)")

    # 9. 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, 'b-', linewidth=2)
    plt.title('DANN训练损失曲线', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\n" + "=" * 80)
    print("DANN迁移学习诊断完成！")
    print("已保存的关键文件:")
    print("  - dann_model.pth: 训练好的模型权重")
    print("  - target_features.npy: 目标域原始特征")
    print("  - tsne_features_after_transfer.npy: 共享特征空间")
    print("  - target_domain_predictions.csv: 预测结果")
    print("=" * 80)

    return model, results_df


if __name__ == "__main__":
    # 执行主程序
    model, results = main()