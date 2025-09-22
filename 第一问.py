import os
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal
import scipy.stats
import scipy.fft
import pywt
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 全局常量 - SKF 6205轴承参数
SAMPLING_RATE = 12000  # Hz
N_BALLS = 9  # 滚动体数量
D_BALL = 0.3126  # inches, 滚动体直径
D_PITCH = 1.537  # inches, 节圆直径
ALPHA_CONTACT = 0  # degrees, 接触角

# 中文标签映射
LABEL_MAPPING = {
    'Normal': '正常',
    'Inner_Ring_Fault': '内圈故障',
    'Outer_Ring_Fault': '外圈故障',
    'Ball_Fault': '滚动体故障',
    'Unknown': '未知'
}


def load_and_preprocess(file_path):
    """
    加载和预处理.mat文件

    Parameters:
    -----------
    file_path : str
        .mat文件路径

    Returns:
    --------
    normalized_signal : np.array
        归一化后的信号
    rpm : float
        转速
    """
    # 加载.mat文件
    mat_data = scipy.io.loadmat(file_path)

    # 动态查找数据键（以_DE_time结尾）
    data_key = None
    for key in mat_data.keys():
        if key.endswith('_DE_time'):
            data_key = key
            break

    # 如果没有找到DE_time，可能是Normal数据
    if data_key is None:
        # 对于Normal数据，可能是X097_DE_time或类似格式
        for key in mat_data.keys():
            if 'DE_time' in key or 'X' in key and '_DE_time' in key:
                data_key = key
                break

        # 如果还是没有找到，尝试其他可能的键
        if data_key is None:
            for key in mat_data.keys():
                if not key.startswith('__'):  # 跳过元数据
                    data_key = key
                    break

    if data_key is None:
        raise ValueError(f"无法在文件 {file_path} 中找到数据键")

    # 提取信号数据并展平为1D数组
    signal = mat_data[data_key].flatten()

    # 提取RPM值
    rpm = 1772  # 默认值（1HP负载）
    if 'RPM' in mat_data:
        rpm_value = mat_data['RPM']
        if isinstance(rpm_value, np.ndarray):
            rpm = float(rpm_value.flatten()[0])
        else:
            rpm = float(rpm_value)
    else:
        # 从文件名中尝试提取RPM
        filename = os.path.basename(file_path)
        if 'rpm' in filename.lower():
            import re
            rpm_match = re.search(r'(\d+)rpm', filename.lower())
            if rpm_match:
                rpm = float(rpm_match.group(1))

    # Z-score归一化
    normalized_signal = (signal - np.mean(signal)) / np.std(signal)

    return normalized_signal, rpm


def get_label_from_path(file_path):
    """
    从文件路径中提取标签

    Parameters:
    -----------
    file_path : str
        文件路径

    Returns:
    --------
    label : str
        故障类型标签
    """
    # 将路径转换为小写以便匹配
    path_lower = file_path.lower()

    # 检查是否为Normal数据
    if 'normal' in path_lower or '48khz_normal_data' in path_lower or 'n_' in os.path.basename(path_lower):
        return 'Normal'
    # 检查Inner Ring故障
    elif '/ir/' in path_lower or '\\ir\\' in path_lower or 'ir' in os.path.basename(path_lower):
        return 'Inner_Ring_Fault'
    # 检查Outer Ring故障
    elif '/or/' in path_lower or '\\or\\' in path_lower or 'or' in os.path.basename(path_lower):
        return 'Outer_Ring_Fault'
    # 检查Ball故障
    elif '/b/' in path_lower or '\\b\\' in path_lower or (
            'b0' in os.path.basename(path_lower) and 'or' not in os.path.basename(path_lower).lower()
    ):
        return 'Ball_Fault'
    else:
        # 默认标签
        return 'Unknown'


def calculate_fault_frequencies(rpm):
    """
    计算理论故障特征频率

    Parameters:
    -----------
    rpm : float
        转速（转/分钟）

    Returns:
    --------
    fault_freqs : dict
        包含各种故障特征频率的字典
    """
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


def extract_time_features(signal):
    """
    提取时域特征（7个特征）

    Parameters:
    -----------
    signal : np.array
        输入信号

    Returns:
    --------
    features : dict
        时域特征字典
    """
    # RMS（均方根）
    rms = np.sqrt(np.mean(signal ** 2))

    # 峰度
    kurtosis = scipy.stats.kurtosis(signal)

    # 偏度
    skewness = scipy.stats.skew(signal)

    # 峰值
    peak = np.max(np.abs(signal))

    # 峭度因子（Clearance Factor）
    clearance_factor = peak / (np.mean(np.sqrt(np.abs(signal))) ** 2)

    # 脉冲因子（Impulse Factor）
    impulse_factor = peak / np.mean(np.abs(signal))

    # 形状因子（Shape Factor）
    shape_factor = rms / np.mean(np.abs(signal))

    return {
        'RMS': rms,
        'Kurtosis': kurtosis,
        'Skewness': skewness,
        'Peak': peak,
        'Clearance_Factor': clearance_factor,
        'Impulse_Factor': impulse_factor,
        'Shape_Factor': shape_factor
    }


def extract_frequency_features(signal, fault_freqs):
    """
    提取频域特征（5个特征）

    Parameters:
    -----------
    signal : np.array
        输入信号
    fault_freqs : dict
        故障特征频率字典

    Returns:
    --------
    features : dict
        频域特征字典
    """
    # 计算解析信号
    analytical_signal = scipy.signal.hilbert(signal)

    # 获取包络
    envelope = np.abs(analytical_signal)

    # 计算包络的FFT
    fft_envelope = scipy.fft.fft(envelope)
    fft_magnitude = np.abs(fft_envelope)

    # 定义频率轴
    n_samples = len(envelope)
    freq_axis = scipy.fft.fftfreq(n_samples, 1 / SAMPLING_RATE)[:n_samples // 2]
    fft_magnitude = fft_magnitude[:n_samples // 2]

    # 定义搜索窗口（±5 Hz）
    window_hz = 5

    # 提取BPFO及其谐波的幅值
    bpfo_amp = 0
    for harmonic in [1, 2, 3]:  # 基频和前两个谐波
        target_freq = fault_freqs['BPFO'] * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bpfo_amp = max(bpfo_amp, np.max(fft_magnitude[idx_range]))

    # 提取BPFI及其谐波的幅值
    bpfi_amp = 0
    for harmonic in [1, 2, 3]:
        target_freq = fault_freqs['BPFI'] * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bpfi_amp = max(bpfi_amp, np.max(fft_magnitude[idx_range]))

    # 提取BSF及其谐波的幅值
    bsf_amp = 0
    for harmonic in [1, 2, 3]:
        target_freq = fault_freqs['BSF'] * harmonic
        idx_range = np.where((freq_axis >= target_freq - window_hz) &
                             (freq_axis <= target_freq + window_hz))[0]
        if len(idx_range) > 0:
            bsf_amp = max(bsf_amp, np.max(fft_magnitude[idx_range]))

    # 提取转频幅值
    fr_amp = 0
    target_freq = fault_freqs['fr']
    idx_range = np.where((freq_axis >= target_freq - window_hz) &
                         (freq_axis <= target_freq + window_hz))[0]
    if len(idx_range) > 0:
        fr_amp = np.max(fft_magnitude[idx_range])

    # 计算频谱质心
    spectral_centroid = np.sum(freq_axis * fft_magnitude) / np.sum(fft_magnitude)

    return {
        'BPFO_Amplitude': bpfo_amp,
        'BPFI_Amplitude': bpfi_amp,
        'BSF_Amplitude': bsf_amp,
        'Fr_Amplitude': fr_amp,
        'Spectral_Centroid': spectral_centroid
    }


def extract_time_frequency_features(signal):
    """
    提取时频域特征（8个特征）- 使用小波包变换

    Parameters:
    -----------
    signal : np.array
        输入信号

    Returns:
    --------
    features : dict
        时频域特征字典
    """
    # 执行3级小波包变换
    wavelet = 'db1'
    max_level = 3
    wp = pywt.WaveletPacket(data=signal, wavelet=wavelet, mode='symmetric', maxlevel=max_level)

    # 获取叶节点（第3级共有2^3=8个节点）
    nodes = [node.path for node in wp.get_level(max_level, 'natural')]

    # 计算每个节点的能量
    energies = {}
    for i, node_path in enumerate(nodes, 1):
        node = wp[node_path]
        coeffs = node.data
        energy = np.sum(coeffs ** 2)
        energies[f'WPT_E{i}'] = energy

    return energies


def extract_all_features(signal, rpm, fault_freqs):
    """
    提取所有特征（主函数）

    Parameters:
    -----------
    signal : np.array
        输入信号
    rpm : float
        转速
    fault_freqs : dict
        故障特征频率

    Returns:
    --------
    features : dict
        所有特征的字典
    """
    # 提取时域特征
    time_features = extract_time_features(signal)

    # 提取频域特征
    freq_features = extract_frequency_features(signal, fault_freqs)

    # 提取时频域特征
    time_freq_features = extract_time_frequency_features(signal)

    # 合并所有特征
    all_features = {}
    all_features.update(time_features)
    all_features.update(freq_features)
    all_features.update(time_freq_features)

    return all_features


def plot_time_domain_signals(features_df):
    """
    绘制时域信号图（2x2子图）

    Parameters:
    -----------
    features_df : pd.DataFrame
        特征数据框
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('不同故障类型的代表性时域信号', fontsize=14)

    # 为每个故障类型选择一个代表性样本
    fault_types = ['Normal', 'Inner_Ring_Fault', 'Outer_Ring_Fault', 'Ball_Fault']
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for fault_type, pos in zip(fault_types, positions):
        # 找到该类型的第一个样本
        sample_files = features_df[features_df['Label'] == fault_type]['Filename'].values

        if len(sample_files) > 0:
            # 重新加载信号用于绘图
            try:
                signal, _ = load_and_preprocess(sample_files[0])
                # 只显示前2000个采样点
                signal_segment = signal[:2000]
                time_axis = np.arange(len(signal_segment)) / SAMPLING_RATE * 1000  # 转换为毫秒

                ax = axes[pos]
                ax.plot(time_axis, signal_segment, 'b-', linewidth=0.5)
                # 使用中文标签
                ax.set_title(f'{LABEL_MAPPING.get(fault_type, fault_type)}')
                ax.set_xlabel('时间 (ms)')
                ax.set_ylabel('振幅')
                ax.grid(True, alpha=0.3)
            except:
                axes[pos].text(0.5, 0.5, f'无 {LABEL_MAPPING.get(fault_type, fault_type)} 数据',
                               ha='center', va='center')
                axes[pos].set_title(f'{LABEL_MAPPING.get(fault_type, fault_type)}')

    plt.tight_layout()
    plt.show()


def plot_envelope_spectrum(features_df):
    """
    绘制内圈故障的包络谱图

    Parameters:
    -----------
    features_df : pd.DataFrame
        特征数据框
    """
    # 找到一个内圈故障样本
    ir_samples = features_df[features_df['Label'] == 'Inner_Ring_Fault']['Filename'].values

    if len(ir_samples) == 0:
        print("没有找到内圈故障样本")
        return

    # 加载信号
    signal, rpm = load_and_preprocess(ir_samples[0])

    # 计算故障频率
    fault_freqs = calculate_fault_frequencies(rpm)

    # 计算包络谱
    analytical_signal = scipy.signal.hilbert(signal)
    envelope = np.abs(analytical_signal)

    # FFT
    fft_envelope = scipy.fft.fft(envelope)
    fft_magnitude = np.abs(fft_envelope)
    n_samples = len(envelope)
    freq_axis = scipy.fft.fftfreq(n_samples, 1 / SAMPLING_RATE)[:n_samples // 2]
    fft_magnitude = fft_magnitude[:n_samples // 2]

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(freq_axis[:2000], fft_magnitude[:2000], 'b-', linewidth=0.8, label='包络谱')

    # 标记BPFI及其谐波
    colors = ['r', 'orange', 'yellow', 'green']
    for i in range(1, 5):
        freq = fault_freqs['BPFI'] * i
        if freq < freq_axis[2000]:
            plt.axvline(x=freq, color=colors[i - 1], linestyle='--', alpha=0.7,
                        label=f'内圈故障频率 x{i} ({freq:.1f} Hz)')

    plt.xlabel('频率 (Hz)')
    plt.ylabel('幅度')
    plt.title(f'内圈故障样本 (转速={rpm:.0f} RPM)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 500)  # 限制显示范围到500Hz
    plt.tight_layout()
    plt.show()


def main():
    """
    主函数：协调整个处理流程
    """
    print("=" * 60)
    print("轴承故障特征提取程序")
    print("=" * 60)

    # 指定数据根目录
    # 注意：根据实际数据位置调整路径
    root_dirs = [
        '数据集/源域数据集/12kHz_DE_data/',
        '数据集/源域数据集/48kHz_Normal_data/'  # Normal数据在这里
    ]

    # 初始化特征列表
    all_features_list = []

    # 收集所有.mat文件
    mat_files = []
    for root_dir in root_dirs:
        if os.path.exists(root_dir):
            for root, dirs, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('.mat'):
                        mat_files.append(os.path.join(root, file))

    if len(mat_files) == 0:
        print("错误：未找到任何.mat文件。请检查数据路径设置。")
        print(f"当前搜索路径：{root_dirs}")
        return

    print(f"找到 {len(mat_files)} 个.mat文件")
    print("开始特征提取...")

    # 处理每个文件
    for file_path in tqdm(mat_files, desc="处理文件"):
        try:
            # 加载和预处理
            signal, rpm = load_and_preprocess(file_path)

            # 获取标签
            label = get_label_from_path(file_path)

            # 计算故障频率
            fault_freqs = calculate_fault_frequencies(rpm)

            # 提取所有特征
            features = extract_all_features(signal, rpm, fault_freqs)

            # 组合特征
            feature_dict = {
                'Filename': file_path,
                'Label': label,
                'RPM': rpm
            }
            feature_dict.update(features)

            # 添加到列表
            all_features_list.append(feature_dict)

        except Exception as e:
            print(f"\n处理文件 {file_path} 时出错: {e}")
            continue

    # 创建DataFrame
    features_df = pd.DataFrame(all_features_list)

    # 保存到CSV
    output_file = 'bearing_features.csv'
    features_df.to_csv(output_file, index=False)
    print(f"\n特征已保存到 {output_file}")

    # 显示统计信息（使用中文标签）
    print("\n数据统计：")
    label_counts = features_df['Label'].value_counts()
    print("故障类型分布：")
    for label, count in label_counts.items():
        chinese_label = LABEL_MAPPING.get(label, label)
        print(f"  {chinese_label}: {count} 个样本")

    print(f"\n总样本数：{len(features_df)}")
    print(f"特征维度：{len(features_df.columns) - 3} 个特征（不包括Filename, Label, RPM）")

    # 生成可视化
    print("\n生成可视化...")
    plot_time_domain_signals(features_df)
    plot_envelope_spectrum(features_df)

    print("\n处理完成！")

    return features_df


if __name__ == "__main__":
    # 执行主程序
    features_df = main()

    # 显示前几行数据
    if features_df is not None:
        print("\n特征数据预览：")
        print(features_df.head())