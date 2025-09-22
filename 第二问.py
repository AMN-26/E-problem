import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
import os
import sys
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# 设置编码环境变量，解决Windows中文路径问题
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.platform.startswith('win'):
    import locale

    locale.setlocale(locale.LC_ALL, 'C')

warnings.filterwarnings('ignore')

# 设置中文显示支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')


class BearingFaultDiagnosisSystem:
    """
    轴承故障诊断分类系统类
    """

    def __init__(self, csv_file_path='bearing_features.csv'):
        """
        初始化诊断系统

        Parameters:
        csv_file_path (str): CSV文件路径
        """
        self.csv_file_path = csv_file_path
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_models = {}
        self.results = {}

    def load_and_prepare_data(self):
        """
        加载并预处理数据
        """
        print("=" * 60)
        print("第一步: 数据加载与预处理")
        print("=" * 60)

        try:
            # 加载CSV文件
            self.data = pd.read_csv(self.csv_file_path)
            print(f"✓ 成功加载数据文件: {self.csv_file_path}")
            print(f"✓ 数据形状: {self.data.shape}")

            # 显示数据基本信息
            print(f"✓ 列名: {list(self.data.columns)}")
            print(f"✓ 标签分布:")
            print(self.data['Label'].value_counts())

        except FileNotFoundError:
            print(f"❌ 错误: 找不到文件 {self.csv_file_path}")
            return False
        except Exception as e:
            print(f"❌ 数据加载错误: {e}")
            return False

        # 分离特征和标签
        # 特征X: 除了Filename, Label, RPM之外的所有列
        feature_columns = [col for col in self.data.columns if col not in ['Filename', 'Label', 'RPM']]
        X = self.data[feature_columns]
        y = self.data['Label']

        print(f"✓ 特征数量: {X.shape[1]}")
        print(f"✓ 样本数量: {X.shape[0]}")

        # 标签编码
        y_encoded = self.label_encoder.fit_transform(y)
        self.label_names = self.label_encoder.classes_
        print(f"✓ 标签编码映射: {dict(zip(self.label_names, range(len(self.label_names))))}")

        # 数据划分 (80% 训练集, 20% 测试集)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded  # 保持各类别比例
        )

        print(f"✓ 训练集大小: {self.X_train.shape}")
        print(f"✓ 测试集大小: {self.X_test.shape}")

        # 特征标准化
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        print("✓ 特征标准化完成")
        print("✓ 数据预处理阶段完成!\n")

        return True

    def define_models_and_parameters(self):
        """
        定义模型和超参数搜索范围
        """
        print("=" * 60)
        print("第二步: 模型定义与超参数设置")
        print("=" * 60)

        # 定义模型及其超参数搜索范围
        self.models = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan']
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10]
                }
            },
            'Support Vector Machine': {
                'model': SVC(random_state=42),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'gamma': ['scale', 'auto', 0.1, 1],
                    'kernel': ['rbf', 'linear']
                }
            }
        }

        print("✓ 已定义以下模型:")
        for name in self.models.keys():
            print(f"  - {name}")
        print("✓ 模型定义完成!\n")

    def train_and_optimize_models(self):
        """
        训练模型并进行超参数优化
        """
        print("=" * 60)
        print("第三步: 模型训练与超参数优化")
        print("=" * 60)

        for name, config in self.models.items():
            print(f"\n🔄 正在训练 {name}...")

            # 使用GridSearchCV进行超参数优化
            # 注意：在Windows系统中，如果用户路径包含中文字符，使用n_jobs=1避免编码问题
            grid_search = GridSearchCV(
                estimator=config['model'],
                param_grid=config['params'],
                cv=5,  # 5折交叉验证
                scoring='f1_macro',  # 使用宏平均F1分数作为评估指标
                n_jobs=1,  # 避免Windows中文路径编码问题
                verbose=0
            )

            # 在训练集上进行训练
            grid_search.fit(self.X_train_scaled, self.y_train)

            # 保存最佳模型
            self.best_models[name] = grid_search.best_estimator_

            print(f"✓ {name} 训练完成")
            print(f"  最佳参数: {grid_search.best_params_}")
            print(f"  交叉验证最佳得分: {grid_search.best_score_:.4f}")

        print("\n✓ 所有模型训练完成!\n")

    def evaluate_models(self):
        """
        在测试集上评估所有模型
        """
        print("=" * 60)
        print("第四步: 模型性能评估")
        print("=" * 60)

        for name, model in self.best_models.items():
            print(f"\n📊 评估 {name}:")
            print("-" * 40)

            # 在测试集上进行预测
            y_pred = model.predict(self.X_test_scaled)

            # 计算性能指标
            accuracy = accuracy_score(self.y_test, y_pred)
            f1_macro = f1_score(self.y_test, y_pred, average='macro')
            f1_weighted = f1_score(self.y_test, y_pred, average='weighted')

            # 保存结果
            self.results[name] = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'y_pred': y_pred
            }

            # 打印分类报告
            print("分类报告:")
            print(classification_report(
                self.y_test, y_pred,
                target_names=self.label_names,
                digits=4
            ))

        print("✓ 模型评估完成!\n")

    def create_performance_summary(self):
        """
        创建性能汇总表格
        """
        print("=" * 60)
        print("第五步: 性能汇总")
        print("=" * 60)

        # 创建汇总DataFrame
        summary_data = []
        for name, metrics in self.results.items():
            summary_data.append({
                '模型': name,
                '准确率': f"{metrics['accuracy']:.4f}",
                '宏平均F1': f"{metrics['f1_macro']:.4f}",
                '加权平均F1': f"{metrics['f1_weighted']:.4f}"
            })

        summary_df = pd.DataFrame(summary_data)

        # 按准确率排序
        summary_df['准确率_数值'] = summary_df['准确率'].astype(float)
        summary_df = summary_df.sort_values('准确率_数值', ascending=False)
        summary_df = summary_df.drop('准确率_数值', axis=1)

        print("📋 模型性能汇总表:")
        print(summary_df.to_string(index=False))

        # 找出最佳模型
        best_model_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['accuracy'])

        print(f"\n🏆 最佳模型: {best_model_name}")
        print(f"   准确率: {self.results[best_model_name]['accuracy']:.4f}")
        print(f"   宏平均F1: {self.results[best_model_name]['f1_macro']:.4f}")

        return best_model_name, summary_df

    def plot_confusion_matrix(self, best_model_name):
        """
        绘制最佳模型的混淆矩阵

        Parameters:
        best_model_name (str): 最佳模型名称
        """
        print(f"\n📈 绘制 {best_model_name} 的混淆矩阵...")

        # 获取预测结果
        y_pred = self.results[best_model_name]['y_pred']

        # 计算混淆矩阵
        cm = confusion_matrix(self.y_test, y_pred)

        # 创建图形
        plt.figure(figsize=(10, 8))

        # 使用seaborn绘制热力图
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.label_names,
            yticklabels=self.label_names,
            cbar_kws={'label': '样本数量'}
        )

        plt.title(f'{best_model_name} - 混淆矩阵\n准确率: {self.results[best_model_name]["accuracy"]:.4f}',
                  fontsize=16, fontweight='bold')
        plt.xlabel('预测标签', fontsize=14)
        plt.ylabel('真实标签', fontsize=14)

        # 调整标签字体大小
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12, rotation=0)

        plt.tight_layout()
        plt.show()

        print("✓ 混淆矩阵绘制完成!")

    def run_complete_analysis(self):
        """
        运行完整的分析流程
        """
        print("🚀 开始轴承故障诊断分析系统")
        print("=" * 60)

        # 步骤1: 数据加载与预处理
        if not self.load_and_prepare_data():
            print("❌ 数据加载失败，程序终止")
            return

        # 步骤2: 定义模型和参数
        self.define_models_and_parameters()

        # 步骤3: 训练和优化模型
        self.train_and_optimize_models()

        # 步骤4: 评估模型
        self.evaluate_models()

        # 步骤5: 创建性能汇总和可视化
        best_model_name, summary_df = self.create_performance_summary()
        self.plot_confusion_matrix(best_model_name)

        print("\n" + "=" * 60)
        print("🎉 轴承故障诊断分析完成!")
        print("=" * 60)

        return {
            'best_model': best_model_name,
            'best_model_object': self.best_models[best_model_name],
            'results': self.results,
            'summary': summary_df,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler
        }


def main():
    """
    主函数 - 运行轴承故障诊断系统
    """
    # 创建诊断系统实例
    diagnosis_system = BearingFaultDiagnosisSystem('bearing_features.csv')

    # 运行完整分析
    results = diagnosis_system.run_complete_analysis()

    return results


if __name__ == "__main__":
    # 运行主程序
    final_results = main()