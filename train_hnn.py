import torch

import numpy as np

import matplotlib.pyplot as plt

from hamiltonian_nn import HNN, hnn_loss


# 数据生成函数

def generate_pendulum_data(num_samples=10000, total_time=10.0, noise_std=0.01):

    """生成带时间扰动的单摆运动数据"""

    t = np.linspace(0, total_time, num_samples)

    # 添加时间扰动

    t += np.random.normal(0, noise_std, num_samples)
    

    # 单摆解析解 (q=θ, p=θ')

    A = np.random.uniform(0.5, 1.5, num_samples)
    q = A * np.sin(t)  # 角度
    p = A * np.cos(t)  # 角速度
    

    # 计算导数 (dq/dt = p, dp/dt = -q)

    dq_dt = p

    dp_dt = -q
    

    # 组合成四维数据

    return np.column_stack([q, p, dq_dt, dp_dt])


# 数据加载函数

class PendulumDataset(torch.utils.data.Dataset):

    def __init__(self, time_series, device):

        self.states = time_series[:, :2]

        self.ders = time_series[:, 2:]

        self.device = device
        

    def __len__(self):

        return len(self.states)
    

    def __getitem__(self, idx):

        return {

            'qp': torch.FloatTensor(self.states[idx][:2]).requires_grad_(True),

            'target': torch.FloatTensor(self.ders[idx][:2])

        }


# 训练函数

def train_model(num_epochs=10):

    # 设备检测

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # 初始化模型和优化器

    model = HNN().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    

    # 生成模拟数据

    time_series = generate_pendulum_data(num_samples=20000, total_time=10.0, noise_std=0.01)

    dataset = PendulumDataset(time_series, device)

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    

    # 训练循环

    for epoch in range(num_epochs):

        total_loss = 0

        for batch in loader:

            qp = batch['qp'].to(device)

            target = batch['target'].to(device)
            

            # 前向传播

            with torch.set_grad_enabled(True):

                pred = model.hamilton_equations(qp)

                loss = hnn_loss(pred, target)
            

            # 反向传播

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()
            

            total_loss += loss.item()
        

        if epoch % 10 == 0:

            print(f'Epoch {epoch}, Loss: {total_loss/len(loader):.4f}')
    

    # 保存模型

    torch.save(model.state_dict(), 'pendulum_hnn.pth')
    return model


# 可视化函数

def visualize_results(true_data, pred_data):
    plt.figure(figsize=(8,6))
    
    plt.scatter(true_data[:,0], true_data[:,1], s=1, c='blue', alpha=0.5, label='True')
    plt.scatter(pred_data[:,0], pred_data[:,1], s=1, c='red', alpha=0.5, label='Predicted')
    
    plt.title('Phase Space Comparison')
    plt.xlabel('q')
    plt.ylabel('p')
    plt.legend()
    
    # 设置坐标轴等比例
# 尝试使用 plt.axis('equal') 替代 plt.gca().set_aspect('equal')
    plt.axis('equal')
    # 统一坐标范围
    plt.xlim(-1.5, 1.5) 
    plt.ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('phase_space.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = train_model().to(device)

    # 生成测试数据
    test_numpy =generate_pendulum_data(num_samples=100, total_time=10.0, noise_std=0.01)
    test_data = torch.FloatTensor(test_numpy[:, :2]).to(device)
    test_labels = torch.FloatTensor(test_numpy[:, 2:4]).to(device)

    # 预测
    preds = model.hamilton_equations(test_data.requires_grad_(True))
    
    # 可视化对比
    visualize_results(test_numpy[:, 2:4], preds.cpu().detach().numpy())