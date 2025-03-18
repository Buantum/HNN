import torch
import numpy as np
import matplotlib.pyplot as plt
from hamiltonian_nn import HNN

def generate_sine_wave(t_span=(0, 10), dt=0.1):
    t = np.arange(t_span[0], t_span[1], dt)
    q_true = np.sin(t)
    return t, q_true

class LeapfrogIntegrator:
    def __init__(self, model, dt=0.1):
        self.model = model
        self.dt = dt

    def integrate(self, initial_state, steps):
        device = next(self.model.parameters()).device
        # 添加批次维度
        q = torch.tensor([[initial_state[0]]], requires_grad=True, device=device)
        p = torch.tensor([[initial_state[1]]], requires_grad=True, device=device)
        trajectory = []
        
        for _ in range(steps):
            # 蛙跳法步骤
            # 保持梯度计算上下文
            # 使用张量运算保持梯度
            # 保持二维输入形状
            state = torch.cat([q, p], dim=-1)
            # 保持梯度计算上下文
            state = torch.cat([q, p], dim=-1).requires_grad_(True)
            gradients = self.model.hamilton_equations(state)
            p = p + 0.5 * self.dt * gradients[:,1].unsqueeze(1)
            q = q + self.dt * gradients[:,0].unsqueeze(1)
            
            state = torch.cat([q, p], dim=-1).requires_grad_(True)
            gradients = self.model.hamilton_equations(state)
            p = p + 0.5 * self.dt * gradients[:,1].unsqueeze(1)
            
            trajectory.append(q.detach().clone().cpu().numpy())
        
        return np.array(trajectory)

def visualize_time_comparison(t, q_true, q_pred):
    plt.figure(figsize=(12, 6))
    
    plt.plot(t, q_true, 'b-', label='Analytical Solution: $q=sin(t)$', alpha=0.7)
    plt.plot(t, q_pred, 'r--', label='HNN Leapfrog Integration', alpha=0.9)
    
    plt.title('Time-Displacement Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement (q)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('time_comparison.png', dpi=300)
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载训练好的模型
    model = HNN().to(device)
    model = model.to(device)
    model.load_state_dict(torch.load('pendulum_hnn.pth'))
    model.eval()
    
    # 生成对比数据
    t_span = (0, 10)
    dt = 0.001
    t, q_true = generate_sine_wave(t_span, dt)
    
    # 进行蛙跳积分
    integrator = LeapfrogIntegrator(model, dt=dt)
    q_pred = integrator.integrate(initial_state=(0.0, 1.0), steps=len(t)).squeeze()
    
    # 可视化对比
    visualize_time_comparison(t, q_true, q_pred)