import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

def create_privacy_comparison():
    categories = ['Traditional AI', 'HemoSparse']
    attack_rates = [62.8, 50.0]
    colors = ['#ff6b6b', '#51cf66']
    
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
    bars = ax.bar(categories, attack_rates, color=colors, width=0.6)
    
    ax.axhline(y=50, color='#333333', linestyle='--', linewidth=2)
    ax.text(0.5, 53, 'Coin Flip Random Guess Level', ha='center', va='bottom', fontsize=18, fontweight='bold', color='#333333')
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{height}%',
                ha='center', va='bottom', fontsize=36, fontweight='bold')
    
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.tick_params(axis='x', labelsize=26, labelrotation=0)
    ax.tick_params(axis='y', labelsize=20)
    
    fig.suptitle('Hacker Attack Success Rate Comparison', fontsize=36, fontweight='bold', y=0.98)
    ax.set_title('HemoSparse completely blocks privacy theft', fontsize=24, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('隐私保护对比.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print('✅ 隐私保护对比.png generated successfully!')

def create_accuracy_comparison():
    categories = ['Traditional AI', 'HemoSparse']
    accuracies = [95.59, 93.63]
    colors = ['#74c0fc', '#4dabf7']
    
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
    bars = ax.bar(categories, accuracies, color=colors, width=0.6)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{height}%',
                ha='center', va='bottom', fontsize=32, fontweight='bold')
    
    ax.set_ylim(85, 100)
    ax.set_yticks([85, 90, 95, 100])
    ax.tick_params(axis='x', labelsize=26, labelrotation=0)
    ax.tick_params(axis='y', labelsize=20)
    
    fig.suptitle('Blood Cell Diagnosis Accuracy Comparison', fontsize=36, fontweight='bold', y=0.98)
    ax.set_title('Nearly identical accuracy, clinically sufficient', fontsize=24, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('诊断准确率对比.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print('✅ 诊断准确率对比.png generated successfully!')

def create_computational_efficiency():
    categories = ['Traditional AI', 'HemoSparse']
    efficiencies = [100, 0.3]
    colors = ['#868e96', '#51cf66']
    
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300)
    
    bars = ax.barh(categories, efficiencies, color=colors, height=0.6)
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if i == 1:
            ax.text(width + 3, bar.get_y() + bar.get_height()/2.,
                    f'{width}%',
                    ha='left', va='center', fontsize=32, fontweight='bold')
        else:
            ax.text(width / 2, bar.get_y() + bar.get_height()/2.,
                    f'{width}%',
                    ha='center', va='center', fontsize=32, fontweight='bold', color='white')
    
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis='y', labelsize=26)
    ax.tick_params(axis='x', labelsize=20)
    
    fig.suptitle('AI Computational Cost Comparison', fontsize=36, fontweight='bold', y=0.98)
    ax.set_title('HemoSparse uses only 0.3% computation, saves 99.7%', fontsize=24, pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    
    plt.tight_layout()
    plt.savefig('计算量对比.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print('✅ 计算量对比.png generated successfully!')

def create_summary_infographic():
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 9), dpi=300)
    axes = [ax1, ax2, ax3]
    
    for ax in axes:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
    
    fig.suptitle('HemoSparse Core Advantages', fontsize=44, fontweight='bold', y=0.97)
    
    privacy_bg = '#d3f9d8'
    rect1 = plt.Rectangle((0.05, 0.05), 0.9, 0.9, facecolor=privacy_bg, alpha=0.6)
    ax1.add_patch(rect1)
    ax1.text(0.5, 0.75, 'PRIVACY', fontsize=52, ha='center', va='center', fontweight='bold', color='#2b8a3e')
    ax1.text(0.5, 0.5, 'Hacker Attack = 50%', fontsize=32, fontweight='bold', ha='center')
    ax1.text(0.5, 0.3, 'Same as random coin flip', fontsize=22, ha='center', color='#2b8a3e')
    ax1.text(0.5, 0.18, 'Completely unguessable', fontsize=22, ha='center', color='#2b8a3e')
    
    accuracy_bg = '#dbe4ff'
    rect2 = plt.Rectangle((0.05, 0.05), 0.9, 0.9, facecolor=accuracy_bg, alpha=0.6)
    ax2.add_patch(rect2)
    ax2.text(0.5, 0.75, 'ACCURACY', fontsize=52, ha='center', va='center', fontweight='bold', color='#364fc7')
    ax2.text(0.5, 0.5, 'Diagnosis: 93.63%', fontsize=32, fontweight='bold', ha='center')
    ax2.text(0.5, 0.3, 'Clinically sufficient', fontsize=22, ha='center', color='#364fc7')
    ax2.text(0.5, 0.18, 'Nearly identical to traditional AI', fontsize=22, ha='center', color='#364fc7')
    
    efficiency_bg = '#e7f5ff'
    rect3 = plt.Rectangle((0.05, 0.05), 0.9, 0.9, facecolor=efficiency_bg, alpha=0.6)
    ax3.add_patch(rect3)
    ax3.text(0.5, 0.75, 'EFFICIENCY', fontsize=52, ha='center', va='center', fontweight='bold', color='#1864ab')
    ax3.text(0.5, 0.5, '99.7% Computation Saved', fontsize=32, fontweight='bold', ha='center')
    ax3.text(0.5, 0.3, 'Runs on portable devices', fontsize=22, ha='center', color='#1864ab')
    ax3.text(0.5, 0.18, 'Long battery life', fontsize=22, ha='center', color='#1864ab')
    
    plt.subplots_adjust(top=0.92, bottom=0.05, hspace=0.25)
    plt.savefig('核心优势总结.png', dpi=300, bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print('✅ 核心优势总结.png generated successfully!')

def main():
    print('🚀 Generating HemoSparse public-facing visualization charts...\n')
    
    create_privacy_comparison()
    create_accuracy_comparison()
    create_computational_efficiency()
    create_summary_infographic()
    
    print('\n✅ All charts generated successfully! Saved in current directory.')

if __name__ == '__main__':
    main()
