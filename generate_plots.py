import matplotlib.pyplot as plt
import numpy as np

# Plot 1: F1 Score Comparison
labels = ['HotpotQA', 'MuSiQue']
std_rag_f1 = [44.6, 31.2]
ircot_f1 = [53.9, 40.1]
hamh_rag_f1 = [63.0, 49.6]

x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(7, 4))
rects1 = ax.bar(x - width, std_rag_f1, width, label='Standard RAG', color='#bdc3c7')
rects2 = ax.bar(x, ircot_f1, width, label='IRCoT', color='#3498db')
rects3 = ax.bar(x + width, hamh_rag_f1, width, label='HAMH-RAG (Ours)', color='#2ecc71')

ax.set_ylabel('Token F1 Score')
ax.set_title('Performance Comparison on Multi-Hop QA Benchmarks')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.set_ylim(0, 80)
ax.grid(axis='y', linestyle='--', alpha=0.7)

for rects in [rects1, rects2, rects3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

fig.tight_layout()
plt.savefig('paper/hamhrag_f1_comparison.png', dpi=300, bbox_inches='tight')

# Plot 2: Latency Reduction
fig2, ax2 = plt.subplots(figsize=(7, 4))
routes = ['Vector Only', 'Vector First', 'Hybrid Parallel', 'Graph First (Fallback)']
p50 = [12, 15, 21, 38]
p90 = [18, 24, 35, 61]

x2 = np.arange(len(routes))
w2 = 0.35

rects4 = ax2.bar(x2 - w2/2, p50, w2, label='P50 (Median)', color='#9b59b6')
rects5 = ax2.bar(x2 + w2/2, p90, w2, label='P90', color='#e74c3c')

ax2.set_ylabel('Latency (ms)')
ax2.set_title('Retrieval Latency by Query Routing Strategy')
ax2.set_xticks(x2)
ax2.set_xticklabels(routes, rotation=15)
ax2.legend()
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for rects in [rects4, rects5]:
    for rect in rects:
        height = rect.get_height()
        ax2.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

fig2.tight_layout()
plt.savefig('paper/hamhrag_latency_chart.png', dpi=300, bbox_inches='tight')
